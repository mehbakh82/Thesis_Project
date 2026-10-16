#!/usr/bin/env python3
"""LoRA-adapt Gemma 4 E4B for Persian speech input (ASR + spoken-request replies).

Trains LoRA adapters on the language-model attention/MLP projections while the
audio encoder stays frozen. Only assistant tokens contribute to the loss. A
session-held-out slice of the training manifest gives a validation loss; no
evaluation panel or test audio is used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_responder_candidates_v2 import ELDERLY_PROMPT_V1  # noqa: E402

ASR_PROMPT = (
    "Transcribe this Persian (Farsi) speech verbatim in Persian script. "
    "Output only the transcription, with no translation or commentary."
)
RESPOND_PROMPT = "به گفتهٔ صوتی کاربر پاسخ بده."


def read_audio(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def messages_for(example: dict[str, Any], audio: np.ndarray, with_target: bool) -> list[dict[str, Any]]:
    if example["task"] == "asr":
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": ASR_PROMPT}]}
        ]
    else:
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": ELDERLY_PROMPT_V1}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": RESPOND_PROMPT}]},
        ]
    if with_target:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": example["target"]}]})
    return msgs


def encode(processor: Any, example: dict[str, Any]) -> dict[str, Any]:
    audio = read_audio(example["audio_filepath"])
    full = processor.apply_chat_template(
        messages_for(example, audio, True), tokenize=True, return_dict=True, return_tensors="pt"
    )
    prompt = processor.apply_chat_template(
        messages_for(example, audio, False),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    labels = full["input_ids"].clone()
    labels[:, : prompt["input_ids"].shape[-1]] = -100
    full["labels"] = labels
    return {k: v[0] for k, v in full.items()}


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    import torch

    length = max(item["input_ids"].shape[0] for item in batch)
    frames = max(item["input_features"].shape[0] for item in batch)
    out: dict[str, Any] = {}
    out["input_ids"] = torch.stack([torch.nn.functional.pad(i["input_ids"], (0, length - i["input_ids"].shape[0]), value=pad_id) for i in batch])
    out["attention_mask"] = torch.stack([torch.nn.functional.pad(i["attention_mask"], (0, length - i["attention_mask"].shape[0])) for i in batch])
    out["labels"] = torch.stack([torch.nn.functional.pad(i["labels"], (0, length - i["labels"].shape[0]), value=-100) for i in batch])
    out["mm_token_type_ids"] = torch.stack([torch.nn.functional.pad(i["mm_token_type_ids"], (0, length - i["mm_token_type_ids"].shape[0])) for i in batch])
    out["input_features"] = torch.stack([torch.nn.functional.pad(i["input_features"], (0, 0, 0, frames - i["input_features"].shape[0])) for i in batch])
    out["input_features_mask"] = torch.stack([torch.nn.functional.pad(i["input_features_mask"], (0, frames - i["input_features_mask"].shape[0])) for i in batch])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--val-share", type=int, default=5, help="percent of sessions held out")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    examples = [json.loads(line) for line in args.train.read_text(encoding="utf-8").splitlines() if line]

    def held_out(session: str) -> bool:
        return int(hashlib.sha256(f"gemma-speech:{session}".encode()).hexdigest(), 16) % 100 < args.val_share

    train = [e for e in examples if not held_out(e["session_id"])]
    val = [e for e in examples if held_out(e["session_id"])][:200]
    processor = AutoProcessor.from_pretrained(str(args.model))
    model = AutoModelForImageTextToText.from_pretrained(str(args.model), dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    targets = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(train) * args.epochs / (args.batch * args.accumulate))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / 30) * max(0.05, 1 - s / max(1, steps))
    )
    pad_id = processor.tokenizer.pad_token_id or 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = (args.out_dir / "metrics.jsonl").open("a", encoding="utf-8")

    def validation_loss() -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), args.batch):
                batch = collate([encode(processor, e) for e in val[i : i + args.batch]], pad_id)
                batch = {k: v.cuda() for k, v in batch.items()}
                batch["input_features"] = batch["input_features"].to(torch.bfloat16)
                total += float(model(**batch).loss) * len(val[i : i + args.batch])
                count += len(val[i : i + args.batch])
        model.train()
        return total / max(1, count)

    model.train()
    order = list(range(len(train)))
    started = time.time()
    step = 0
    base_val = validation_loss()
    log.write(json.dumps({"step": 0, "val_loss": base_val}) + "\n")
    log.flush()
    print(json.dumps({"train_examples": len(train), "val_examples": len(val), "trainable_parameters": trainable, "steps": steps, "val_loss_step0": base_val}), flush=True)
    micro = 0
    while step < steps:
        random.shuffle(order)
        for start in range(0, len(order), args.batch):
            batch = collate([encode(processor, train[i]) for i in order[start : start + args.batch]], pad_id)
            batch = {k: v.cuda() for k, v in batch.items()}
            batch["input_features"] = batch["input_features"].to(torch.bfloat16)
            loss = model(**batch).loss / args.accumulate
            loss.backward()
            micro += 1
            if micro % args.accumulate:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            record = {"step": step, "loss": float(loss) * args.accumulate, "lr": scheduler.get_last_lr()[0], "elapsed_s": round(time.time() - started, 1)}
            if step % args.eval_every == 0 or step == steps:
                record["val_loss"] = validation_loss()
                model.save_pretrained(str(args.out_dir / f"step_{step:05d}"))
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 10 == 0:
                print(json.dumps(record), flush=True)
            if step >= steps:
                break
    summary = {
        "steps": step,
        "val_loss_step0": base_val,
        "train_manifest_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "trainable_parameters": trainable,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
