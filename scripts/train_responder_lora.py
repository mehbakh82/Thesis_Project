#!/usr/bin/env python3
"""Train the frozen compact Persian responder LoRA without emitting private text."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_cascade_real_service_v4 import atomic_write, qwen_revision  # noqa: E402
from thesis_s2s.metrics import gpu_inventory  # noqa: E402

SEED = 20260906
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE_MANIFEST_SHA256 = "aabf12268cce2854ef8b16ffd9339c6d703339c6b4c50a23a105447047a13a86"
TRAIN_EXPORT_SHA256 = "a84c9f1f924e8544499bd502413c469d30a10b98ceefc4e5a1f853217af0e517"
VAL_EXPORT_SHA256 = "a2762495830c82ce12d3dc4cc8469e2110ae21a77ea91f9b70498a6812e187a7"
EXPECTED_TRAIN_ROWS = 6419
EXPECTED_VAL_ROWS = 131
EXPECTED_DEV_ROWS = 39
EXPECTED_ELIGIBLE_TRAIN_ROWS = 2576
EXPECTED_ELIGIBLE_DEV_ROWS = 12
EXPECTED_TEST_ROWS = 204
SPLIT_SEED = "20260906-responder-lora-v1"
DEV_SESSION_HASHES = (
    "20b9c9f30f140385942bc1311733ef0fcaa7a892f83cbca5e04b308984886f27",
    "0fe76c73675252a81c2f14ce4dd1ed6e5fc96d5903130f945996c9ee671631ef",
)
SYSTEM_PROMPT = "فقط با خط فارسی و بدون هیچ حرف یا واژه لاتین، کوتاه و طبیعی پاسخ بده."
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
MAX_USER_TOKENS = 256
MAX_RESPONSE_TOKENS = 64
MAX_SEQUENCE_TOKENS = 512
MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
EPOCHS = 2
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_FRACTION = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def session_order_key(session_id: str) -> str:
    return sha256_text(SPLIT_SEED + "\0" + session_id)


def load_locked_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    counts = {"train": 0, "val": 0, "test": 0, "other": 0}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("source manifest must contain only JSON objects")
            split = str(row.get("split") or "")
            if split not in counts:
                counts["other"] += 1
                continue
            counts[split] += 1
            if split in {"train", "val"}:
                required = ("text", "response_text", "session_id", "utt_id")
                if not all(isinstance(row.get(key), str) and row[key].strip() for key in required):
                    raise ValueError("train/validation row lacks a required nonempty string")
                if row.get("training_use_authorized") is not True:
                    raise ValueError("row is not authorized for internal training")
                if row.get("internal_research_authorized") is not True:
                    raise ValueError("row is not authorized for internal research")
                if row.get("is_natural_dialogue") is not True:
                    raise ValueError("row is not marked as natural dialogue")
                if row.get("manual_qa_waived") is not True:
                    raise ValueError("row is outside the documented QA-waiver policy")
                (train if split == "train" else val).append(row)
    return train, val, counts


def split_development_rows(val_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sessions = sorted({str(row["session_id"]) for row in val_rows}, key=session_order_key)
    if len(sessions) != 5:
        raise ValueError("expected exactly five validation sessions")
    hashes = tuple(sha256_text(session) for session in sessions[:2])
    if hashes != DEV_SESSION_HASHES:
        raise ValueError("development-session hash drift")
    dev_sessions = set(sessions[:2])
    return [row for row in val_rows if str(row["session_id"]) in dev_sessions]


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [int(value) for value in encoded]


def first_response_sentence(text: str) -> str:
    normalized = " ".join(text.strip().split())
    match = re.search(r"[.!؟]", normalized)
    if match is not None:
        sentence = normalized[: match.end()].strip()
        if sentence:
            return sentence
    return normalized


def eligible_complete_response_rows(
    tokenizer: Any, rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if 0
        < len(
            _token_ids(
                tokenizer,
                first_response_sentence(str(row["response_text"])),
            )
        )
        <= MAX_RESPONSE_TOKENS
    ]


def encode_dialogue(tokenizer: Any, row: dict[str, Any]) -> dict[str, list[int]]:
    user_ids = _token_ids(tokenizer, str(row["text"]).strip())[-MAX_USER_TOKENS:]
    response_ids = _token_ids(tokenizer, first_response_sentence(str(row["response_text"])))[
        :MAX_RESPONSE_TOKENS
    ]
    if not user_ids or not response_ids:
        raise ValueError("tokenization produced an empty user or response")
    user_text = tokenizer.decode(user_ids, skip_special_tokens=True).strip()
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": response_text}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )["input_ids"]
    prefix = [int(value) for value in prefix]
    full = [int(value) for value in full]
    if full[: len(prefix)] != prefix:
        raise ValueError("chat template does not preserve the generation prefix")
    if len(full) > MAX_SEQUENCE_TOKENS:
        raise ValueError("locked token caps unexpectedly exceed maximum sequence length")
    labels = [-100] * len(prefix) + full[len(prefix) :]
    if not any(value != -100 for value in labels):
        raise ValueError("assistant-only labels are empty")
    return {"input_ids": full, "labels": labels}


def encode_rows(tokenizer: Any, rows: Iterable[dict[str, Any]]) -> list[dict[str, list[int]]]:
    return [encode_dialogue(tokenizer, row) for row in rows]


class DialogueDataset:
    def __init__(self, rows: list[dict[str, list[int]]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def collate_dialogues(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Any]:
    import torch

    width = max(len(row["input_ids"]) for row in batch)
    inputs: list[list[int]] = []
    labels: list[list[int]] = []
    masks: list[list[int]] = []
    for row in batch:
        padding = width - len(row["input_ids"])
        inputs.append(row["input_ids"] + [pad_token_id] * padding)
        labels.append(row["labels"] + [-100] * padding)
        masks.append([1] * len(row["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def evaluate_loss(model: Any, loader: Any, device: Any) -> tuple[float, int]:
    import torch

    model.eval()
    weighted_loss = 0.0
    target_tokens = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            count = int((batch["labels"] != -100).sum().item())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            weighted_loss += float(loss.item()) * count
            target_tokens += count
    model.train()
    if target_tokens <= 0:
        raise ValueError("development loader contains no assistant target tokens")
    return weighted_loss / target_tokens, target_tokens


def adapter_tree(path: Path) -> tuple[list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        file_hash = sha256_file(file_path)
        size = file_path.stat().st_size
        entries.append({"path": relative, "bytes": size, "sha256": file_hash})
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
        digest.update(b"\n")
    if not entries:
        raise ValueError("adapter directory is empty")
    return entries, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/processed/manifests/conversations.jsonl"),
    )
    parser.add_argument(
        "--train-export",
        type=Path,
        default=Path("data/processed/moshi_finetune/train.jsonl"),
    )
    parser.add_argument(
        "--val-export",
        type=Path,
        default=Path("data/processed/moshi_finetune/val.jsonl"),
    )
    parser.add_argument("--protocol", type=Path, default=Path("docs/RESPONDER_LORA_PROTOCOL.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/responder_lora_v1"))
    parser.add_argument(
        "--report", type=Path, default=Path("results/training/responder_lora_v1.json")
    )
    args = parser.parse_args()

    source_path = (ROOT / args.source_manifest).resolve()
    train_export_path = (ROOT / args.train_export).resolve()
    val_export_path = (ROOT / args.val_export).resolve()
    protocol_path = (ROOT / args.protocol).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    report_path = (ROOT / args.report).resolve()
    for required in (source_path, train_export_path, val_export_path, protocol_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_dir.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite responder training artifacts")
    locked_hashes = {
        "source_manifest": sha256_file(source_path),
        "train_export": sha256_file(train_export_path),
        "val_export": sha256_file(val_export_path),
    }
    if locked_hashes != {
        "source_manifest": SOURCE_MANIFEST_SHA256,
        "train_export": TRAIN_EXPORT_SHA256,
        "val_export": VAL_EXPORT_SHA256,
    }:
        raise RuntimeError("locked data hash drift")
    measured_revision = qwen_revision(BASE_MODEL)
    if measured_revision != BASE_REVISION:
        raise RuntimeError("base-model revision drift")

    train_rows, val_rows, split_counts = load_locked_rows(source_path)
    if split_counts != {
        "train": EXPECTED_TRAIN_ROWS,
        "val": EXPECTED_VAL_ROWS,
        "test": EXPECTED_TEST_ROWS,
        "other": 0,
    }:
        raise RuntimeError("source split-count drift")
    dev_rows = split_development_rows(val_rows)
    if len(train_rows) != EXPECTED_TRAIN_ROWS or len(dev_rows) != EXPECTED_DEV_ROWS:
        raise RuntimeError("training/development row-count drift")
    train_sessions = {str(row["session_id"]) for row in train_rows}
    dev_sessions = {str(row["session_id"]) for row in dev_rows}
    if train_sessions & dev_sessions:
        raise RuntimeError("train/development session leakage")

    import peft
    import torch
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("the frozen run requires CUDA BF16 training")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eligible_train_rows = eligible_complete_response_rows(tokenizer, train_rows)
    eligible_dev_rows = eligible_complete_response_rows(tokenizer, dev_rows)
    if (
        len(eligible_train_rows) != EXPECTED_ELIGIBLE_TRAIN_ROWS
        or len(eligible_dev_rows) != EXPECTED_ELIGIBLE_DEV_ROWS
    ):
        raise RuntimeError("complete-response eligibility-count drift")
    encoded_train = encode_rows(tokenizer, eligible_train_rows)
    encoded_dev = encode_rows(tokenizer, eligible_dev_rows)
    collator = lambda batch: collate_dialogues(batch, int(tokenizer.pad_token_id))  # noqa: E731
    generator = torch.Generator().manual_seed(SEED)
    train_dataset: Any = DialogueDataset(encoded_train)
    dev_dataset: Any = DialogueDataset(encoded_dev)
    train_loader: Any = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
    )
    dev_loader: Any = DataLoader(
        dev_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    model: Any = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to("cuda")  # type: ignore[arg-type]
    model.config.use_cache = False
    base_dev_loss, dev_target_tokens = evaluate_loss(model, dev_loader, torch.device("cuda"))
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=list(TARGET_MODULES),
        ),
    )
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = math.ceil(total_steps * WARMUP_FRACTION)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir.mkdir(parents=True)
    epoch_reports: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_weighted_loss = 0.0
        epoch_target_tokens = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to("cuda") for key, value in batch.items()}
            target_count = int((batch["labels"] != -100).sum().item())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            (loss / GRADIENT_ACCUMULATION).backward()
            epoch_weighted_loss += float(loss.item()) * target_count
            epoch_target_tokens += target_count
            if batch_index % GRADIENT_ACCUMULATION == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "optimizer_step": global_step,
                                "total_steps": total_steps,
                                "mean_train_loss_so_far": round(
                                    epoch_weighted_loss / epoch_target_tokens, 6
                                ),
                            }
                        ),
                        flush=True,
                    )
        dev_loss, measured_dev_tokens = evaluate_loss(model, dev_loader, torch.device("cuda"))
        if measured_dev_tokens != dev_target_tokens or not math.isfinite(dev_loss):
            raise RuntimeError("invalid development loss")
        epoch_dir = output_dir / f"epoch{epoch}"
        model.save_pretrained(epoch_dir, safe_serialization=True)
        files, tree_hash = adapter_tree(epoch_dir)
        epoch_reports.append(
            {
                "epoch": epoch,
                "optimizer_step": global_step,
                "train_assistant_token_loss": round(epoch_weighted_loss / epoch_target_tokens, 8),
                "train_assistant_tokens": epoch_target_tokens,
                "development_assistant_token_loss": round(dev_loss, 8),
                "development_assistant_tokens": dev_target_tokens,
                "adapter_relative_path": epoch_dir.relative_to(ROOT).as_posix(),
                "adapter_tree_sha256": tree_hash,
                "adapter_files": files,
            }
        )
        print(
            json.dumps(
                {
                    "epoch_complete": epoch,
                    "development_loss": round(dev_loss, 8),
                    "adapter_tree_sha256": tree_hash,
                }
            ),
            flush=True,
        )

    selected = min(epoch_reports, key=lambda row: row["development_assistant_token_loss"])
    validity = {
        "locked_data_hashes_match": locked_hashes
        == {
            "source_manifest": SOURCE_MANIFEST_SHA256,
            "train_export": TRAIN_EXPORT_SHA256,
            "val_export": VAL_EXPORT_SHA256,
        },
        "base_revision_exact": measured_revision == BASE_REVISION,
        "split_counts_exact": split_counts
        == {
            "train": EXPECTED_TRAIN_ROWS,
            "val": EXPECTED_VAL_ROWS,
            "test": EXPECTED_TEST_ROWS,
            "other": 0,
        },
        "train_rows_exact": len(train_rows) == EXPECTED_TRAIN_ROWS,
        "development_rows_exact": len(dev_rows) == EXPECTED_DEV_ROWS,
        "eligible_training_rows_exact": len(eligible_train_rows) == EXPECTED_ELIGIBLE_TRAIN_ROWS,
        "eligible_development_rows_exact": len(eligible_dev_rows) == EXPECTED_ELIGIBLE_DEV_ROWS,
        "train_development_sessions_disjoint": not bool(train_sessions & dev_sessions),
        "all_loaded_rows_training_authorized": all(
            row.get("training_use_authorized") is True
            and row.get("internal_research_authorized") is True
            for row in [*train_rows, *dev_rows]
        ),
        "test_rows_not_in_loaders": all(
            row.get("split") in {"train", "val"} for row in [*train_rows, *dev_rows]
        ),
        "exactly_two_epochs_completed": len(epoch_reports) == EPOCHS and global_step == total_steps,
        "all_losses_finite": all(
            math.isfinite(float(value))
            for value in [
                base_dev_loss,
                *(
                    item[key]
                    for item in epoch_reports
                    for key in (
                        "train_assistant_token_loss",
                        "development_assistant_token_loss",
                    )
                ),
            ]
        ),
        "selected_adapter_tree_hashed": bool(selected["adapter_tree_sha256"]),
    }
    valid = all(validity.values())
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if valid else "invalid",
        "evidence_class": "automatic_supervised_compact_responder_lora_training",
        "claim": {
            "training_completed": valid,
            "checkpoint_selected_by_development_loss_only": valid,
            "semantic_improvement_measured": False,
            "human_quality_result": False,
            "test_result": False,
        },
        "data": {
            "source_train_rows": len(train_rows),
            "source_development_rows": len(dev_rows),
            "eligible_train_rows": len(eligible_train_rows),
            "eligible_development_rows": len(eligible_dev_rows),
            "eligibility_rule": "first_nonempty_sentence_at_most_64_base_tokens",
            "source_split_counts": split_counts,
            "train_sessions": len(train_sessions),
            "development_sessions": len(dev_sessions),
            "development_session_sha256": list(DEV_SESSION_HASHES),
            "human_verified": False,
            "manual_qa_waived": True,
            "internal_training_authorized": True,
            "source_license_verified": False,
            "redistribution_allowed": False,
        },
        "base": {"model": BASE_MODEL, "revision": measured_revision},
        "optimization": {
            "seed": SEED,
            "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
            "assistant_only_loss": True,
            "max_user_tokens": MAX_USER_TOKENS,
            "max_response_tokens": MAX_RESPONSE_TOKENS,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": list(TARGET_MODULES),
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch_size": MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_fraction": WARMUP_FRACTION,
            "warmup_steps": warmup_steps,
            "optimizer_steps": global_step,
            "trainable_parameters": trainable_parameters,
            "total_parameters_with_adapter": total_parameters,
        },
        "base_development_assistant_token_loss": round(base_dev_loss, 8),
        "epochs": epoch_reports,
        "selection": {
            "rule": "lowest_finite_development_assistant_token_loss",
            "selected_epoch": selected["epoch"],
            "selected_adapter_relative_path": selected["adapter_relative_path"],
            "selected_adapter_tree_sha256": selected["adapter_tree_sha256"],
            "development_loss_reduction_fraction_vs_base": round(
                (base_dev_loss - float(selected["development_assistant_token_loss"]))
                / base_dev_loss,
                8,
            ),
        },
        "validity_requirements": validity,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu_inventory(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_seconds": round(time.time() - started, 3),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "artifacts": {
            **{key + "_sha256": value for key, value in locked_hashes.items()},
            "protocol_sha256": sha256_file(protocol_path),
            "trainer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "limitations": [
            "Dialogue targets are automatically aligned and not manually verified.",
            "Development loss is not evidence of response relevance or human quality.",
            "The source license is unverified; adapters remain local and unredistributed.",
            "The test split is unused, but combined-manifest metadata was audited before freeze.",
        ],
        "privacy": {
            "plaintext_user_text_stored": False,
            "plaintext_response_text_stored": False,
            "audio_stored": False,
            "private_source_path_stored": False,
            "adapter_committed_or_uploaded": False,
        },
    }
    atomic_write(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_epoch": selected["epoch"],
                "base_dev_loss": report["base_development_assistant_token_loss"],
                "selected_dev_loss": selected["development_assistant_token_loss"],
                "loss_reduction_fraction": report["selection"][
                    "development_loss_reduction_fraction_vs_base"
                ],
                "report": report_path.relative_to(ROOT).as_posix(),
            }
        )
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
