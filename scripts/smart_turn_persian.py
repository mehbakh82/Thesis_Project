#!/usr/bin/env python3
"""Stage 2 of the end-of-turn experiment: Smart Turn v3.2 fine-tuned on Persian.

Protocol: ``docs/END_OF_TURN_PROTOCOL.md`` (stage 2). Run with an interpreter
that has torch, transformers and onnx.

``convert`` rebuilds the PyTorch model of the public Smart Turn training code
(``SmartTurnV3Model``, BSD-2, pipecat-ai/smart-turn) from the published
full-precision ONNX checkpoint and checks that both give the same output.

``mine`` writes Persian examples from the training-split conversation pairs
only (sessions of the development clips and of every loopback test panel
excluded): *incomplete* = a diarized user turn cut at an internal pause of at
least 100 ms after which the same speaker continues; *complete* = the turn up
to its end, where the other speaker takes over. Labels come from diarization
and a VAD, not from people.

``train`` fine-tunes from the converted checkpoint with a session-held-out
validation split (the model class, loss and optimizer settings of the public
training code; batch 64 for the smaller data set); ``export`` writes the best
epoch to ONNX for the live gate (needs ``onnx``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RATE = 16_000
MAX_SAMPLES = 8 * RATE
SEED = 20261020


def build_model() -> Any:
    import torch
    from torch import nn
    from transformers import WhisperConfig, WhisperPreTrainedModel
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    class SmartTurnV3Model(WhisperPreTrainedModel):  # type: ignore[misc]
        """``SmartTurnV3Model`` of pipecat-ai/smart-turn ``train.py`` (BSD-2-Clause, Daily)."""

        def __init__(self, config: WhisperConfig) -> None:
            super().__init__(config)
            config.max_source_positions = 400
            self.encoder = WhisperEncoder(config)
            hidden = config.d_model
            self.pool_attention = nn.Sequential(nn.Linear(hidden, 256), nn.Tanh(), nn.Linear(256, 1))
            self.classifier = nn.Sequential(
                nn.Linear(hidden, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
            )

        def forward(self, input_features: Any) -> Any:
            hidden = self.encoder(input_features=input_features).last_hidden_state
            weights = torch.softmax(self.pool_attention(hidden), dim=1)
            return self.classifier(torch.sum(hidden * weights, dim=1))  # logits

    config = WhisperConfig.from_pretrained("openai/whisper-tiny")
    return SmartTurnV3Model(config)


def extract(args: argparse.Namespace) -> int:
    """Read the ONNX initializers (needs ``onnx``) into an .npz keyed by PyTorch parameter name.

    Linear weights are stored transposed as anonymous MatMul inputs; they are
    recovered from the name of the node that uses them.
    """

    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(args.onnx)).graph
    weights = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    out = {name.removeprefix("inner."): value for name, value in weights.items() if name.startswith("inner.")}
    consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)

    def bias_of(tensor: str, hops: int = 2) -> str | None:
        # the named bias added to this MatMul output (directly or after a scaling Mul)
        for node in consumers.get(tensor, []):
            named = [t for t in node.input if t.startswith("inner.") and t.endswith(".bias")]
            if node.op_type == "Add" and named:
                return named[0]
            if node.op_type == "Mul" and hops > 1:
                found = bias_of(node.output[0], hops - 1)
                if found:
                    return found
        return None

    unbiased: dict[str, list[str]] = {}  # input tensor -> anonymous weights without a bias (k_proj)
    siblings: dict[str, list[str]] = {}
    for node in graph.node:
        if node.op_type not in {"MatMul", "Gemm"}:
            continue
        anonymous = [t for t in node.input if t in weights and not t.startswith("inner.")]
        if not anonymous:
            continue
        source = node.input[0]
        bias = bias_of(node.output[0])
        if bias is not None:
            key = bias.removeprefix("inner.").removesuffix(".bias") + ".weight"
            out[key] = weights[anonymous[0]].T
            siblings.setdefault(source, []).append(key)
        else:
            unbiased.setdefault(source, []).append(anonymous[0])
    for source, tensors in unbiased.items():
        layer = {k.rsplit(".", 2)[0] for k in siblings.get(source, []) if ".self_attn." in k}
        if len(tensors) != 1 or len(layer) != 1:
            raise SystemExit(f"cannot place unbiased weights {tensors} on input {source}")
        out[f"{layer.pop()}.k_proj.weight"] = weights[tensors[0]].T
    np.savez(args.out, **out)
    print(json.dumps({"tensors": len(out)}))
    return 0


def convert(args: argparse.Namespace) -> int:
    """Load the extracted weights into the training model and check it against the ONNX model."""

    import onnxruntime as ort
    import torch

    model = build_model().eval()
    state = model.state_dict()
    weights = dict(np.load(args.weights))
    loaded = {k: v for k, v in weights.items() if k in state and tuple(state[k].shape) == v.shape}
    missing = sorted(set(state) - set(loaded))
    if missing:
        raise SystemExit(f"unmapped parameters: {missing[:10]} ({len(missing)})")
    model.load_state_dict({k: torch.tensor(v) for k, v in loaded.items()})
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    features = rng.standard_normal((4, 80, 800)).astype(np.float32)
    reference = session.run(None, {"input_features": features})[0].reshape(-1)
    with torch.no_grad():
        ours = torch.sigmoid(model(torch.tensor(features))).numpy().reshape(-1)
    error = float(np.abs(ours - reference).max())
    print(json.dumps({"parameters": len(loaded), "max_abs_diff_vs_onnx": error}))
    if error > 1e-4:
        raise SystemExit("converted model does not match the ONNX checkpoint")
    torch.save(model.state_dict(), args.out)
    return 0


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        if handle.getnchannels() > 1:
            frames = frames.reshape(-1, handle.getnchannels()).mean(axis=1)
    return frames.astype(np.float32) / 32768.0


def voiced_frames(audio: np.ndarray) -> np.ndarray:
    from thesis_s2s.runtime.vad import FRAME_SAMPLES, default_scorer

    scorer = default_scorer()
    count = audio.size // FRAME_SAMPLES
    return np.array([scorer.score(audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]) for i in range(count)])


def mine(args: argparse.Namespace) -> int:
    """Examples from the user channel of the training-split conversation pairs.

    Each pair's user channel holds one speaker's turn that ends where the other
    speaker takes over (the pair's response). Both classes are cut at the last
    voiced frame before a pause of at least 100 ms (protocol amendment 1).
    *complete*: the end of the turn, kept only if the speaker stopped naturally
    (at least 100 ms of non-silent audio follows); *incomplete*: a randomly
    chosen internal pause (at least 1 s after the start and 0.3 s before the
    end), after which the same speaker continues.
    """

    from thesis_s2s.runtime.vad import FRAME_SAMPLES

    excluded = set(json.loads(args.exclude_sessions.read_text(encoding="utf-8")))
    records = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line]
    rng = random.Random(SEED)
    rng.shuffle(records)
    frame_s = FRAME_SAMPLES / RATE
    out_dir = args.work / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    examples: list[dict[str, Any]] = []
    skipped = {"excluded_session": 0, "no_natural_end": 0, "no_internal_pause": 0, "too_short": 0}
    for record in records:
        if len(examples) >= 2 * args.max_pairs:
            break
        audio_path = Path(record["path"])
        meta = json.loads(audio_path.with_suffix(".json").read_text(encoding="utf-8"))
        session = meta["source_session_id"]
        if session in excluded:
            skipped["excluded_session"] += 1
            continue
        with wave.open(str(audio_path), "rb") as handle:
            stereo = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").reshape(-1, 2)
        user = stereo[:, 1].astype(np.float32) / 32768.0  # channel 1 is the user stream
        # amendment 1: cut both classes where a pause begins, as the live gate sees audio
        voiced = voiced_frames(user) >= 0.35
        if voiced.sum() * frame_s < 1.5:
            skipped["too_short"] += 1
            continue
        frames = user[: user.size // FRAME_SAMPLES * FRAME_SAMPLES].reshape(-1, FRAME_SAMPLES)
        present = np.sqrt((frames**2).mean(axis=1)) > 1e-4  # not the digital silence after the cut
        last = int(np.flatnonzero(voiced)[-1]) + 1
        content_end = int(np.flatnonzero(present)[-1]) + 1 if present.any() else 0
        if (content_end - last) * frame_s < 0.1:
            skipped["no_natural_end"] += 1  # the speaker was still voicing where the turn was cut
            continue
        end = last * frame_s
        pauses, run = [], 0
        for f in range(int(np.flatnonzero(voiced)[0]), last):
            if voiced[f]:
                cut = (f - run) * frame_s
                if run * frame_s >= 0.1 and cut >= 1.0 and cut <= end - 0.3:
                    pauses.append(cut)
                run = 0
            else:
                run += 1
        if not pauses:
            skipped["no_internal_pause"] += 1
            continue
        for label, cut in (("complete", end), ("incomplete", rng.choice(pauses))):
            clip = user[max(0, int(cut * RATE) - MAX_SAMPLES) : int(cut * RATE)]
            name = hashlib.sha256(f"{audio_path.name}|{label}|{cut:.3f}".encode()).hexdigest()[:16]
            np.save(out_dir / f"{name}.npy", (clip * 32767).astype(np.int16))
            examples.append({"id": name, "label": label, "session_id": session, "seconds": round(clip.size / RATE, 3)})
    sessions = sorted({e["session_id"] for e in examples})
    rng.shuffle(sessions)
    val = set(sessions[: max(1, len(sessions) // 10)])
    for e in examples:
        e["fold"] = "val" if e["session_id"] in val else "train"
    (args.work / "examples.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8"
    )
    summary = {
        fold: {lab: sum(e["fold"] == fold and e["label"] == lab for e in examples) for lab in ("complete", "incomplete")}
        for fold in ("train", "val")
    }
    print(json.dumps({"pairs_seen": sum(skipped.values()) + len(examples) // 2, "skipped": skipped,
                      "sessions": len(sessions), "examples": summary}, ensure_ascii=False))
    return 0


def train(args: argparse.Namespace) -> int:
    import torch

    from thesis_s2s.runtime.turn_end import smart_turn_features

    torch.manual_seed(SEED)
    from transformers import WhisperFeatureExtractor

    mel = WhisperFeatureExtractor(chunk_length=8).mel_filters
    examples = [json.loads(line) for line in (args.work / "examples.jsonl").read_text(encoding="utf-8").splitlines()]

    def tensors(fold: str) -> tuple[Any, Any]:
        rows = [e for e in examples if e["fold"] == fold]
        feats = np.concatenate(
            [smart_turn_features(np.load(args.work / "examples" / f"{e['id']}.npy").astype(np.float32) / 32767, mel)
             for e in rows]
        )
        labels = np.array([e["label"] == "complete" for e in rows], dtype=np.float32)
        return torch.tensor(feats), torch.tensor(labels)

    x_train, y_train = tensors("train")
    x_val, y_val = tensors("val")
    model = build_model()
    model.load_state_dict(torch.load(args.init, map_location="cpu"))
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    steps_per_epoch = (len(x_train) + args.batch - 1) // args.batch
    total = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, int(0.2 * total))) * max(0.0, (total - s) / total) * 1.0
    )

    def evaluate() -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            probs = torch.cat([torch.sigmoid(model(x_val[i : i + 128].cuda())).cpu().view(-1)
                               for i in range(0, len(x_val), 128)])
        pred = probs >= 0.5
        truth = y_val >= 0.5
        tp = int((pred & truth).sum())
        tn = int((~pred & ~truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return {"accuracy": (tp + tn) / max(1, len(y_val)), "f1": 2 * precision * recall / max(1e-9, precision + recall),
                "complete_recall": recall, "incomplete_recall": tn / max(1, tn + fp)}

    history = [{"epoch": 0, **evaluate()}]
    print(json.dumps(history[0]), flush=True)
    best = (history[0]["f1"], 0)
    torch.save(model.state_dict(), args.work / "best.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            x, y = x_train[idx].cuda(), y_train[idx].cuda()
            pos_weight = ((y == 0).sum() / (y == 1).sum().clamp(min=1)).clamp(min=0.1, max=10.0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).view(-1), y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        history.append({"epoch": epoch, **evaluate()})
        print(json.dumps(history[-1]), flush=True)
        if history[-1]["f1"] > best[0]:
            best = (history[-1]["f1"], epoch)
            torch.save(model.state_dict(), args.work / "best.pt")
    receipt = {
        "init_sha256": hashlib.sha256(args.init.read_bytes()).hexdigest(),
        "examples_sha256": hashlib.sha256((args.work / "examples.jsonl").read_bytes()).hexdigest(),
        "train_examples": len(x_train), "val_examples": len(x_val),
        "epochs": args.epochs, "batch": args.batch, "learning_rate": 5e-5, "warmup_ratio": 0.2,
        "weight_decay": 0.01, "history": history, "best_epoch": best[1],
        "best_sha256": hashlib.sha256((args.work / "best.pt").read_bytes()).hexdigest(),
    }
    (args.work / "training.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best[1], "best_val_f1": best[0]}))
    return 0


def export(args: argparse.Namespace) -> int:
    """Export the best epoch to ONNX (needs ``onnx``) and check it against PyTorch."""

    import torch

    model = build_model()
    model.load_state_dict(torch.load(args.work / "best.pt", map_location="cpu"))
    model.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_features: Any) -> Any:
            return torch.sigmoid(self.inner(input_features))

    wrapper = Wrapper(model).eval()
    example = torch.randn(1, 80, 800)
    torch.onnx.export(wrapper, (example,), str(args.out), input_names=["input_features"], output_names=["logits"],
                      dynamic_axes={"input_features": {0: "batch"}}, opset_version=18, dynamo=False)
    import onnx
    from onnx.reference import ReferenceEvaluator

    features = np.random.default_rng(1).standard_normal((2, 80, 800)).astype(np.float32)
    reference = ReferenceEvaluator(onnx.load(str(args.out))).run(None, {"input_features": features})[0]
    with torch.no_grad():
        ours = wrapper(torch.tensor(features)).numpy()
    error = float(np.abs(np.asarray(reference) - ours).max())
    receipt_path = args.work / "training.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["onnx_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    receipt["onnx_max_abs_diff_vs_torch"] = error
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"onnx": str(args.out), "max_abs_diff_vs_torch": error}))
    if error > 1e-4:
        raise SystemExit("exported model does not match PyTorch")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("convert")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("mine")
    p.add_argument("--pairs", type=Path, required=True, help="training-split pair list (moshi_finetune/train.jsonl)")
    p.add_argument("--exclude-sessions", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--max-pairs", type=int, default=6000)
    p = sub.add_parser("train")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p = sub.add_parser("export")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return {"extract": extract, "convert": convert, "mine": mine, "train": train, "export": export}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
