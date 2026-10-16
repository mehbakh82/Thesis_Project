#!/usr/bin/env python3
"""Barge-in detector v2 on independently labeled real interaction events.

See docs/BARGEIN_DETECTOR_V2_PROTOCOL.md. Stage ``develop`` trains every
candidate on the training sessions and chooses the model and threshold on the
validation sessions only. Stage ``final`` re-fits the frozen choice on
train+validation and evaluates the untouched test sessions exactly once.

Labels are Gemini two-pass agreed labels (independent automatic labels, not
human-verified). Features are energy, F0, voicing, ZCR, and MFCC statistics
over a decision window from 100 ms before to 480 ms after the speech onset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
import wave
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_s2s.bargein.features import FeatureConfig, frame_feature_matrix  # noqa: E402

RATE = 16_000
PRE_S = 0.10
POST_S = 0.48
CONTEXT_S = 1.00  # pre-onset reference for speaker-contrast features
SPLIT_SEED = "detector-v2-20261016"
POSITIVE = {"interrupt", "turn_change"}
NEGATIVE = {"backchannel", "noise", "none"}
BOOTSTRAP = 2000


def read_clip(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768.0


def split_of(session: str) -> str:
    bucket = int(hashlib.sha256(f"{SPLIT_SEED}:{session}".encode()).hexdigest(), 16) % 100
    return "test" if bucket < 15 else "val" if bucket < 30 else "train"


def onset_seconds(event: dict[str, Any]) -> float:
    base = float(event["onset_in_clip"])
    estimates = [
        p.get("onset_time_s")
        for p in (event.get("pass_a") or {}, event.get("pass_b") or {})
        if isinstance(p.get("onset_time_s"), (int, float))
    ]
    near = [float(e) for e in estimates if abs(float(e) - base) <= 0.5]
    return float(np.mean(near)) if near else base


def window(audio: np.ndarray, onset: float) -> np.ndarray:
    """Return [onset - CONTEXT_S, onset + POST_S]; the last PRE_S+POST_S is the decision window."""

    start = max(0, int((onset - CONTEXT_S) * RATE))
    end = min(audio.size, int((onset + POST_S) * RATE))
    segment = audio[start:end]
    want = int((CONTEXT_S + POST_S) * RATE)
    if segment.size < want:
        segment = np.concatenate([np.zeros(want - segment.size, np.float32), segment])
    return segment


def features(segment: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    split = int((CONTEXT_S - PRE_S) * RATE)
    reference = frame_feature_matrix(segment[: split], cfg)
    frames = frame_feature_matrix(segment[split:], cfg)  # decision window (n_frames, 30)
    half = max(1, frames.shape[0] // 2)
    ref_voiced = reference[reference[:, 3] > 0.5] if (reference[:, 3] > 0.5).any() else reference
    cur_voiced = frames[frames[:, 3] > 0.5] if (frames[:, 3] > 0.5).any() else frames
    contrast = np.concatenate(
        [
            cur_voiced.mean(axis=0) - ref_voiced.mean(axis=0),
            [
                float(np.linalg.norm(cur_voiced[:, 4:17].mean(0) - ref_voiced[:, 4:17].mean(0))),
                float(np.median(cur_voiced[:, 2]) - np.median(ref_voiced[:, 2])),
                float(frames[:, 0].mean() - reference[:, 0].mean()),
                float(reference[:, 3].mean()),
            ],
        ]
    )
    energy = frames[:, 0]
    voiced = frames[:, 3]
    slope = np.polyfit(np.arange(energy.size), energy, 1)[0] if energy.size > 1 else 0.0
    return np.concatenate(
        [
            frames.mean(axis=0),
            frames.std(axis=0),
            frames.max(axis=0),
            frames[:half].mean(axis=0),
            frames[half:].mean(axis=0),
            [voiced.mean(), float(slope), float(np.percentile(energy, 90) - np.percentile(energy, 10))],
            contrast,
        ]
    )


class Augmenter:
    """Train-only noise, room-response, and residual-echo augmentation."""

    def __init__(self, noise_dir: Path, rir_dir: Path, echo_dir: Path | None, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.noises = sorted(noise_dir.rglob("*.wav"))
        self.rirs = sorted(rir_dir.rglob("*.wav"))
        self.echoes = sorted(echo_dir.glob("*.wav")) if echo_dir and echo_dir.is_dir() else []

    def _load(self, path: Path, length: int) -> np.ndarray:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
            channels = handle.getnchannels()
        audio = data.reshape(-1, channels)[:, 0].astype(np.float32) / 32768.0
        if rate != RATE:
            from thesis_s2s.audio import resample

            audio = resample(audio, rate, RATE)
        if audio.size < length:
            audio = np.tile(audio, int(np.ceil(length / max(1, audio.size))))
        start = int(self.rng.integers(0, max(1, audio.size - length)))
        return audio[start : start + length]

    def __call__(self, segment: np.ndarray) -> np.ndarray:
        out = segment.copy()
        if self.rirs and self.rng.random() < 0.5:
            rir = self._load(self.rng.choice(self.rirs), int(0.3 * RATE))
            rir = rir / (np.max(np.abs(rir)) + 1e-9)
            out = np.convolve(out, rir)[: out.size]
            out *= np.max(np.abs(segment)) / (np.max(np.abs(out)) + 1e-9)
        power = np.mean(out**2) + 1e-9
        if self.noises:
            noise = self._load(self.rng.choice(self.noises), out.size)
            snr = self.rng.uniform(5.0, 20.0)
            noise *= np.sqrt(power / (np.mean(noise**2) + 1e-9) / 10 ** (snr / 10))
            out = out + noise
        if self.echoes and self.rng.random() < 0.7:
            echo = self._load(self.rng.choice(self.echoes), out.size)
            level = self.rng.uniform(-25.0, -10.0)
            echo *= np.sqrt(power / (np.mean(echo**2) + 1e-9) * 10 ** (level / 10))
            out = out + echo
        return np.clip(out, -1.0, 1.0).astype(np.float32)


def metrics(y: np.ndarray, p: np.ndarray, sessions: list[str], threshold: float) -> dict[str, Any]:
    pred = (p >= threshold).astype(int)

    def core(yy: np.ndarray, pp: np.ndarray) -> dict[str, float]:
        tp = int(((pp == 1) & (yy == 1)).sum())
        tn = int(((pp == 0) & (yy == 0)).sum())
        fp = int(((pp == 1) & (yy == 0)).sum())
        fn = int(((pp == 0) & (yy == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "accuracy": (tp + tn) / max(1, yy.size),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "false_accept_rate": fp / (fp + tn) if fp + tn else 0.0,
            "false_reject_rate": fn / (fn + tp) if fn + tp else 0.0,
            "balanced_accuracy": 0.5 * (recall + (tn / (tn + fp) if tn + fp else 0.0)),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    out: dict[str, Any] = core(y, pred)
    groups: dict[str, list[int]] = {}
    for index, session in enumerate(sessions):
        groups.setdefault(session, []).append(index)
    keys = sorted(groups)
    rng = random.Random(f"{SPLIT_SEED}:bootstrap")
    draws: dict[str, list[float]] = {"accuracy": [], "f1": []}
    for _ in range(BOOTSTRAP):
        idx = [i for _k in keys for i in groups[keys[rng.randrange(len(keys))]]]
        m = core(y[idx], pred[idx])
        draws["accuracy"].append(m["accuracy"])
        draws["f1"].append(m["f1"])
    for name, values in draws.items():
        values.sort()
        out[f"{name}_session_ci95"] = [values[int(0.025 * BOOTSTRAP)], values[int(0.975 * BOOTSTRAP) - 1]]
    out["events"] = int(y.size)
    out["sessions"] = len(keys)
    return out


def build_models() -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        ),
        "gbdt": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=300, l2_regularization=1.0,
            class_weight="balanced", random_state=0,
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                validation_fraction=0.15, max_iter=500, random_state=0,
            ),
        ),
    }


def energy_rule_scores(segments: list[np.ndarray]) -> np.ndarray:
    """Baseline: share of post-onset 20 ms frames above -35 dBFS."""

    scores = []
    for seg in segments:
        post = seg[int(CONTEXT_S * RATE) :]
        frames = post[: post.size // 320 * 320].reshape(-1, 320)
        level = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
        scores.append(float((level > -35.0).mean()) if frames.size else 0.0)
    return np.asarray(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("develop", "final"), required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--noise-dir", type=Path, required=True)
    parser.add_argument("--rir-dir", type=Path, required=True)
    parser.add_argument("--echo-dir", type=Path)
    parser.add_argument("--augment-copies", type=int, default=2)
    parser.add_argument("--choice", type=Path, help="develop receipt (required for final)")
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "final" and args.out.exists():
        raise SystemExit("refusing to overwrite the one-time final test receipt")

    events = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line]
    events = [e for e in events if e.get("agreed") and e.get("label") in POSITIVE | NEGATIVE]
    cfg = FeatureConfig()
    data: dict[str, dict[str, list[Any]]] = {
        s: {"x": [], "y": [], "seg": [], "session": [], "label": []} for s in ("train", "val", "test")
    }
    for event in events:
        clip = args.clips / f"{event['event_id']}.wav"
        if not clip.is_file():
            continue
        segment = window(read_clip(clip), onset_seconds(event))
        if segment.size < int((CONTEXT_S + POST_S) * RATE):
            continue
        split = split_of(event["session_id"])
        bucket = data[split]
        bucket["seg"].append(segment)
        bucket["y"].append(int(event["label"] in POSITIVE))
        bucket["session"].append(event["session_id"])
        bucket["label"].append(event["label"])
    augmenter = Augmenter(args.noise_dir, args.rir_dir, args.echo_dir, seed=20261016)

    def matrix(split: str, augment: bool) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for seg, y in zip(data[split]["seg"], data[split]["y"], strict=True):
            xs.append(features(seg, cfg))
            ys.append(y)
            if augment:
                for _ in range(args.augment_copies):
                    xs.append(features(augmenter(seg), cfg))
                    ys.append(y)
        return np.asarray(xs), np.asarray(ys)

    summary = {
        s: {
            "events": len(data[s]["y"]),
            "sessions": len(set(data[s]["session"])),
            "labels": dict(Counter(data[s]["label"])),
        }
        for s in data
    }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "stage": args.stage,
        "protocol": "docs/BARGEIN_DETECTOR_V2_PROTOCOL.md",
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "label_source": "gemini-3.8-flash two-pass agreed (independent automatic, not human)",
        "positive_classes": sorted(POSITIVE),
        "negative_classes": sorted(NEGATIVE),
        "decision_window_s": [-PRE_S, POST_S],
        "speaker_contrast_reference_s": [-CONTEXT_S, -PRE_S],
        "splits": summary,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.stage == "develop":
        from sklearn.model_selection import GroupKFold

        pool_seg = data["train"]["seg"] + data["val"]["seg"]
        pool_y = np.asarray(data["train"]["y"] + data["val"]["y"])
        pool_sessions = data["train"]["session"] + data["val"]["session"]
        clean = np.asarray([features(seg, cfg) for seg in pool_seg])
        augmented = [
            [features(augmenter(seg), cfg) for _ in range(args.augment_copies)] for seg in pool_seg
        ]
        folds = list(GroupKFold(n_splits=5).split(clean, pool_y, groups=pool_sessions))
        candidates: dict[str, Any] = {}
        for name in build_models():
            oof = np.zeros(pool_y.size)
            for train_idx, held_idx in folds:
                x_fit = np.vstack(
                    [clean[train_idx]] + [np.asarray([augmented[i][c] for i in train_idx]) for c in range(args.augment_copies)]
                )
                y_fit = np.concatenate([pool_y[train_idx]] * (1 + args.augment_copies))
                model = build_models()[name]
                model.fit(x_fit, y_fit)
                oof[held_idx] = model.predict_proba(clean[held_idx])[:, 1]
            best = max(
                (metrics(pool_y, oof, pool_sessions, t)["balanced_accuracy"], t)
                for t in np.linspace(0.2, 0.8, 25)
            )
            candidates[name] = {
                "threshold": float(best[1]),
                "cross_validation": metrics(pool_y, oof, pool_sessions, float(best[1])),
            }
        energy = energy_rule_scores(pool_seg)
        best_e = max(
            (metrics(pool_y, energy, pool_sessions, t)["balanced_accuracy"], t)
            for t in np.linspace(0.1, 0.9, 33)
        )
        candidates["energy_rule"] = {
            "threshold": float(best_e[1]),
            "cross_validation": metrics(pool_y, energy, pool_sessions, float(best_e[1])),
        }
        chosen = max(
            (n for n in candidates if n != "energy_rule"),
            key=lambda n: candidates[n]["cross_validation"]["balanced_accuracy"],
        )
        receipt.update(
            selection="session-grouped 5-fold CV on train+val, pooled out-of-fold balanced accuracy",
            candidates=candidates,
            chosen_model=chosen,
            chosen_threshold=candidates[chosen]["threshold"],
        )
    else:
        choice = json.loads(args.choice.read_text())
        name, threshold = choice["chosen_model"], float(choice["chosen_threshold"])
        x_tr, y_tr = matrix("train", augment=True)
        x_va, y_va = matrix("val", augment=True)
        model = build_models()[name]
        model.fit(np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va]))
        x_te, y_te = matrix("test", augment=False)
        p_te = model.predict_proba(x_te)[:, 1]
        receipt.update(
            chosen_model=name,
            chosen_threshold=threshold,
            develop_receipt_sha256=hashlib.sha256(args.choice.read_bytes()).hexdigest(),
            test=metrics(y_te, p_te, data["test"]["session"], threshold),
        )
        strict = [i for i, lab in enumerate(data["test"]["label"]) if lab in {"interrupt", "backchannel"}]
        if strict:
            ys = np.asarray([int(data["test"]["label"][i] == "interrupt") for i in strict])
            receipt["test_interrupt_vs_backchannel"] = metrics(
                ys, p_te[strict], [data["test"]["session"][i] for i in strict], threshold
            )
        energy = energy_rule_scores(data["test"]["seg"])
        receipt["test_energy_rule"] = metrics(
            y_te, energy, data["test"]["session"], float(choice["candidates"]["energy_rule"]["threshold"])
        )
        if args.model_out:
            args.model_out.parent.mkdir(parents=True, exist_ok=True)
            with args.model_out.open("wb") as handle:
                pickle.dump(
                    {
                        "model": model,
                        "threshold": threshold,
                        "feature_window_s": [-PRE_S, POST_S],
                        "reference_window_s": [-CONTEXT_S, -PRE_S],
                    },
                    handle,
                )
            receipt["model_sha256"] = hashlib.sha256(args.model_out.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt.get(k) for k in ("stage", "splits", "chosen_model", "chosen_threshold")}, ensure_ascii=False, indent=1))
    if args.stage == "develop":
        for name, cand in receipt["candidates"].items():
            v = cand["cross_validation"]
            print(f"{name:20s} val bal {v['balanced_accuracy']:.3f} acc {v['accuracy']:.3f} f1 {v['f1']:.3f} far {v['false_accept_rate']:.3f} frr {v['false_reject_rate']:.3f} thr {cand['threshold']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
