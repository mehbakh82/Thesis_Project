#!/usr/bin/env python3
"""Deployment-faithful barge-in simulation from real recordings.

Each scenario is the microphone signal during one assistant reply: the reply
itself as residual echo after acoustic echo cancellation, quiet background
noise, and one user event at an early, middle, or late point of the reply:

* ``interrupt``  – a real single-speaker Persian speech segment (1.2-4 s);
* ``backchannel`` – a real isolated listener token (ASR-lexicon verified);
* ``noise``      – a real non-speech sound burst (point-source noise set);
* ``none``       – nothing.

The deployed two-stage controller (`BargeInController`, Silero VAD) runs on
20 ms frames. ``stop`` is the positive decision; accuracy/F1 treat
``interrupt`` as positive and every other event as negative. Sources are split
by session (speech, backchannels) or file (noise, replies) so the optional
energy/F0/MFCC MLP confirm-classifier is trained and tested on disjoint audio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import statistics
import sys
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_s2s.bargein.features import FeatureConfig, frame_feature_matrix  # noqa: E402
from thesis_s2s.runtime.bargein_live import BargeInController  # noqa: E402
from thesis_s2s.runtime.vad import SileroScorer, default_scorer  # noqa: E402

RATE = 16_000
FRAME = 320
CLASSES = ("interrupt", "backchannel", "noise", "none")
TIMINGS = ("early", "mid", "late")
SPLIT_SEED = "bargein-sim-20261016"
USER_RMS_DB = -23.0


def read(path: Path | str) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        channels = handle.getnchannels()
    audio = data.reshape(-1, channels)[:, 0].astype(np.float32) / 32768.0
    if rate != RATE:
        from thesis_s2s.audio import resample

        audio = resample(audio, rate, RATE)
    return audio


def is_test(key: str) -> bool:
    return int(hashlib.sha256(f"{SPLIT_SEED}:{key}".encode()).hexdigest(), 16) % 100 < 35


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9)


def at_db(x: np.ndarray, target_db: float) -> np.ndarray:
    return x * 10 ** ((target_db - rms_db(x)) / 20)


def window_features(window: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    frames = frame_feature_matrix(window, cfg)
    half = max(1, frames.shape[0] // 2)
    return np.concatenate(
        [
            frames.mean(0), frames.std(0), frames.max(0),
            frames[:half].mean(0), frames[half:].mean(0), [frames[:, 3].mean()],
        ]
    )


class Pools:
    def __init__(self, pool: Path, noise_dir: Path, reply_dir: Path, rir_dir: Path) -> None:
        rows = [json.loads(line) for line in pool.read_text(encoding="utf-8").splitlines() if line]
        self.items: dict[tuple[str, str], list[Any]] = {}
        for row in rows:
            split = "test" if is_test(row["session_id"]) else "train"
            kind = "interrupt" if row["kind"] == "speech" else "backchannel"
            self.items.setdefault((split, kind), []).append(row["audio_filepath"])
        for path in sorted(noise_dir.rglob("*.wav")):
            split = "test" if is_test(path.name) else "train"
            self.items.setdefault((split, "noise"), []).append(str(path))
        for path in sorted(reply_dir.glob("*.wav")):
            split = "test" if is_test(path.name) else "train"
            self.items.setdefault((split, "reply"), []).append(str(path))
        self.rirs = sorted(rir_dir.rglob("*.wav"))

    def counts(self) -> dict[str, int]:
        return {f"{s}/{k}": len(v) for (s, k), v in sorted(self.items.items())}


def noise_burst(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    length = int(rng.uniform(0.2, 1.2) * RATE)
    if audio.size <= length:
        return audio
    frames = audio[: audio.size // FRAME * FRAME].reshape(-1, FRAME)
    loud = int(np.argmax((frames**2).mean(1)))
    start = max(0, min(audio.size - length, loud * FRAME - length // 3))
    return audio[start : start + length]


def make_scenario(pools: Pools, split: str, kind: str, timing: str, rng: random.Random) -> dict[str, Any]:
    reply = read(rng.choice(pools.items[(split, "reply")]))
    user_db = USER_RMS_DB + rng.uniform(-4, 4)
    mic = np.zeros(reply.size + int(2.5 * RATE), np.float32)
    echo_db = user_db + rng.uniform(-32, -18)
    mic[: reply.size] += at_db(reply, echo_db)
    background = read(rng.choice(pools.items[(split, "noise")]))
    background = np.tile(background, int(np.ceil(mic.size / max(1, background.size))))[: mic.size]
    mic += at_db(background, user_db + rng.uniform(-38, -24))
    reply_s = reply.size / RATE
    onset_s = {
        "early": rng.uniform(0.3, 0.9),
        "mid": rng.uniform(0.4, 0.6) * reply_s,
        "late": max(0.3, reply_s - rng.uniform(0.4, 1.0)),
    }[timing]
    event_s = 0.0
    if kind != "none":
        source = read(rng.choice(pools.items[(split, kind)]))
        if kind == "noise":
            source = noise_burst(source, rng)
            level = user_db + rng.uniform(-6, 3)
        else:
            level = user_db
        if pools.rirs and rng.random() < 0.5:
            rir = read(rng.choice(pools.rirs))[: int(0.3 * RATE)]
            source = np.convolve(source, rir / (np.max(np.abs(rir)) + 1e-9))[: source.size]
        source = at_db(source, level)
        start = int(onset_s * RATE)
        end = min(mic.size, start + source.size)
        mic[start:end] += source[: end - start]
        event_s = source.size / RATE
    return {
        "kind": kind,
        "timing": timing,
        "mic": np.clip(mic, -1, 1).astype(np.float32),
        "onset_s": onset_s,
        "event_s": event_s,
        "reply_s": reply_s,
    }


def run_controller(scenario: dict[str, Any], classifier: Any | None, threshold: float) -> dict[str, Any]:
    controller = BargeInController(
        scorer=default_scorer(),
        classifier=(lambda w: float(classifier.predict_proba(window_features(w, CFG)[None])[0, 1]))
        if classifier is not None
        else None,
        classifier_threshold=threshold,
    )
    events = []
    mic = scenario["mic"]
    for start in range(0, mic.size - FRAME + 1, FRAME):
        events.extend(controller.push(mic[start : start + FRAME], sample_offset=start if start == 0 else None))
    onset = scenario["onset_s"]
    in_event = [e for e in events if e.onset_sample / RATE >= onset - 0.25]
    early_ducks = [e for e in events if e.kind == "duck" and e.onset_sample / RATE < onset - 0.25]
    stop = next((e for e in in_event if e.kind == "stop"), None)
    duck = next((e for e in in_event if e.kind == "duck"), None)
    if scenario["kind"] == "none":
        stop = next((e for e in events if e.kind == "stop"), None)
    return {
        "stopped": stop is not None,
        "ducked": duck is not None,
        "false_ducks_before_event": len(early_ducks),
        "duck_latency_ms": 1000 * (duck.decided_at_sample / RATE - onset) if duck else None,
        "stop_latency_ms": 1000 * (stop.decided_at_sample / RATE - onset) if stop else None,
        "detected_onset_error_ms": 1000 * (duck.onset_sample / RATE - onset) if duck else None,
    }


CFG = FeatureConfig()


def classify_windows(scenarios: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for sc in scenarios:
        anchor = sc["onset_s"] + 0.35
        if sc["kind"] == "none":
            anchor = sc["onset_s"]
        end = int(anchor * RATE)
        window = sc["mic"][max(0, end - int(0.4 * RATE)) : end]
        if window.size < int(0.3 * RATE):
            continue
        xs.append(window_features(window, CFG))
        ys.append(int(sc["kind"] == "interrupt"))
    return np.asarray(xs), np.asarray(ys)


def summarize(results: list[dict[str, Any]], scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([sc["kind"] == "interrupt" for sc in scenarios])
    p = np.asarray([r["stopped"] for r in results])
    tp, tn = int((p & y).sum()), int((~p & ~y).sum())
    fp, fn = int((p & ~y).sum()), int((~p & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    by_kind = {
        kind: {
            "scenarios": int(sum(sc["kind"] == kind for sc in scenarios)),
            "stop_rate": float(np.mean([r["stopped"] for r, sc in zip(results, scenarios, strict=True) if sc["kind"] == kind])),
        }
        for kind in CLASSES
    }
    by_timing = {
        t: float(np.mean([r["stopped"] for r, sc in zip(results, scenarios, strict=True) if sc["kind"] == "interrupt" and sc["timing"] == t]))
        for t in TIMINGS
    }

    def lat(name: str) -> dict[str, float | None]:
        vals = [r[name] for r, sc in zip(results, scenarios, strict=True) if sc["kind"] == "interrupt" and r[name] is not None]
        if not vals:
            return {"n": 0}
        vals.sort()
        return {
            "n": len(vals),
            "median": statistics.median(vals),
            "p90": vals[int(0.9 * (len(vals) - 1))],
            "p95": vals[int(0.95 * (len(vals) - 1))],
            "max": vals[-1],
            "share_le_150ms": float(np.mean([v <= 150 for v in vals])),
        }

    return {
        "accuracy": (tp + tn) / max(1, y.size),
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision": precision,
        "recall_interrupt_stopped": recall,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "by_kind": by_kind,
        "interrupt_stop_rate_by_timing": by_timing,
        "duck_latency_ms": lat("duck_latency_ms"),
        "stop_latency_ms": lat("stop_latency_ms"),
        "false_ducks_before_event_rate": float(np.mean([r["false_ducks_before_event"] > 0 for r in results])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--noise-dir", type=Path, required=True)
    parser.add_argument("--reply-dir", type=Path, required=True)
    parser.add_argument("--rir-dir", type=Path, required=True)
    parser.add_argument("--per-class-train", type=int, default=150)
    parser.add_argument("--per-class-test", type=int, default=150)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not isinstance(default_scorer(), SileroScorer):
        raise SystemExit("the simulation requires the deployed Silero VAD model")

    pools = Pools(args.pool, args.noise_dir, args.reply_dir, args.rir_dir)
    rng = random.Random(20261016)

    def build(split: str, per_class: int) -> list[dict[str, Any]]:
        return [
            make_scenario(pools, split, kind, TIMINGS[i % 3], rng)
            for kind in CLASSES
            for i in range(per_class)
        ]

    train = build("train", args.per_class_train)
    test = build("test", args.per_class_test)

    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_tr, y_tr = classify_windows(train)
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=500, random_state=0),
    )
    mlp.fit(x_tr, y_tr)
    variants: dict[str, Any] = {}
    for name, classifier, threshold in (
        ("silero_duration_only", None, 0.7),
        ("silero_plus_mlp", mlp, 0.7),
    ):
        results = [run_controller(sc, classifier, threshold) for sc in test]
        variants[name] = summarize(results, test)
    x_te, y_te = classify_windows(test)
    mlp_window_accuracy = float((mlp.predict(x_te) == y_te).mean())
    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        with args.model_out.open("wb") as handle:
            pickle.dump({"model": mlp, "threshold": 0.7, "window_s": 0.4}, handle)
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "evidence_class": "simulated_deployment_events_from_real_recordings",
        "human_verified": False,
        "pools": pools.counts(),
        "scenarios": {"train": len(train), "test": len(test)},
        "levels_db": {
            "user_rms": [USER_RMS_DB - 4, USER_RMS_DB + 4],
            "residual_echo_relative": [-32, -18],
            "background_relative": [-38, -24],
            "noise_burst_relative": [-6, 3],
        },
        "controller": {"duck_frames": 2, "confirm_speech_ms": 480, "resume_silence_ms": 320},
        "mlp_window_test_accuracy": mlp_window_accuracy,
        "variants": variants,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pool_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, v in variants.items():
        print(name, json.dumps({k: v[k] for k in ("accuracy", "f1", "by_kind", "duck_latency_ms", "stop_latency_ms", "interrupt_stop_rate_by_timing", "false_ducks_before_event_rate")}, ensure_ascii=False))
    print("pools", pools.counts(), "mlp window acc", round(mlp_window_accuracy, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
