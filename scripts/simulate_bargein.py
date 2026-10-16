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
20 ms frames, confirming either by duration/acoustic classifier or lexically
with the deployed ASR (``lexical`` variants: the reply text is the echo
vocabulary). Backchannel clips are selected by a *different* ASR listener
than the one used for the lexical confirm (see the pool's ``asr_model_sha256``). ``stop`` is the positive decision; accuracy/F1 treat
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
from thesis_s2s.bargein.lexicon import echo_vocabulary  # noqa: E402
from thesis_s2s.runtime.bargein_live import BargeInController  # noqa: E402
from thesis_s2s.runtime.cascade import _asr_transcript, _wav_bytes, asr_http  # noqa: E402
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
    def __init__(self, pool: Path, noise_dir: Path, reply_dir: Path, rir_dir: Path, reply_texts: Path | None = None) -> None:
        rows = [json.loads(line) for line in pool.read_text(encoding="utf-8").splitlines() if line]
        self.reply_text: dict[str, str] = {}
        if reply_texts is not None:
            for line in reply_texts.read_text(encoding="utf-8").splitlines():
                if line:
                    item = json.loads(line)
                    self.reply_text[item["file"]] = item["text"]
        self.listeners = sorted({str(r["asr_model_sha256"]) for r in rows if r.get("asr_model_sha256")})
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
    reply_path = rng.choice(pools.items[(split, "reply")])
    reply = read(reply_path)
    user_db = USER_RMS_DB + rng.uniform(-4, 4)
    mic = np.zeros(reply.size + int(2.5 * RATE), np.float32)
    echo_db = user_db + rng.uniform(-32, -18)
    delay = int(rng.uniform(0.02, 0.06) * RATE)  # echo path delay after the played reference
    played = at_db(reply, -20.0)
    reference = np.zeros(mic.size, np.float32)
    reference[: reply.size] = played
    mic[delay : delay + reply.size] += at_db(reply, echo_db)[: mic.size - delay]
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
    frames = reference[: reference.size // FRAME * FRAME].reshape(-1, FRAME)
    envelope = np.repeat(np.sqrt((frames**2).mean(axis=1)), FRAME)
    reference_level = np.zeros(mic.size, np.float32)
    reference_level[: envelope.size] = envelope
    return {
        "kind": kind,
        "timing": timing,
        "reference_level": reference_level,
        "mic": np.clip(mic, -1, 1).astype(np.float32),
        "onset_s": onset_s,
        "event_s": event_s,
        "reply_s": reply_s,
        "echo_words": echo_vocabulary([pools.reply_text.get(Path(reply_path).name, "")]),
    }


def run_controller(
    scenario: dict[str, Any],
    classifier: Any | None,
    threshold: float,
    margin: float | None = 12.0,
    dtd_margin: float | None = None,
    transcriber: Any | None = None,
) -> dict[str, Any]:
    controller = BargeInController(
        transcriber=transcriber,
        echo_words=lambda: scenario["echo_words"],
        scorer=default_scorer(),
        echo_margin_db=margin,
        dtd_margin_db=dtd_margin if dtd_margin is not None else 10.0,
        classifier=(lambda w: float(classifier.predict_proba(window_features(w, CFG)[None])[0, 1]))
        if classifier is not None
        else None,
        classifier_threshold=threshold,
    )
    events = []
    mic = scenario["mic"]
    reference = scenario["reference_level"] if dtd_margin is not None else None
    for start in range(0, mic.size - FRAME + 1, FRAME):
        events.extend(
            controller.push(
                mic[start : start + FRAME],
                sample_offset=start if start == 0 else None,
                reference_level=None if reference is None else reference[start : start + FRAME],
            )
        )
    onset = scenario["onset_s"]
    in_event = [e for e in events if e.onset_sample / RATE >= onset - 0.25]
    early_ducks = [e for e in events if e.kind == "duck" and e.onset_sample / RATE < onset - 0.25]
    stop = next((e for e in in_event if e.kind == "stop"), None)
    duck = next((e for e in in_event if e.kind == "duck"), None)
    if scenario["kind"] == "none":
        stop = next((e for e in events if e.kind == "stop"), None)
    lexical = [e for e in events if e.transcript is not None]
    return {
        "asr_calls_decisions": len(lexical),
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
    parser.add_argument("--stage", choices=("dev", "final"), required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--noise-dir", type=Path, required=True)
    parser.add_argument("--reply-dir", type=Path, required=True)
    parser.add_argument("--rir-dir", type=Path, required=True)
    parser.add_argument("--reply-texts", type=Path, help="JSONL file/text of the reply clips (echo vocabulary)")
    parser.add_argument("--per-class-train", type=int, default=150)
    parser.add_argument("--per-class-test", type=int, default=150)
    parser.add_argument("--choice", type=Path, help="dev receipt (required for final)")
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not isinstance(default_scorer(), SileroScorer):
        raise SystemExit("the simulation requires the deployed Silero VAD model")
    if args.stage == "final" and args.out.exists():
        raise SystemExit("refusing to overwrite the one-time final simulation receipt")

    pools = Pools(args.pool, args.noise_dir, args.reply_dir, args.rir_dir, args.reply_texts)

    def transcribe(audio: np.ndarray) -> str:
        return _asr_transcript(asr_http(_wav_bytes(audio)))

    confirm_asr: dict[str, Any] | None = None
    if args.reply_texts is not None:
        result = asr_http(_wav_bytes(np.zeros(RATE // 2, np.float32)))
        confirm_asr = {"model_sha256": result.get("model_sha256"), "decoder": result.get("decoder")}
        if confirm_asr["model_sha256"] in pools.listeners:
            raise SystemExit("the backchannel pool was selected by the same ASR as the lexical confirm")
    rng = random.Random(20261016)

    def build(split: str, per_class: int) -> list[dict[str, Any]]:
        return [
            make_scenario(pools, split, kind, TIMINGS[i % 3], rng)
            for kind in CLASSES
            for i in range(per_class)
        ]

    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit_mlp(scenarios: list[dict[str, Any]]) -> Any:
        x, y = classify_windows(scenarios)
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=500, random_state=0),
        )
        model.fit(x, y)
        return model

    train = build("train", args.per_class_train)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "stage": args.stage,
        "evidence_class": "simulated_deployment_events_from_real_recordings",
        "human_verified": False,
        "pools": pools.counts(),
        "levels_db": {
            "user_rms": [USER_RMS_DB - 4, USER_RMS_DB + 4],
            "residual_echo_relative": [-32, -18],
            "background_relative": [-38, -24],
            "noise_burst_relative": [-6, 3],
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pool_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest(),
        "pool_selection_asr_sha256": pools.listeners,
        "lexical_confirm_asr": confirm_asr,
        "reply_texts_sha256": hashlib.sha256(args.reply_texts.read_bytes()).hexdigest() if args.reply_texts else None,
    }
    if args.stage == "dev":
        order = list(range(len(train)))
        random.Random(7).shuffle(order)
        fit = [train[i] for i in order[: len(order) // 2]]
        dev = [train[i] for i in order[len(order) // 2 :]]
        mlp = fit_mlp(fit)
        variants: dict[str, Any] = {}
        configs = [(m, None) for m in (12.0, 16.0)] + [(12.0, d) for d in (6.0, 10.0, 14.0)]
        for margin, dtd in configs:
            for name, classifier in (("duration_only", None), ("plus_mlp", mlp)):
                key = f"{name}_margin_{margin}_dtd_{dtd}"
                results = [run_controller(sc, classifier, 0.7, margin, dtd) for sc in dev]
                variants[key] = summarize(results, dev)
        if confirm_asr is not None:
            for dtd in (None, 6.0, 10.0, 14.0):
                key = f"lexical_margin_12.0_dtd_{dtd}"
                results = [run_controller(sc, None, 0.7, 12.0, dtd, transcribe) for sc in dev]
                variants[key] = summarize(results, dev)
        chosen = max(variants, key=lambda k: (variants[k]["accuracy"], variants[k]["f1"]))
        receipt.update(dev_scenarios=len(dev), fit_scenarios=len(fit), variants=variants, chosen_variant=chosen)
        for key, v in variants.items():
            print(key, round(v["accuracy"], 3), round(v["f1"], 3), {k: round(x["stop_rate"], 2) for k, x in v["by_kind"].items()}, v["duck_latency_ms"].get("median"))
        print("chosen", chosen)
    else:
        choice = json.loads(args.choice.read_text())
        chosen = choice["chosen_variant"]
        margin_text, dtd_text = chosen.split("_margin_")[1].split("_dtd_")
        margin = None if margin_text == "None" else float(margin_text)
        dtd = None if dtd_text == "None" else float(dtd_text)
        mlp = fit_mlp(train) if chosen.startswith("plus_mlp") else None
        lexical = chosen.startswith("lexical")
        if lexical and confirm_asr is None:
            raise SystemExit("the chosen lexical variant needs --reply-texts and the confirm ASR")
        test = build("test", args.per_class_test)
        results = [run_controller(sc, mlp, 0.7, margin, dtd, transcribe if lexical else None) for sc in test]
        x_te, y_te = classify_windows(test)
        reference_mlp = mlp or fit_mlp(train)
        receipt.update(
            chosen_variant=chosen,
            dev_receipt_sha256=hashlib.sha256(args.choice.read_bytes()).hexdigest(),
            test_scenarios=len(test),
            test=summarize(results, test),
            mlp_window_test_accuracy=float((reference_mlp.predict(x_te) == y_te).mean()),
        )
        if args.model_out and mlp is not None:
            args.model_out.parent.mkdir(parents=True, exist_ok=True)
            with args.model_out.open("wb") as handle:
                pickle.dump(
                    {"model": mlp, "threshold": 0.7, "window_s": 0.4, "echo_margin_db": margin, "dtd_margin_db": dtd},
                    handle,
                )
        print(json.dumps(receipt["test"], indent=1)[:3000])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
