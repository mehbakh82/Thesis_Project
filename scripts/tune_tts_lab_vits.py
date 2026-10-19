#!/usr/bin/env python3
"""Lab VITS inference optimization on the development texts.

Protocol: ``docs/TTS_LAB_VITS_OPTIMIZATION_PROTOCOL.md``. Three subcommands:

``check-serving``
    Step 1 criteria on the development texts, written to a receipt: the
    phoneme strings of every sentence chunk with the in-process espeak-ng (and
    the cached version/voice queries) against the fork's per-call processes
    (must be identical), and the faster 22.05 → 16 kHz resampler against
    ``kaiser_best`` on the model's raw output (SNR per clip at least 40 dB).
``synth``
    Synthesizes the development texts without digits for a set of named
    configurations (``grid``: the 18 inference settings of step 3; ``split``:
    the step 4 sentence-handling pair around ``--base``; ``punct``: the step 4b
    Persian-punctuation pair around ``--base``; ``tokenize``: the step 2b
    tokenizer-fix pair around ``--base``) and writes one combined
    panel whose ids are ``<config>:<row_id>``, for ``scripts/asr_predict.py``.
    Every text is synthesized with the same seed in every configuration.
``score``
    Scores a synthesized set with the two ASR listeners and DNSMOS and applies
    the protocol's selection rule; writes a write-once receipt (no text).

Run ``check-serving`` and ``synth`` with an interpreter that has the lab
repository's dependencies and ``star_tts_plus`` (``--plus``); run ``score``
with the project environment and onnxruntime.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RATE = 16_000
NOISE = (0.667, 0.333, 0.0)
NOISE_DP = (1.0, 0.5, 0.0)
LENGTH = (1.0, 1.1)
TIE_PP = 0.005  # step 3: settings within 0.5 percentage points of the best CER tie
PAUSE_MS = 120.0
LISTENERS = {"nemo_v4_ctc": "asr_nemo", "seamless_v5ft_direct": "asr_seamless"}


def config_name(noise: float, noise_dp: float, length: float, split: str = "service") -> str:
    name = f"ns{noise:g}_dp{noise_dp:g}_ls{length:g}"
    return name if split == "service" else f"{name}_{split}{PAUSE_MS:g}"


def grid() -> dict[str, dict[str, Any]]:
    return {
        config_name(n, d, s): {"noise_scale": n, "noise_scale_dp": d, "length_scale": s, "split": "service"}
        for n, d, s in itertools.product(NOISE, NOISE_DP, LENGTH)
    }


def split_pair(base: str) -> dict[str, dict[str, Any]]:
    settings = dict(grid()[base])
    punct = {**settings, "split": "punct", "pause_ms": PAUSE_MS}
    return {base: settings, config_name(settings["noise_scale"], settings["noise_scale_dp"], settings["length_scale"], "punct"): punct}


def resolve(name: str) -> dict[str, Any]:
    """Settings of a named configuration (grid name plus step suffixes)."""

    for suffix, key in (("_tokfix", "tokenizer_fix"), ("_fapunct", "persian_punct")):
        if name.endswith(suffix):
            return {**resolve(name.removesuffix(suffix)), key: True}
    grid_name = name.removesuffix(f"_punct{PAUSE_MS:g}")
    return dict(split_pair(grid_name)[name])


def punct_pair(base: str) -> dict[str, dict[str, Any]]:
    settings = resolve(base)
    return {base: settings, f"{base}_fapunct": {**settings, "persian_punct": True}}


def tokenize_pair(base: str) -> dict[str, dict[str, Any]]:
    settings = resolve(base)
    return {base: settings, f"{base}_tokfix": {**settings, "tokenizer_fix": True}}


def load_texts(path: Path, *, digits: bool) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r["reply"].strip() and (digits or not r["has_digits"])]


def seed_for(row_id: str) -> int:
    return int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def engine(args: argparse.Namespace, **options: Any) -> Any:
    sys.path.insert(0, str(args.plus))
    from star_tts_plus.engine import Options, StarTTS

    return StarTTS(args.repo, Options(cuda=True, fast_espeak=True, numbers=True, **options))


def check_serving(args: argparse.Namespace) -> int:
    import contextlib
    import io
    import os

    import torch

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite write-once receipt {args.out}")
    texts = load_texts(args.texts, digits=True)
    sys.path.insert(0, str(args.plus))
    sys.path.insert(0, str(args.repo))
    from star_tts_plus.engine import Options, StarTTS, resample
    from star_tts_plus.espeak_fast import install

    tts = StarTTS(args.repo, Options(cuda=True, numbers=True))  # the fork's espeak processes
    previous = os.getcwd()
    os.chdir(args.repo)  # the front-end opens its data files by relative path
    from TTS.tts.utils.text.persian.Phonemizer.phonemizer import Text2Phoneme
    from TTS.tts.utils.text.phonemizers import espeak_wrapper

    g2p = Text2Phoneme()
    chunks = [chunk for row in texts for chunk in tts.chunks(row["reply"])]

    def phonemes() -> tuple[list[str], float]:
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            out = [g2p.farsi_text_to_phonemes(chunk) for chunk in chunks]
        return out, 1000 * (time.perf_counter() - started) / len(chunks)

    reference, ms_processes = {}, {}
    for fix in (False, True):  # the fork's tokenizer and the step 2b fix
        tts.set_tokenizer_fix(fix)
        reference[fix], ms_processes[fix] = phonemes()
    install(espeak_wrapper)
    identical, ms_in_process = {}, {}
    for fix in (False, True):
        tts.set_tokenizer_fix(fix)
        fast, ms_in_process[fix] = phonemes()
        identical[fix] = sum(a == b for a, b in zip(reference[fix], fast, strict=True))
    tts.set_tokenizer_fix(False)
    os.chdir(previous)

    snrs, timing = [], {"kaiser_best": [], args.method: []}
    for row in texts:
        torch.manual_seed(seed_for(row["row_id"]))
        raw = tts.raw(row["reply"])
        outputs = {}
        for method in timing:
            started = time.perf_counter()
            outputs[method] = resample(raw, method)
            timing[method].append(1000 * (time.perf_counter() - started))
        ref, new = outputs["kaiser_best"], outputs[args.method]
        n = min(ref.size, new.size)
        snrs.append(float(10 * np.log10(np.sum(ref[:n] ** 2) / (np.sum((ref[:n] - new[:n]) ** 2) + 1e-20))))
    receipt = {
        "schema_version": 1,
        "experiment": "tts_lab_vits_dev_serving",
        "protocol": "docs/TTS_LAB_VITS_OPTIMIZATION_PROTOCOL.md (step 1, amendment 1)",
        "split": "development texts (all, including digits)",
        "texts": len(texts),
        "texts_sha256": hashlib.sha256(args.texts.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((args.repo / "models/model.pth").read_bytes()).hexdigest(),
        "espeak": {
            "chunks": len(chunks),
            "identical_phoneme_strings": {
                "fork_tokenizer": identical[False],
                "tokenizer_fix": identical[True],
            },
            "passes": identical[False] == identical[True] == len(chunks),
            "front_end_ms_per_chunk": {
                "fork_processes": ms_processes[False],
                "in_process_cached": ms_in_process[False],
            },
            "chunks_changed_by_tokenizer_fix": sum(
                a != b for a, b in zip(reference[False], reference[True], strict=True)
            ),
        },
        "resampler": {
            "method": args.method,
            "reference": "kaiser_best (librosa / resampy)",
            "snr_db_min": min(snrs),
            "snr_db_median": statistics.median(snrs),
            "criterion_snr_db_min": 40.0,
            "passes": min(snrs) >= 40.0,
            "ms_per_clip_median": {m: statistics.median(t) for m, t in timing.items()},
        },
        "device": torch.cuda.get_device_name(0),
        "text_retained": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"espeak": receipt["espeak"], "resampler": receipt["resampler"]}))
    return 0


def synth(args: argparse.Namespace) -> int:
    import torch

    configs = {"grid": grid, "split": split_pair, "punct": punct_pair, "tokenize": tokenize_pair}[args.set](
        *(() if args.set == "grid" else (args.base,))
    )
    tts = engine(args, resample=args.resample)
    texts = load_texts(args.texts, digits=False)
    panel, timing = [], {}
    for name, settings in configs.items():
        tts.configure(settings["noise_scale"], settings["noise_scale_dp"], settings["length_scale"])
        tts.options.split = settings["split"]
        tts.options.pause_ms = settings.get("pause_ms", 0.0)
        tts.options.persian_punct = settings.get("persian_punct", False)
        tts.set_tokenizer_fix(settings.get("tokenizer_fix", False))
        started = time.perf_counter()
        for row in texts:
            torch.manual_seed(seed_for(row["row_id"]))
            audio = tts.tts(row["reply"])
            path = args.work / name / f"{row['row_id']}.wav"
            write_wav(path, audio)
            panel.append({"id": f"{name}:{row['row_id']}", "audio_filepath": str(path), "text": row["reply"], "duration": round(audio.size / RATE, 3)})
        timing[name] = time.perf_counter() - started
        print(name, f"{timing[name]:.0f} s", flush=True)
    (args.work / f"panel_{args.set}.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in panel), encoding="utf-8"
    )
    summary = {
        "set": args.set,
        "configs": configs,
        "texts": len(texts),
        "texts_sha256": hashlib.sha256(args.texts.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((args.repo / "models/model.pth").read_bytes()).hexdigest(),
        "resample": args.resample,
        "seconds_per_config": timing,
        "device": torch.cuda.get_device_name(0),
    }
    (args.work / f"synthesis_{args.set}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


def score(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate_tts_candidates_v2 import Dnsmos

    from thesis_s2s.eval.asr_normalize import NORMALIZER_VERSION, error_counts

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite write-once receipt {args.out}")
    synthesis = json.loads((args.work / f"synthesis_{args.set}.json").read_text(encoding="utf-8"))
    panel_path = args.work / f"panel_{args.set}.jsonl"
    panel = [json.loads(line) for line in panel_path.read_text(encoding="utf-8").splitlines() if line]
    hashes = {panel_path.name: hashlib.sha256(panel_path.read_bytes()).hexdigest()}
    counts: dict[str, dict[str, list[dict[str, int]]]] = {name: {lis: [] for lis in LISTENERS} for name in synthesis["configs"]}
    for listener, folder in LISTENERS.items():
        path = args.work / folder / f"{args.set}.jsonl"
        hashes[f"{folder}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        hyps = {p["id"]: p for p in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}
        for row in panel:
            prediction = hyps[row["id"]]
            if prediction.get("error"):
                raise SystemExit(f"listener {listener} failed on {row['id']}")
            counts[row["id"].split(":", 1)[0]][listener].append(error_counts(row["text"], prediction["hypothesis"]))
    dnsmos = Dnsmos(args.dnsmos)
    quality: dict[str, list[dict[str, float]]] = {name: [] for name in synthesis["configs"]}
    for row in panel:
        quality[row["id"].split(":", 1)[0]].append(dnsmos(read_wav(row["audio_filepath"])))
    configs: dict[str, Any] = {}
    for name, settings in synthesis["configs"].items():
        entry: dict[str, Any] = {"settings": settings}
        for listener in LISTENERS:
            rows = counts[name][listener]
            entry[f"cer_{listener}"] = sum(c["char_errors"] for c in rows) / sum(c["reference_chars"] for c in rows)
        entry["cer_two_listener_mean"] = statistics.fmean(entry[f"cer_{lis}"] for lis in LISTENERS)
        for key in ("sig", "bak", "ovrl"):
            entry[f"dnsmos_{key}"] = statistics.fmean(q[key] for q in quality[name])
        entry["seconds_audio_median"] = statistics.median(r["duration"] for r in panel if r["id"].startswith(f"{name}:"))
        configs[name] = entry
    best_cer = min(c["cer_two_listener_mean"] for c in configs.values())
    if args.set == "grid":
        tied = [n for n, c in configs.items() if c["cer_two_listener_mean"] <= best_cer + TIE_PP + 1e-12]
        selected = max(tied, key=lambda n: configs[n]["dnsmos_ovrl"])
        rule = "lowest two-listener CER; among settings within 0.5 pp of it, highest DNSMOS OVRL"
        decision: dict[str, Any] = {"tied_within_0_5pp": sorted(tied), "selected": selected}
    else:
        base, punct = list(synthesis["configs"])
        not_worse = configs[punct]["cer_two_listener_mean"] <= configs[base]["cer_two_listener_mean"]
        if args.set == "tokenize":  # a correctness fix: judged on intelligibility, DNSMOS reported
            keep = not_worse
            rule = "tokenizer fix kept if development CER is not worse (amendment 2)"
        else:
            keep = not_worse and configs[punct]["dnsmos_ovrl"] >= configs[base]["dnsmos_ovrl"]
            change = "punctuation-aware split" if args.set == "split" else "Persian punctuation mapping"
            rule = f"{change} kept if development CER is not worse and DNSMOS OVRL is not lower"
        decision = {"base": base, "candidate": punct, "candidate_kept": keep, "selected": punct if keep else base}
    receipt = {
        "schema_version": 1,
        "experiment": f"tts_lab_vits_dev_{args.set}",
        "protocol": "docs/TTS_LAB_VITS_OPTIMIZATION_PROTOCOL.md",
        "split": "development (responder v2 replies of the other arms; no evaluation text), texts without digits",
        "texts": synthesis["texts"],
        "texts_sha256": synthesis["texts_sha256"],
        "model_sha256": synthesis["model_sha256"],
        "resample": synthesis["resample"],
        "device": synthesis["device"],
        "listeners": list(LISTENERS),
        "normalizer": NORMALIZER_VERSION,
        "dnsmos_model_sha256": hashlib.sha256(args.dnsmos.read_bytes()).hexdigest(),
        "input_sha256": hashes,
        "configs": configs,
        "rule": rule,
        "decision": decision,
        "text_retained": False,
        "claim_boundary": "Development-set automatic proxies used for selection only; not an evaluation result.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({n: [round(c["cer_two_listener_mean"], 4), round(c["dnsmos_ovrl"], 3)] for n, c in configs.items()}))
    print(json.dumps(decision))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check-serving", "synth"):
        command = sub.add_parser(name)
        command.add_argument("--repo", type=Path, required=True, help="lab TTS repository")
        command.add_argument("--plus", type=Path, required=True, help="directory containing star_tts_plus")
        command.add_argument("--texts", type=Path, required=True)
        command.add_argument("--work", type=Path, required=True)
    sub.choices["check-serving"].add_argument("--method", default="kaiser_best_poly")
    sub.choices["check-serving"].add_argument("--out", type=Path, required=True)
    sub.choices["synth"].add_argument("--set", choices=("grid", "split", "punct", "tokenize"), required=True)
    sub.choices["synth"].add_argument("--base", help="selected configuration (for --set split and punct)")
    sub.choices["synth"].add_argument("--resample", default="kaiser_best")
    command = sub.add_parser("score")
    command.add_argument("--work", type=Path, required=True)
    command.add_argument("--set", choices=("grid", "split", "punct", "tokenize"), required=True)
    command.add_argument("--dnsmos", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "synth" and args.set != "grid" and not args.base:
        parser.error(f"--set {args.set} needs --base")
    return {"check-serving": check_serving, "synth": synth, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
