#!/usr/bin/env python3
"""Evaluate Moshi v7a candidates under docs/MOSHI_V7_ROMANIZED_ALIGNMENT_PROTOCOL.md.

The official Moshi server is exercised through the same harness as v2-v6
(`run_candidate_server`). A thin wrapper around the harness's exercise routine
additionally retains each row's decoded assistant speech so that the
predeclared secondary Persian ASR intelligibility measurement can be computed
later. Row gates use the romanized training targets written by
`scripts/prepare_moshi_v7.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.evaluate_moshi_v2_runtime_candidates as harness  # noqa: E402
from scripts.build_moshi_client import tree_manifest  # noqa: E402
from scripts.evaluate_moshi_v2_runtime_candidates import (  # noqa: E402
    candidate_static_validation,
    load_jsonl,
    run_candidate_server,
    write_report,
)
from scripts.validate_moshi_adapter import (  # noqa: E402
    expected_adapter_schema,
    json_object,
    project_path,
    sha256_file,
)

PANEL_INDICES = (0, 3, 7, 11, 15, 19, 23, 27, 31)
EXPECTED_STEPS = (50, 100, 150, 200)
FULL_TEXT_PARAMETERS = {
    "depformer_text_emb.weight",
    "text_emb.weight",
    "text_linear.frozen_W.weight",
}
ASCII_LETTER_FRACTION_THRESHOLD = 0.8
SPEECH_RMS_THRESHOLD = 1e-3
MAX_PREFIX_CER = 0.75
MIN_NORMALIZED_TEXT_LENGTH = 4
MIN_TEXT_LOSS_REDUCTION = 0.30
FUSE_LORA_AT_RUNTIME = False
DECODED_SAMPLE_RATE = 24_000

_CAPTURED: list[Any] = []


def normalize_roman(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    value = re.sub(r"[^a-z]+", " ", value)
    return " ".join(value.split())


def ascii_letter_fraction(text: str) -> float:
    letters = [c for c in unicodedata.normalize("NFKC", text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum("a" <= c.lower() <= "z" for c in letters) / len(letters)


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, lc in enumerate(left, start=1):
        current = [i]
        for j, rc in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (lc != rc)))
        previous = current
    return previous[-1]


def prefix_cer(generated: str, target: str) -> float:
    gen, tgt = normalize_roman(generated), normalize_roman(target)
    if not gen:
        return 1.0
    prefix = tgt[: len(gen)]
    return edit_distance(gen, prefix) / max(len(gen), len(prefix), 1)


def romanized_target(row: dict[str, Any]) -> tuple[str, str]:
    """Return (romanized target, Persian-source sha256) from the v7 sidecar."""

    sidecar = Path(str(row["path"])).with_suffix(".json")
    metadata = json_object(sidecar)
    words = [str(item[0]) for item in metadata.get("alignments") or []]
    if not words or metadata.get("v7_text_representation") is None:
        raise ValueError(f"not a v7 sidecar: {sidecar}")
    return " ".join(words), str(metadata.get("v7_source_persian_text_sha256"))


class _RecordingOpusReader:
    """Proxy that records every decoded PCM block of the harness's reader."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def append_bytes(self, payload: bytes) -> Any:
        return self._inner.append_bytes(payload)

    def read_pcm(self) -> Any:
        import numpy as np

        pcm = self._inner.read_pcm()
        if getattr(pcm, "size", 0):
            _CAPTURED.append(np.asarray(pcm, dtype="float32").reshape(-1).copy())
        return pcm


def _install_audio_capture(audio_dir: Path) -> None:
    import numpy as np
    import sphn

    original_exercise = harness.exercise_server
    original_reader = sphn.OpusStreamReader

    async def exercise_and_retain(**kwargs: Any) -> dict[str, Any]:
        _CAPTURED.clear()
        sphn.OpusStreamReader = lambda rate: _RecordingOpusReader(original_reader(rate))
        try:
            result = await original_exercise(**kwargs)
        finally:
            sphn.OpusStreamReader = original_reader
        pcm = np.concatenate(_CAPTURED) if _CAPTURED else np.zeros(0, dtype="float32")
        out = audio_dir / f"{Path(str(kwargs['input_audio'])).stem}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        clipped = (np.clip(pcm, -1.0, 1.0) * 32767.0).round().astype("<i2")
        with wave.open(str(out), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(DECODED_SAMPLE_RATE)
            handle.writeframes(clipped.tobytes())
        result["generated_audio_retained"] = True
        result["generated_audio_path"] = out.relative_to(ROOT).as_posix()
        result["generated_audio_file_sha256"] = sha256_file(out)
        return result

    harness.exercise_server = exercise_and_retain


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=Path("configs/moshi_h100_v7a_overfit.yaml"))
    parser.add_argument("--reevaluation", type=Path, default=Path("results/moshi_v7a_overfit_reevaluation.json"))
    parser.add_argument("--data-report", type=Path, default=Path("results/moshi_v7a_overfit_data.json"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/MOSHI_V7_ROMANIZED_ALIGNMENT_PROTOCOL.md"))
    parser.add_argument("--policy", type=Path, default=Path("configs/moshi_v7a_overfit_policy.json"))
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/moshi_v6_text_greedy_runtime.json"))
    parser.add_argument("--client-report", type=Path, default=Path("results/hardware/moshi_client_build.json"))
    parser.add_argument("--out", type=Path, default=Path("results/moshi_v7a_overfit_runtime.json"))
    parser.add_argument("--audio-dir", type=Path, default=Path("checkpoints/moshi_v7a_overfit/runtime_audio"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18998)
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--response-timeout", type=float, default=20.0)
    parser.add_argument("--input-seconds", type=float, default=5.0)
    parser.add_argument("--post-input-silence-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("v7 diagnostic runtime is restricted to loopback")

    paths = {
        name: (ROOT / value).resolve()
        for name, value in {
            "training_config": args.training_config,
            "reevaluation": args.reevaluation,
            "data_report": args.data_report,
            "protocol": args.protocol,
            "policy": args.policy,
            "runtime_config": args.runtime_config,
            "client_report": args.client_report,
        }.items()
    }
    out_path = (ROOT / args.out).resolve()
    audio_dir = (ROOT / args.audio_dir).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v7 runtime evidence: {out_path}")

    training_config = yaml.safe_load(paths["training_config"].read_text(encoding="utf-8"))
    reevaluation = json_object(paths["reevaluation"])
    data_report = json_object(paths["data_report"])
    policy = json_object(paths["policy"])
    client_report = json_object(paths["client_report"])
    runtime_config = json_object(paths["runtime_config"])
    data_config = training_config.get("data") or {}
    moshi_paths = training_config.get("moshi_paths") or {}
    lora_config = training_config.get("lora") or {}
    manifest = project_path(ROOT, data_config.get("train_data"), "train_data")
    rows = load_jsonl(manifest)

    base_config_path = project_path(ROOT, moshi_paths.get("config_path"), "config_path")
    base_model_path = project_path(ROOT, moshi_paths.get("moshi_path"), "moshi_path")
    mimi_path = project_path(ROOT, moshi_paths.get("mimi_path"), "mimi_path")
    tokenizer_path = project_path(ROOT, moshi_paths.get("tokenizer_path"), "tokenizer_path")
    expected_config = json_object(base_config_path)
    expected_config.update(
        {
            "lora": lora_config.get("enable"),
            "lora_rank": lora_config.get("rank"),
            "lora_scaling": lora_config.get("scaling"),
        }
    )
    generation_config = dict(runtime_config)
    generation_config.pop("lm_gen_config", None)
    expected_schema = expected_adapter_schema(
        expected_config, ft_embed=False, embedding_names=FULL_TEXT_PARAMETERS
    )
    client_dist = client_report.get("dist") or {}
    static_dir = (ROOT / str(client_dist.get("path"))).resolve()
    static_files, static_tree_hash = tree_manifest(static_dir)
    candidates_by_step = {
        int(c["step"]): c for c in reevaluation.get("candidates") or [] if isinstance(c, dict)
    }
    protocol_rel = paths["protocol"].relative_to(ROOT).as_posix()
    protocol_commit = _git("log", "-1", "--format=%H", "--", protocol_rel)
    first_checkpoint = (
        ROOT
        / str(training_config["run_dir"])
        / "checkpoints"
        / f"checkpoint_{EXPECTED_STEPS[0]:06d}"
    )
    protocol_committed_before_training = bool(protocol_commit) and (
        int(_git("show", "-s", "--format=%ct", protocol_commit))
        <= int(first_checkpoint.stat().st_mtime)
        if first_checkpoint.exists()
        else False
    )
    preconditions = {
        "protocol_committed_before_training": protocol_committed_before_training,
        "protocol_unchanged_since_commit": _git("status", "--porcelain", "--", protocol_rel) == "",
        "data_freeze_current": data_report.get("passes") is True
        and data_report.get("final_test_accessed") is False
        and (data_report.get("output") or {}).get("manifest_sha256") == sha256_file(manifest),
        "thirty_two_train_only_rows": len(rows) == 32
        and data_config.get("train_data") == data_config.get("eval_data")
        and "test" not in manifest.as_posix().lower(),
        "reevaluation_complete_and_in_sample": reevaluation.get("status") == "passed"
        and reevaluation.get("heldout_test_used") is False
        and sorted(candidates_by_step) == list(EXPECTED_STEPS),
        "policy_current": policy.get("mode") == "persian_text_head_adaptation_romanized_word_aligned"
        and set(policy.get("trainable_full_parameters") or []) == FULL_TEXT_PARAMETERS,
        "runtime_model_config_matches_training": generation_config == expected_config,
        "client_build_current": client_report.get("valid") is True
        and static_tree_hash == client_dist.get("tree_sha256")
        and len(static_files) == len(client_dist.get("files") or []),
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"v7 runtime preconditions failed: {preconditions}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment": "moshi_v7a_romanized_word_aligned",
        "diagnostic_only": True,
        "generalization_claim_allowed": False,
        "final_test_accessed": False,
        "controlled_baseline": "results/moshi_v6_text_greedy_runtime.json",
        "protocol": {
            "panel_indices": list(PANEL_INDICES),
            "all_nine_rows_must_pass": True,
            "speech_rms_threshold": SPEECH_RMS_THRESHOLD,
            "ascii_letter_fraction_threshold": ASCII_LETTER_FRACTION_THRESHOLD,
            "maximum_target_prefix_character_error_rate": MAX_PREFIX_CER,
            "minimum_normalized_text_length": MIN_NORMALIZED_TEXT_LENGTH,
            "replacement_character_allowed": False,
            "minimum_text_loss_reduction_from_step_50": MIN_TEXT_LOSS_REDUCTION,
            "decoding": runtime_config.get("lm_gen_config"),
            "official_server_fuse_lora": FUSE_LORA_AT_RUNTIME,
            "secondary_metric": "Persian ASR CER of retained generated speech (reported, not gated)",
        },
        "preconditions": preconditions,
        "candidates": [],
        "selection": None,
        "artifacts": {
            **{f"{name}_sha256": sha256_file(path) for name, path in paths.items()},
            "protocol_commit": protocol_commit,
            "training_manifest_sha256": sha256_file(manifest),
            "base_model_sha256": sha256_file(base_model_path),
            "mimi_sha256": sha256_file(mimi_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "launcher_sha256": sha256_file(ROOT / "scripts/moshi_train_entry.py"),
        },
    }
    write_report(out_path, report)
    _install_audio_capture(audio_dir)

    run_dir = (ROOT / str(training_config["run_dir"])).resolve()
    for step in EXPECTED_STEPS:
        candidate = candidates_by_step[step]
        consolidated = run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"
        adapter_path = consolidated / "lora.safetensors"
        config_path = consolidated / "config.json"
        static_validation = candidate_static_validation(
            candidate=candidate,
            adapter_path=adapter_path,
            config_path=config_path,
            expected_config=expected_config,
            expected_schema=expected_schema,
            ft_embed=False,
            embedding_names=FULL_TEXT_PARAMETERS,
        )
        runtime = run_candidate_server(
            step=step,
            adapter_path=adapter_path,
            config_path=config_path,
            moshi_paths=moshi_paths,
            base_model_path=base_model_path,
            mimi_path=mimi_path,
            tokenizer_path=tokenizer_path,
            static_dir=static_dir,
            validation_manifest=manifest,
            validation_rows=rows,
            host=args.host,
            port=args.port,
            ready_timeout=args.ready_timeout,
            response_timeout=args.response_timeout,
            input_seconds=args.input_seconds,
            post_input_silence_seconds=args.post_input_silence_seconds,
            panel_indices=PANEL_INDICES,
            split_label="train_only_in_sample_diagnostic",
            fuse_lora=FUSE_LORA_AT_RUNTIME,
            server_config_path=paths["runtime_config"],
        )
        for panel_row in runtime["panel"]:
            index = int(panel_row["validation_index"])
            target, persian_sha = romanized_target(rows[index])
            stream = panel_row.get("stream") or {}
            generated = str(stream.get("generated_text") or "")
            normalized = normalize_roman(generated)
            cer = prefix_cer(generated, target)
            requirements = dict(panel_row["requirements"])
            # The harness's Persian-script check does not apply to romanized text;
            # it is replaced by the predeclared ASCII-letter rule below.
            requirements.pop("persian_script_observed", None)
            requirements.update(
                {
                    "normalized_text_length_at_least_4": len(normalized) >= MIN_NORMALIZED_TEXT_LENGTH,
                    "ascii_letter_fraction_at_least_0_80": ascii_letter_fraction(generated)
                    >= ASCII_LETTER_FRACTION_THRESHOLD,
                    "no_unicode_replacement_character": "�" not in generated,
                    "target_prefix_cer_at_most_0_75": cer <= MAX_PREFIX_CER,
                }
            )
            panel_row["requirements"] = requirements
            panel_row["target"] = {
                "romanized_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                "persian_source_sha256": persian_sha,
                "normalized_length": len(normalize_roman(target)),
            }
            panel_row["target_prefix_character_error_rate"] = cer
            panel_row["passes"] = all(requirements.values())
        runtime["panel_pass_count"] = sum(row["passes"] is True for row in runtime["panel"])
        runtime["all_panel_output_gates_pass"] = runtime["panel_pass_count"] == 9
        runtime["static_adapter_validation"] = static_validation
        runtime["text_eval_loss"] = float(candidate["text_eval_loss"])
        runtime["audio_eval_loss"] = float(candidate["audio_eval_loss"])
        runtime["eval_loss"] = float(candidate["eval_loss"])
        runtime["candidate_runtime_passes"] = (
            static_validation["passes"] is True
            and runtime["official_server_runtime_passes"] is True
            and runtime["all_panel_output_gates_pass"] is True
        )
        runtime["step"] = step
        runtime["adapter_path"] = adapter_path.relative_to(ROOT).as_posix()
        report["candidates"].append(runtime)
        write_report(out_path, report)

    passing = [c for c in report["candidates"] if c["candidate_runtime_passes"]]
    selected = min(passing, key=lambda c: int(c["step"])) if passing else None
    step50 = next(c for c in report["candidates"] if c["step"] == 50)
    loss_rule = bool(selected) and (
        selected["step"] == 50
        or selected["text_eval_loss"] <= (1 - MIN_TEXT_LOSS_REDUCTION) * step50["text_eval_loss"]
    )
    report["selection"] = (
        {"step": selected["step"], "adapter_path": selected["adapter_path"], "text_loss_rule_passes": loss_rule}
        if selected
        else None
    )
    report["panel_pass_counts"] = {str(c["step"]): c["panel_pass_count"] for c in report["candidates"]}
    report["requirements"] = {
        **preconditions,
        "all_candidates_evaluated": [c["step"] for c in report["candidates"]] == list(EXPECTED_STEPS),
        "all_static_adapter_checks_passed": all(
            c["static_adapter_validation"]["passes"] is True for c in report["candidates"]
        ),
        "all_official_servers_exercised": all(
            c["official_server_runtime_passes"] is True for c in report["candidates"]
        ),
    }
    report["diagnostic_positive"] = bool(selected) and loss_rule
    report["diagnostic_outcome"] = "positive" if report["diagnostic_positive"] else "negative"
    report["status"] = "passed" if all(report["requirements"].values()) else "failed"
    report["completed_at"] = datetime.now(UTC).isoformat()
    write_report(out_path, report)
    print(json.dumps({k: report[k] for k in ("panel_pass_counts", "diagnostic_outcome", "status")}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
