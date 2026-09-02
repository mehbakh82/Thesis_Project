#!/usr/bin/env python3
"""Evaluate Moshi v6 overfit candidates under the frozen in-sample protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
FULL_TEXT_PARAMETERS = {
    "depformer_text_emb.weight",
    "text_emb.weight",
    "text_linear.frozen_W.weight",
}
PERSIAN_LETTER_FRACTION_THRESHOLD = 0.8
SPEECH_RMS_THRESHOLD = 1e-3
MAX_PREFIX_CER = 0.75
MIN_NORMALIZED_TEXT_LENGTH = 4
FUSE_LORA_AT_RUNTIME = False
PARENT_V6_RUNTIME_SHA256 = "fc3c394a76e8866baf40fe649a0ec5c78d0b857aa1307d3c2cbd9caf16961d6b"
PARENT_V6_1_RUNTIME_SHA256 = "220e487e0a62e232c4ba479e87abdace4d3df4515d2a94c647371a882a4bc748"
TEXT_GREEDY_LM_GEN_CONFIG = {
    "use_sampling": True,
    "temp": 0.8,
    "temp_text": 0.7,
    "top_k": 250,
    "top_k_text": 1,
}


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).replace("ي", "ی").replace("ك", "ک")
    value = re.sub(r"[^\w\u0600-\u06ff]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split()).strip().lower()


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def prefix_character_error_rate(generated: str, target: str) -> float:
    generated_normalized = normalize_text(generated)
    target_normalized = normalize_text(target)
    if not generated_normalized:
        return 1.0
    target_prefix = target_normalized[: len(generated_normalized)]
    denominator = max(len(generated_normalized), len(target_prefix), 1)
    return edit_distance(generated_normalized, target_prefix) / denominator


def target_for_row(row: dict[str, Any]) -> str:
    metadata_path = Path(str(row["path"])).resolve().with_suffix(".json")
    metadata = json_object(metadata_path)
    alignments = metadata.get("alignments")
    if (
        not isinstance(alignments, list)
        or len(alignments) != 1
        or not isinstance(alignments[0], list)
        or not alignments[0]
    ):
        raise ValueError(f"expected one target alignment: {metadata_path}")
    target = str(alignments[0][0]).strip()
    if not target:
        raise ValueError(f"empty target text: {metadata_path}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/moshi_h100_v6_overfit.yaml"),
    )
    parser.add_argument(
        "--reevaluation",
        type=Path,
        default=Path("results/moshi_v6_overfit_reevaluation.json"),
    )
    parser.add_argument(
        "--data-report", type=Path, default=Path("results/moshi_v6_overfit_data.json")
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path("results/hardware/moshi_v6_overfit_preflight.json"),
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path("results/hardware/moshi_v6_overfit_probe.json"),
    )
    parser.add_argument("--protocol", type=Path, default=Path("docs/MOSHI_V6_OVERFIT_PROTOCOL.md"))
    parser.add_argument("--policy", type=Path, default=Path("configs/moshi_v6_overfit_policy.json"))
    parser.add_argument(
        "--client-report",
        type=Path,
        default=Path("results/hardware/moshi_client_build.json"),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--text-greedy-followup",
        action="store_true",
        help="Run the frozen v6.1 top-k-text=1 follow-up against unchanged v6 weights.",
    )
    parser.add_argument(
        "--text-dropout-followup",
        action="store_true",
        help="Run the frozen v6.2 scheduled text-input-dropout diagnostic.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18998)
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--response-timeout", type=float, default=20.0)
    parser.add_argument("--input-seconds", type=float, default=5.0)
    parser.add_argument("--post-input-silence-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("v6 diagnostic runtime is restricted to loopback")
    if args.text_greedy_followup and args.text_dropout_followup:
        raise ValueError("v6.1 and v6.2 follow-up modes are mutually exclusive")
    if args.text_dropout_followup:
        args.training_config = Path("configs/moshi_h100_v6_text_dropout.yaml")
        args.reevaluation = Path("results/moshi_v6_text_dropout_reevaluation.json")
        args.preflight = Path("results/hardware/moshi_v6_text_dropout_preflight.json")
        args.probe = Path("results/hardware/moshi_v6_text_dropout_probe.json")
        args.protocol = Path("docs/MOSHI_V6_TEXT_DROPOUT_PROTOCOL.md")
        args.policy = Path("configs/moshi_v6_text_dropout_policy.json")
        args.out = Path("results/moshi_v6_text_dropout_runtime.json")

    training_config_path = (ROOT / args.training_config).resolve()
    reevaluation_path = (ROOT / args.reevaluation).resolve()
    data_report_path = (ROOT / args.data_report).resolve()
    preflight_path = (ROOT / args.preflight).resolve()
    probe_path = (ROOT / args.probe).resolve()
    protocol_path = (
        ROOT
        / (
            Path("docs/MOSHI_V6_TEXT_GREEDY_PROTOCOL.md")
            if args.text_greedy_followup
            else args.protocol
        )
    ).resolve()
    policy_path = (ROOT / args.policy).resolve()
    client_report_path = (ROOT / args.client_report).resolve()
    default_out = (
        Path("results/moshi_v6_text_greedy_runtime.json")
        if args.text_greedy_followup
        else Path("results/moshi_v6_overfit_runtime.json")
    )
    out_path = (ROOT / (args.out or default_out)).resolve()
    runtime_config_path = (
        (ROOT / "configs/moshi_v6_text_greedy_runtime.json").resolve()
        if args.text_greedy_followup or args.text_dropout_followup
        else None
    )
    parent_runtime_path = (
        (
            ROOT
            / (
                "results/moshi_v6_text_greedy_runtime.json"
                if args.text_dropout_followup
                else "results/moshi_v6_overfit_runtime.json"
            )
        ).resolve()
        if args.text_greedy_followup or args.text_dropout_followup
        else None
    )
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v6 runtime evidence: {out_path}")

    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if not isinstance(training_config, dict):
        raise ValueError("training config must be a YAML mapping")
    reevaluation = json_object(reevaluation_path)
    data_report = json_object(data_report_path)
    preflight = json_object(preflight_path)
    probe = json_object(probe_path)
    policy = json_object(policy_path)
    client_report = json_object(client_report_path)
    data_config = training_config.get("data") or {}
    moshi_paths = training_config.get("moshi_paths") or {}
    lora_config = training_config.get("lora") or {}
    training_manifest = project_path(ROOT, data_config.get("train_data"), "train_data")
    rows = load_jsonl(training_manifest)
    if data_config.get("eval_data") != data_config.get("train_data"):
        raise ValueError("v6 diagnostic requires identical train/in-sample eval manifests")
    if len(rows) != 32:
        raise ValueError(f"v6 diagnostic requires 32 rows, found {len(rows)}")

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
    parent_runtime: dict[str, Any] | None = None
    followup_predeclared = True
    if args.text_greedy_followup or args.text_dropout_followup:
        if runtime_config_path is None or parent_runtime_path is None:
            raise AssertionError("follow-up paths were not resolved")
        runtime_config = json_object(runtime_config_path)
        generation_config = runtime_config.pop("lm_gen_config", None)
        if runtime_config != expected_config:
            raise ValueError("follow-up runtime config differs from the frozen model config")
        if generation_config != TEXT_GREEDY_LM_GEN_CONFIG:
            raise ValueError("follow-up generation config differs from the frozen intervention")
        parent_runtime = json_object(parent_runtime_path)
        expected_parent_sha256 = (
            PARENT_V6_1_RUNTIME_SHA256 if args.text_dropout_followup else PARENT_V6_RUNTIME_SHA256
        )
        followup_predeclared = (
            protocol_path.is_file()
            and sha256_file(parent_runtime_path) == expected_parent_sha256
            and parent_runtime.get("status") == "passed"
            and parent_runtime.get("diagnostic_positive") is False
            and parent_runtime.get("diagnostic_outcome") == "negative"
            and parent_runtime.get("final_test_accessed") is False
            and len(parent_runtime.get("candidates") or []) == 4
            and all(
                candidate.get("official_server_runtime_passes") is True
                for candidate in parent_runtime.get("candidates") or []
            )
            and (
                not args.text_dropout_followup
                or (
                    preflight.get("status") == "passed"
                    and preflight.get("preflight_passes") is True
                    and (preflight.get("artifacts") or {}).get("protocol_sha256")
                    == sha256_file(protocol_path)
                )
            )
        )
    expected_schema = expected_adapter_schema(
        expected_config,
        ft_embed=False,
        embedding_names=FULL_TEXT_PARAMETERS,
    )
    client_dist = client_report.get("dist") or {}
    static_dir = (ROOT / str(client_dist.get("path"))).resolve()
    static_files, static_tree_hash = tree_manifest(static_dir)
    reevaluated_candidates = reevaluation.get("candidates")
    if not isinstance(reevaluated_candidates, list):
        raise ValueError("v6 reevaluation has no candidate list")
    candidates_by_step = {
        int(candidate["step"]): candidate
        for candidate in reevaluated_candidates
        if isinstance(candidate, dict)
    }
    expected_steps = [50, 100, 150, 200]

    preconditions = {
        "protocol_predeclared": (
            followup_predeclared
            if args.text_greedy_followup or args.text_dropout_followup
            else (
                protocol_path.is_file()
                and preflight.get("status") == "passed"
                and preflight.get("preflight_passes") is True
                and (preflight.get("artifacts") or {}).get("protocol_sha256")
                == sha256_file(protocol_path)
            )
        ),
        "probe_passed": (
            probe.get("status") == "passed"
            and probe.get("probe_passes") is True
            and probe.get("final_test_accessed") is False
        ),
        "data_freeze_current": (
            data_report.get("passes") is True
            and data_report.get("final_test_accessed") is False
            and (data_report.get("output") or {}).get("manifest_sha256")
            == sha256_file(training_manifest)
        ),
        "reevaluation_complete_and_in_sample": (
            reevaluation.get("status") == "passed"
            and reevaluation.get("reevaluation_passes") is True
            and reevaluation.get("heldout_test_used") is False
            and sorted(candidates_by_step) == expected_steps
            and all(
                candidate.get("sample_count") == 32
                and candidate.get("validation_manifest_sha256") == sha256_file(training_manifest)
                for candidate in candidates_by_step.values()
            )
        ),
        "policy_current": (
            policy.get("mode")
            == (
                "persian_text_head_adaptation_with_scheduled_text_input_dropout"
                if args.text_dropout_followup
                else "persian_text_head_adaptation"
            )
            and set(policy.get("trainable_full_parameters") or []) == FULL_TEXT_PARAMETERS
            and (preflight.get("artifacts") or {}).get("policy_sha256") == sha256_file(policy_path)
        ),
        "panel_exact": len(PANEL_INDICES) == 9
        and max(PANEL_INDICES) < len(rows)
        and all(PANEL_INDICES[index] < PANEL_INDICES[index + 1] for index in range(8)),
        "client_build_current": (
            client_report.get("valid") is True
            and static_tree_hash == client_dist.get("tree_sha256")
            and len(static_files) == len(client_dist.get("files") or [])
        ),
        "final_test_not_an_input": "test" not in training_manifest.as_posix().lower(),
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"v6 runtime preconditions failed: {preconditions}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "scientific_validation_evidence": False,
        "generalization_claim_allowed": False,
        "final_test_accessed": False,
        "followup": {
            "enabled": args.text_greedy_followup or args.text_dropout_followup,
            "intervention": (
                "scheduled_text_input_dropout_0_25_to_0_75"
                if args.text_dropout_followup
                else "text_top_k_25_to_1"
                if args.text_greedy_followup
                else None
            ),
            "weights_reused_without_training": args.text_greedy_followup,
            "parent_diagnostic_sha256": (
                PARENT_V6_1_RUNTIME_SHA256
                if args.text_dropout_followup
                else PARENT_V6_RUNTIME_SHA256
                if args.text_greedy_followup
                else None
            ),
        },
        "protocol": {
            "panel_indices": list(PANEL_INDICES),
            "panel_scope": "same 32 train-only rows used by optimization",
            "all_nine_rows_must_pass": True,
            "speech_rms_threshold": SPEECH_RMS_THRESHOLD,
            "persian_letter_fraction_threshold": PERSIAN_LETTER_FRACTION_THRESHOLD,
            "maximum_target_prefix_character_error_rate": MAX_PREFIX_CER,
            "replacement_character_allowed": False,
            "minimum_text_loss_reduction_from_step_50": 0.30,
            "official_server_fuse_lora": FUSE_LORA_AT_RUNTIME,
        },
        "preconditions": preconditions,
        "candidates": [],
        "selection": None,
        "artifacts": {
            "training_config_sha256": sha256_file(training_config_path),
            "reevaluation_sha256": sha256_file(reevaluation_path),
            "data_report_sha256": sha256_file(data_report_path),
            "preflight_sha256": sha256_file(preflight_path),
            "probe_sha256": sha256_file(probe_path),
            "protocol_sha256": sha256_file(protocol_path),
            "policy_sha256": sha256_file(policy_path),
            "training_manifest_sha256": sha256_file(training_manifest),
            "base_model_sha256": sha256_file(base_model_path),
            "mimi_sha256": sha256_file(mimi_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "runtime_config_sha256": (
                sha256_file(runtime_config_path) if runtime_config_path is not None else None
            ),
            "parent_runtime_sha256": (
                sha256_file(parent_runtime_path) if parent_runtime_path is not None else None
            ),
        },
    }
    write_report(out_path, report)

    run_dir = (ROOT / str(training_config["run_dir"])).resolve()
    for step in expected_steps:
        candidate = candidates_by_step[step]
        consolidated = run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"
        adapter_path = project_path(
            ROOT,
            (consolidated / "lora.safetensors").relative_to(ROOT).as_posix(),
            f"candidate {step} adapter",
        )
        config_path = project_path(
            ROOT,
            (consolidated / "config.json").relative_to(ROOT).as_posix(),
            f"candidate {step} config",
        )
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
            validation_manifest=training_manifest,
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
            server_config_path=runtime_config_path,
        )
        for panel_row in runtime["panel"]:
            manifest_index = int(panel_row["validation_index"])
            target = target_for_row(rows[manifest_index])
            stream = panel_row.get("stream") or {}
            generated = str(stream.get("generated_text") or "")
            normalized_generated = normalize_text(generated)
            persian_fraction = float(
                (stream.get("generated_text_script") or {}).get("persian_letter_fraction") or 0.0
            )
            prefix_cer = prefix_character_error_rate(generated, target)
            panel_row["target"] = {
                "sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                "normalized_length": len(normalize_text(target)),
                "retained": False,
            }
            panel_row["target_prefix_character_error_rate"] = prefix_cer
            panel_row["normalized_generated_text_length"] = len(normalized_generated)
            panel_row["requirements"]["normalized_text_length_at_least_4"] = (
                len(normalized_generated) >= MIN_NORMALIZED_TEXT_LENGTH
            )
            panel_row["requirements"]["persian_letter_fraction_at_least_0_80"] = (
                persian_fraction >= PERSIAN_LETTER_FRACTION_THRESHOLD
            )
            panel_row["requirements"]["no_unicode_replacement_character"] = "�" not in generated
            panel_row["requirements"]["target_prefix_cer_at_most_0_75"] = (
                prefix_cer <= MAX_PREFIX_CER
            )
            panel_row["passes"] = all(panel_row["requirements"].values())
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
        runtime["config_path"] = config_path.relative_to(ROOT).as_posix()
        report["candidates"].append(runtime)
        write_report(out_path, report)

    runtime_positive = [
        candidate
        for candidate in report["candidates"]
        if candidate["candidate_runtime_passes"] is True
    ]
    selected = (
        min(runtime_positive, key=lambda candidate: int(candidate["step"]))
        if runtime_positive
        else None
    )
    step_50 = next(candidate for candidate in report["candidates"] if candidate["step"] == 50)
    loss_rule_passes = False
    if selected is not None:
        loss_rule_passes = selected["step"] == 50 or selected["text_eval_loss"] <= (
            0.70 * step_50["text_eval_loss"]
        )
    diagnostic_positive = selected is not None and loss_rule_passes
    report["selection"] = (
        {
            "step": selected["step"],
            "adapter_path": selected["adapter_path"],
            "config_path": selected["config_path"],
            "rule": "earliest all-nine runtime pass plus frozen text-loss rule",
            "text_loss_rule_passes": loss_rule_passes,
        }
        if selected is not None
        else None
    )
    completion_requirements = {
        **preconditions,
        "all_candidates_evaluated": [candidate["step"] for candidate in report["candidates"]]
        == expected_steps,
        "all_static_adapter_checks_passed": all(
            candidate["static_adapter_validation"]["passes"] is True
            for candidate in report["candidates"]
        ),
        "all_official_servers_exercised": all(
            candidate["official_server_runtime_passes"] is True
            for candidate in report["candidates"]
        ),
    }
    report["requirements"] = completion_requirements
    report["diagnostic_positive"] = diagnostic_positive
    report["diagnostic_outcome"] = "positive" if diagnostic_positive else "negative"
    report["status"] = "passed" if all(completion_requirements.values()) else "failed"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_report(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("v6 diagnostic runtime evaluation was incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
