#!/usr/bin/env python3
"""Evaluate every Moshi v5 checkpoint on the predeclared complete-prompt panel."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_moshi_client import tree_manifest  # noqa: E402
from scripts.evaluate_moshi_v2_runtime_candidates import (  # noqa: E402
    PANEL_INDICES,
    PERSIAN_LETTER_FRACTION_THRESHOLD,
    SPEECH_RMS_THRESHOLD,
    candidate_static_validation,
    load_jsonl,
    run_candidate_server,
    write_report,
)
from scripts.moshi_runtime_panel import complete_prompt_panel  # noqa: E402
from scripts.validate_moshi_adapter import (  # noqa: E402
    expected_adapter_schema,
    json_object,
    project_path,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/moshi_h100_v5.yaml"),
    )
    parser.add_argument(
        "--reevaluation",
        type=Path,
        default=Path("results/moshi_v5_validation_reevaluation.json"),
    )
    parser.add_argument(
        "--split-report",
        type=Path,
        default=Path("results/moshi_v2_split.json"),
    )
    parser.add_argument(
        "--client-report",
        type=Path,
        default=Path("results/hardware/moshi_client_build.json"),
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path("results/hardware/moshi_v5_preflight.json"),
    )
    parser.add_argument(
        "--profile-probe",
        type=Path,
        default=Path("results/hardware/moshi_h100_v5_profile_probe.json"),
    )
    parser.add_argument(
        "--protocol-document",
        type=Path,
        default=Path("docs/MOSHI_V5_SELECTION_PROTOCOL.md"),
    )
    parser.add_argument(
        "--embedding-policy",
        type=Path,
        default=Path("configs/moshi_v5_embedding_policy.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/moshi_v5_runtime_candidates.json"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18998)
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--response-timeout", type=float, default=20.0)
    parser.add_argument("--input-seconds", type=float, default=5.0)
    parser.add_argument("--post-input-silence-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("candidate evaluation is restricted to a loopback host")

    training_config_path = (ROOT / args.training_config).resolve()
    reevaluation_path = (ROOT / args.reevaluation).resolve()
    split_report_path = (ROOT / args.split_report).resolve()
    client_report_path = (ROOT / args.client_report).resolve()
    preflight_path = (ROOT / args.preflight).resolve()
    profile_probe_path = (ROOT / args.profile_probe).resolve()
    protocol_path = (ROOT / args.protocol_document).resolve()
    embedding_policy_path = (ROOT / args.embedding_policy).resolve()
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v5 runtime evidence: {out_path}")

    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if not isinstance(training_config, dict):
        raise ValueError("training config must be a mapping")
    reevaluation = json_object(reevaluation_path)
    split_report = json_object(split_report_path)
    client_report = json_object(client_report_path)
    preflight = json_object(preflight_path)
    profile_probe = json_object(profile_probe_path)
    embedding_policy = json_object(embedding_policy_path)
    declared_embedding_names = embedding_policy.get("trainable_embedding_parameters")
    if not isinstance(declared_embedding_names, list) or not all(
        isinstance(name, str) and name for name in declared_embedding_names
    ):
        raise ValueError("v5 embedding policy has no valid trainable embedding list")
    embedding_names = set(declared_embedding_names)
    data_config = training_config.get("data")
    moshi_paths = training_config.get("moshi_paths")
    lora_config = training_config.get("lora")
    if not all(isinstance(value, dict) for value in (data_config, moshi_paths, lora_config)):
        raise ValueError("training config lacks data, moshi_paths, or lora mappings")
    assert isinstance(data_config, dict)
    assert isinstance(moshi_paths, dict)
    assert isinstance(lora_config, dict)

    validation_manifest = project_path(ROOT, data_config["eval_data"], "data.eval_data")
    validation_rows = load_jsonl(validation_manifest)
    derived_panel_indices, complete_prompt_rows = complete_prompt_panel(
        validation_rows,
        input_seconds=args.input_seconds,
    )
    split_validation = (split_report.get("outputs") or {}).get("validation") or {}
    client_dist = client_report.get("dist") or {}
    static_dir = (ROOT / str(client_dist.get("path"))).resolve()
    static_files, static_tree_hash = tree_manifest(static_dir)
    base_config_path = project_path(ROOT, moshi_paths["config_path"], "config_path")
    base_model_path = project_path(ROOT, moshi_paths["moshi_path"], "moshi_path")
    mimi_path = project_path(ROOT, moshi_paths["mimi_path"], "mimi_path")
    tokenizer_path = project_path(ROOT, moshi_paths["tokenizer_path"], "tokenizer_path")
    expected_config = json_object(base_config_path)
    expected_config.update(
        {
            "lora": lora_config.get("enable"),
            "lora_rank": lora_config.get("rank"),
            "lora_scaling": lora_config.get("scaling"),
        }
    )
    expected_schema = expected_adapter_schema(
        expected_config,
        ft_embed=lora_config.get("ft_embed") is True,
        embedding_names=embedding_names,
    )

    checkpoint_frequency = int(training_config["ckpt_freq"])
    max_steps = int(training_config["max_steps"])
    expected_steps = list(range(checkpoint_frequency, max_steps + 1, checkpoint_frequency))
    run_dir = (ROOT / str(training_config["run_dir"])).resolve()
    reevaluated_candidates = reevaluation.get("candidates")
    if not isinstance(reevaluated_candidates, list):
        raise ValueError("reevaluation report has no candidate list")
    candidate_by_step = {
        int(candidate["step"]): candidate
        for candidate in reevaluated_candidates
        if isinstance(candidate, dict)
    }
    preflight_artifacts = preflight.get("artifacts") or {}
    probe_artifacts = profile_probe.get("artifacts") or {}
    preconditions = {
        "protocol_frozen_before_v5_training": (
            protocol_path.is_file()
            and preflight.get("launch_commit")
            and preflight.get("test_access_started") is False
            and preflight_artifacts.get("protocol_sha256") == sha256_file(protocol_path)
        ),
        "preflight_passed_and_bound_to_config": (
            preflight.get("status") == "passed"
            and preflight.get("preflight_passes") is True
            and preflight_artifacts.get("config_sha256") == sha256_file(training_config_path)
        ),
        "exact_shape_probe_passed_and_bound_to_config": (
            profile_probe.get("status") == "passed"
            and profile_probe.get("full_profile_gate_passes") is True
            and probe_artifacts.get("full_config_sha256") == sha256_file(training_config_path)
        ),
        "split_report_passed": split_report.get("split_passes") is True,
        "validation_manifest_matches_frozen_split": (
            split_validation.get("path") == validation_manifest.relative_to(ROOT).as_posix()
            and split_validation.get("sha256") == sha256_file(validation_manifest)
            and split_validation.get("rows") == len(validation_rows)
        ),
        "complete_prompt_panel_indices_exact": derived_panel_indices == PANEL_INDICES,
        "complete_prompt_candidate_count_exact": len(complete_prompt_rows) == 17,
        "panel_indices_in_range": all(index < len(validation_rows) for index in PANEL_INDICES),
        "all_panel_prompts_complete_within_stream": all(
            row["user_audio_end_seconds"] <= args.input_seconds
            for row in complete_prompt_rows
            if row["manifest_index"] in PANEL_INDICES
        ),
        "v5_profile_exact": (
            lora_config == {"enable": True, "rank": 128, "scaling": 2.0, "ft_embed": False}
            and max_steps == 500
            and checkpoint_frequency == 100
            and int(training_config["num_ckpt_keep"]) == 5
            and training_config.get("first_codebook_weight_multiplier") == 10.0
        ),
        "v5_selective_embedding_policy_exact": (
            embedding_policy.get("mode") == "text_embeddings_only"
            and embedding_names == {"depformer_text_emb.weight", "text_emb.weight"}
            and embedding_policy.get("expected_lora_tensor_count") == 674
            and embedding_policy.get("expected_total_adapter_tensor_count") == 676
            and (embedding_policy.get("objective_change") or {}).get("baseline_value") == 100.0
            and (embedding_policy.get("objective_change") or {}).get("v5_value") == 10.0
            and (embedding_policy.get("launcher_environment") or {})
            == {"name": "MOSHI_TEXT_EMBEDDINGS_ONLY", "required_value": "1"}
            and (
                preflight_artifacts.get("embedding_policy_sha256")
                == sha256_file(embedding_policy_path)
            )
            and (
                (profile_probe.get("embedding_policy") or {}).get("sha256")
                == sha256_file(embedding_policy_path)
            )
        ),
        "reevaluation_passed": (
            reevaluation.get("status") == "passed"
            and reevaluation.get("reevaluation_passes") is True
            and reevaluation.get("heldout_test_used") is False
        ),
        "all_candidates_present": sorted(candidate_by_step) == expected_steps,
        "client_build_valid": client_report.get("valid") is True,
        "client_tree_hash_current": static_tree_hash == client_dist.get("tree_sha256"),
        "client_file_count_current": len(static_files) == len(client_dist.get("files") or []),
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"v5 runtime-panel preconditions failed: {preconditions}")

    report: dict[str, Any] = {
        "schema_version": 5,
        "status": "running",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_validation_evidence": True,
        "human_perceptual_evidence": False,
        "selection_performed": False,
        "protocol_predeclared_before_training": True,
        "protocol_provenance": {
            "document": protocol_path.relative_to(ROOT).as_posix(),
            "document_sha256": sha256_file(protocol_path),
            "inherited_complete_prompt_rule_from_v2": True,
            "frozen_before_v5_optimizer_step": True,
            "selection_criterion_changed_after_training": False,
            "eligibility_thresholds_changed_after_training": False,
        },
        "protocol": {
            "panel_indices": list(PANEL_INDICES),
            "panel_rule": (
                "retain validation rows whose last non-zero user-channel PCM sample is at "
                "or before input_seconds, then select nine floor-spaced manifest-ordered members"
            ),
            "complete_prompt_candidate_count": len(complete_prompt_rows),
            "complete_user_turn_required": True,
            "input_seconds": args.input_seconds,
            "speech_energy_threshold_rms": SPEECH_RMS_THRESHOLD,
            "persian_letter_fraction_threshold": PERSIAN_LETTER_FRACTION_THRESHOLD,
            "all_panel_rows_must_pass": True,
            "generated_audio_retained": False,
            "generated_text_retained": True,
        },
        "preconditions": preconditions,
        "candidates": [],
        "artifacts": {
            "training_config": training_config_path.relative_to(ROOT).as_posix(),
            "training_config_sha256": sha256_file(training_config_path),
            "reevaluation": reevaluation_path.relative_to(ROOT).as_posix(),
            "reevaluation_sha256": sha256_file(reevaluation_path),
            "split_report_sha256": sha256_file(split_report_path),
            "preflight_sha256": sha256_file(preflight_path),
            "profile_probe_sha256": sha256_file(profile_probe_path),
            "protocol_sha256": sha256_file(protocol_path),
            "embedding_policy_sha256": sha256_file(embedding_policy_path),
            "client_report_sha256": sha256_file(client_report_path),
            "client_tree_sha256": static_tree_hash,
            "base_model_sha256": sha256_file(base_model_path),
            "mimi_sha256": sha256_file(mimi_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "server_entry_sha256": sha256_file(ROOT / "scripts/moshi_server_entry.py"),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_report(out_path, report)
    for candidate_index, step in enumerate(expected_steps, start=1):
        candidate = candidate_by_step[step]
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
        print(
            f"v5 runtime panel candidate {candidate_index}/{len(expected_steps)}: step={step}",
            flush=True,
        )
        static_validation = candidate_static_validation(
            candidate=candidate,
            adapter_path=adapter_path,
            config_path=config_path,
            expected_config=expected_config,
            expected_schema=expected_schema,
            ft_embed=False,
            embedding_names=embedding_names,
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
            validation_manifest=validation_manifest,
            validation_rows=validation_rows,
            host=args.host,
            port=args.port,
            ready_timeout=args.ready_timeout,
            response_timeout=args.response_timeout,
            input_seconds=args.input_seconds,
            post_input_silence_seconds=args.post_input_silence_seconds,
        )
        runtime["static_adapter_validation"] = static_validation
        runtime["step"] = step
        runtime["adapter_path"] = adapter_path.relative_to(ROOT).as_posix()
        runtime["config_path"] = config_path.relative_to(ROOT).as_posix()
        runtime["candidate_runtime_passes"] = (
            static_validation["passes"] is True and runtime["candidate_runtime_passes"] is True
        )
        report["candidates"].append(runtime)
        write_report(out_path, report)

    evaluated_steps = [row["step"] for row in report["candidates"]]
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["candidate_count"] = len(report["candidates"])
    report["eligible_steps"] = [
        row["step"] for row in report["candidates"] if row["candidate_runtime_passes"] is True
    ]
    report["requirements"] = {
        **preconditions,
        "all_candidate_steps_evaluated_once": evaluated_steps == expected_steps,
        "all_reports_have_exact_panel_size": all(
            row["panel_count"] == len(PANEL_INDICES) for row in report["candidates"]
        ),
    }
    report["runtime_panel_evaluation_passes"] = all(report["requirements"].values())
    report["status"] = "passed" if report["runtime_panel_evaluation_passes"] else "failed"
    write_report(out_path, report)
    if report["runtime_panel_evaluation_passes"] is not True:
        raise RuntimeError("v5 runtime-panel evaluation was incomplete")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
