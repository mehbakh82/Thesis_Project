#!/usr/bin/env python3
"""Evaluate every Moshi v2 checkpoint on the frozen official-server panel."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_moshi_client import tree_manifest  # noqa: E402
from scripts.moshi_runtime_panel import complete_prompt_panel  # noqa: E402
from scripts.validate_moshi_adapter import (  # noqa: E402
    compare_adapter_schema,
    expected_adapter_schema,
    json_object,
    project_path,
    read_adapter_schema,
    sha256_file,
)
from scripts.validate_moshi_server_runtime import (  # noqa: E402
    ensure_port_available,
    exercise_server,
    listening_socket_report,
    percentile,
    process_gpu_memory_mib,
    strip_ansi,
    wait_for_http,
)

PANEL_INDICES = (0, 11, 33, 51, 55, 74, 85, 87, 106)
SUPERSEDED_PANEL_INDICES = (0, 16, 32, 48, 64, 80, 96, 112, 130)
SPEECH_RMS_THRESHOLD = 1e-3
PERSIAN_LETTER_FRACTION_THRESHOLD = 0.5


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def candidate_static_validation(
    *,
    candidate: dict[str, Any],
    adapter_path: Path,
    config_path: Path,
    expected_config: dict[str, Any],
    expected_schema: dict[str, dict[str, Any]],
    ft_embed: bool,
) -> dict[str, Any]:
    saved_config = json_object(config_path)
    actual_schema, all_values_finite, parameter_count = read_adapter_schema(adapter_path)
    schema_comparison = compare_adapter_schema(expected_schema, actual_schema)
    requirements = {
        "adapter_hash_matches_reevaluation": candidate.get("adapter_sha256")
        == sha256_file(adapter_path),
        "config_hash_matches_reevaluation": candidate.get("config_sha256")
        == sha256_file(config_path),
        "saved_config_matches_training": saved_config == expected_config,
        "all_adapter_values_finite": all_values_finite,
        "all_adapter_names_intended": all(
            "lora" in key or (ft_embed and "emb" in key) for key in actual_schema
        ),
        "adapter_schema_exact": schema_comparison["exact"] is True,
    }
    return {
        "adapter_bytes": adapter_path.stat().st_size,
        "adapter_sha256": sha256_file(adapter_path),
        "config_sha256": sha256_file(config_path),
        "tensor_count": len(actual_schema),
        "parameter_count": parameter_count,
        "schema_comparison": schema_comparison,
        "requirements": requirements,
        "passes": all(requirements.values()),
    }


def run_candidate_server(
    *,
    step: int,
    adapter_path: Path,
    config_path: Path,
    moshi_paths: dict[str, Any],
    base_model_path: Path,
    mimi_path: Path,
    tokenizer_path: Path,
    static_dir: Path,
    validation_manifest: Path,
    validation_rows: list[dict[str, Any]],
    host: str,
    port: int,
    ready_timeout: float,
    response_timeout: float,
    input_seconds: float,
    post_input_silence_seconds: float,
    panel_indices: tuple[int, ...] = PANEL_INDICES,
    split_label: str = "validation",
) -> dict[str, Any]:
    ensure_port_available(host, port)
    command = [
        sys.executable,
        "scripts/moshi_server_entry.py",
        "--host",
        host,
        "--port",
        str(port),
        "--static",
        str(static_dir),
        "--hf-repo",
        str(moshi_paths["hf_repo_id"]),
        "--moshi-weight",
        str(base_model_path),
        "--mimi-weight",
        str(mimi_path),
        "--tokenizer",
        str(tokenizer_path),
        "--config-path",
        str(config_path),
        "--lora-weight",
        str(adapter_path),
    ]
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    panel: list[dict[str, Any]] = []
    ready_seconds: float | None = None
    gpu_memory_mib: int | None = None
    socket_report: dict[str, Any] = {
        "output": "",
        "local_addresses": [],
        "loopback_only": False,
    }
    error: str | None = None
    log_path: Path | None = None
    process: subprocess.Popen[str] | None = None
    clean_output = ""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"moshi-v2-{step:06d}-",
        suffix=".log",
        delete=False,
    ) as log_handle:
        log_path = Path(log_handle.name)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            ready_seconds = wait_for_http(
                f"http://{host}:{port}/",
                process,
                ready_timeout,
            )
            socket_report = listening_socket_report(port)
            gpu_memory_mib = process_gpu_memory_mib(process.pid)
            for validation_index in panel_indices:
                row = validation_rows[validation_index]
                input_audio_path = Path(str(row["path"])).resolve()
                sample_error: str | None = None
                exercise: dict[str, Any] = {}
                try:
                    exercise = asyncio.run(
                        exercise_server(
                            host=host,
                            port=port,
                            static_index=static_dir / "index.html",
                            input_audio=input_audio_path,
                            input_seconds=input_seconds,
                            response_timeout=response_timeout,
                            post_input_silence_seconds=post_input_silence_seconds,
                        )
                    )
                except Exception as exc:  # fail closed while retaining all candidate evidence
                    sample_error = f"{type(exc).__name__}: {exc}"
                requirements = {
                    "input_audio_hash_current": input_audio_path.is_file()
                    and sha256_file(input_audio_path) == row.get("sha256"),
                    "exercise_completed": sample_error is None,
                    "complete_user_turn_streamed": exercise.get(
                        "complete_user_audio_streamed"
                    )
                    is True,
                    "static_bundle_served_exactly": (
                        exercise.get("static_http_status") == 200
                        and exercise.get("static_index_exact") is True
                    ),
                    "websocket_handshake_passed": exercise.get("websocket_handshake_valid") is True,
                    "reconnect_handshake_passed": exercise.get("reconnect_handshake_valid") is True,
                    "audio_messages_received": (
                        int(exercise.get("audio_messages") or 0) > 0
                        and int(exercise.get("audio_payload_bytes") or 0) > 0
                    ),
                    "decoded_audio_finite": (
                        int(exercise.get("decoded_audio_samples") or 0) > 0
                        and exercise.get("decoded_audio_finite") is True
                    ),
                    "speech_energy_observed": float(exercise.get("decoded_audio_rms") or 0.0)
                    >= SPEECH_RMS_THRESHOLD,
                    "text_tokens_observed": int(exercise.get("text_messages") or 0) > 0,
                    "persian_script_observed": float(
                        (exercise.get("generated_text_script") or {}).get("persian_letter_fraction")
                        or 0.0
                    )
                    >= PERSIAN_LETTER_FRACTION_THRESHOLD,
                    "no_unexpected_websocket_messages": int(
                        exercise.get("unexpected_messages") or 0
                    )
                    == 0,
                }
                panel.append(
                    {
                        "validation_index": validation_index,
                        "pair_id": input_audio_path.stem,
                        "input_audio_sha256": sha256_file(input_audio_path)
                        if input_audio_path.is_file()
                        else None,
                        "error": sample_error,
                        "stream": exercise,
                        "requirements": requirements,
                        "passes": all(requirements.values()),
                    }
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)

    if log_path is not None:
        clean_output = strip_ansi(log_path.read_text(encoding="utf-8"))
        log_path.unlink(missing_ok=True)
    frame_times_ms = [
        float(value) for value in re.findall(r"frame handled in ([0-9.]+)ms", clean_output)
    ]
    return_code = process.returncode if process is not None else None
    server_requirements = {
        "server_launch_completed": error is None,
        "server_process_exited_cleanly": return_code == 0,
        "server_bound_loopback_only": socket_report["loopback_only"] is True,
        "official_server_loaded_mimi_and_moshi": all(
            marker in clean_output
            for marker in ("mimi loaded", "moshi loaded", "warming up the model")
        ),
        "server_frame_processing_observed": bool(frame_times_ms),
        "real_model_gpu_allocation_observed": gpu_memory_mib is not None
        and gpu_memory_mib > 10_000,
        "no_cascade_or_tts_fallback_in_command": all(
            forbidden not in " ".join(command).lower() for forbidden in ("piper", "qwen", "formant")
        ),
        "all_frozen_panel_rows_exercised": [row["validation_index"] for row in panel]
        == list(panel_indices),
    }
    panel_pass_count = sum(row["passes"] is True for row in panel)
    server_passes = all(server_requirements.values())
    return {
        "error": error,
        "server": {
            "host": host,
            "port": port,
            "ready_seconds": ready_seconds,
            "process_gpu_memory_mib": gpu_memory_mib,
            "socket": socket_report,
            "exit_code": return_code,
            "frame_processing_ms": {
                "count": len(frame_times_ms),
                "mean": sum(frame_times_ms) / len(frame_times_ms) if frame_times_ms else None,
                "p95": percentile(frame_times_ms, 0.95) if frame_times_ms else None,
                "maximum": max(frame_times_ms) if frame_times_ms else None,
            },
        },
        "input": {
            "split": split_label,
            "manifest": validation_manifest.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(validation_manifest),
            "panel_indices": list(panel_indices),
            "audio_retained": False,
        },
        "panel": panel,
        "panel_pass_count": panel_pass_count,
        "panel_count": len(panel),
        "expected_panel_count": len(panel_indices),
        "server_requirements": server_requirements,
        "official_server_runtime_passes": server_passes,
        "all_panel_output_gates_pass": panel_pass_count == len(panel_indices),
        "candidate_runtime_passes": server_passes and panel_pass_count == len(panel_indices),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/moshi_h100_v2.yaml"),
    )
    parser.add_argument(
        "--reevaluation",
        type=Path,
        default=Path("results/moshi_v2_validation_reevaluation.json"),
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
        "--out",
        type=Path,
        default=Path("results/moshi_v2_runtime_candidates.json"),
    )
    parser.add_argument(
        "--superseded-runtime-report",
        type=Path,
        default=Path("results/moshi_v2_runtime_candidates_truncated_prompt_invalid.json"),
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
    out_path = (ROOT / args.out).resolve()
    superseded_runtime_path = (ROOT / args.superseded_runtime_report).resolve()
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if not isinstance(training_config, dict):
        raise ValueError("training config must be a mapping")
    reevaluation = json_object(reevaluation_path)
    split_report = json_object(split_report_path)
    client_report = json_object(client_report_path)
    data_config = training_config.get("data")
    moshi_paths = training_config.get("moshi_paths")
    lora_config = training_config.get("lora")
    if not all(isinstance(value, dict) for value in (data_config, moshi_paths, lora_config)):
        raise ValueError("training config lacks data, moshi_paths, or lora mappings")
    assert isinstance(data_config, dict)
    assert isinstance(moshi_paths, dict)
    assert isinstance(lora_config, dict)

    validation_manifest = project_path(
        ROOT,
        data_config["eval_data"],
        "data.eval_data",
    )
    validation_rows = load_jsonl(validation_manifest)
    derived_panel_indices, complete_prompt_rows = complete_prompt_panel(
        validation_rows,
        input_seconds=args.input_seconds,
    )
    superseded_runtime = json_object(superseded_runtime_path)
    split_validation = (split_report.get("outputs") or {}).get("validation") or {}
    client_dist = client_report.get("dist") or {}
    static_dir = (ROOT / str(client_dist.get("path"))).resolve()
    static_files, static_tree_hash = tree_manifest(static_dir)
    base_config_path = project_path(ROOT, moshi_paths["config_path"], "config_path")
    base_model_path = project_path(ROOT, moshi_paths["moshi_path"], "moshi_path")
    mimi_path = project_path(ROOT, moshi_paths["mimi_path"], "mimi_path")
    tokenizer_path = project_path(ROOT, moshi_paths["tokenizer_path"], "tokenizer_path")
    base_config = json_object(base_config_path)
    expected_config = dict(base_config)
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
    preconditions = {
        "split_report_passed": split_report.get("split_passes") is True,
        "validation_manifest_matches_frozen_split": (
            split_validation.get("path") == validation_manifest.relative_to(ROOT).as_posix()
            and split_validation.get("sha256") == sha256_file(validation_manifest)
            and split_validation.get("rows") == len(validation_rows)
        ),
        "corrected_panel_indices_exact": derived_panel_indices == PANEL_INDICES,
        "complete_prompt_candidate_count_exact": len(complete_prompt_rows) == 17,
        "panel_indices_in_range": all(index < len(validation_rows) for index in PANEL_INDICES),
        "all_corrected_panel_prompts_complete_within_stream": all(
            row["user_audio_end_seconds"] <= args.input_seconds
            for row in complete_prompt_rows
            if row["manifest_index"] in PANEL_INDICES
        ),
        "superseded_truncated_panel_preserved": (
            superseded_runtime.get("schema_version") == 1
            and (superseded_runtime.get("protocol") or {}).get("panel_indices")
            == list(SUPERSEDED_PANEL_INDICES)
            and superseded_runtime.get("eligible_steps") == []
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
        raise RuntimeError(f"v2 runtime-panel preconditions failed: {preconditions}")

    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_validation_evidence": True,
        "human_perceptual_evidence": False,
        "selection_performed": False,
        "protocol_correction": {
            "frozen_after_training_before_corrected_generation_and_final_test_access": True,
            "reason": (
                "the superseded panel streamed only the first five seconds even when the "
                "source user turn continued beyond that point"
            ),
            "selection_criterion_changed": False,
            "eligibility_thresholds_changed": False,
            "validation_inputs_changed_by_input_only_rule": True,
            "superseded_panel_indices": list(SUPERSEDED_PANEL_INDICES),
            "superseded_runtime_report": superseded_runtime_path.relative_to(ROOT).as_posix(),
            "superseded_runtime_sha256": sha256_file(superseded_runtime_path),
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
            "client_report_sha256": sha256_file(client_report_path),
            "client_tree_sha256": static_tree_hash,
            "base_model_sha256": sha256_file(base_model_path),
            "mimi_sha256": sha256_file(mimi_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "server_entry_sha256": sha256_file(ROOT / "scripts/moshi_server_entry.py"),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "superseded_runtime_sha256": sha256_file(superseded_runtime_path),
        },
    }
    write_report(out_path, report)
    for candidate_index, step in enumerate(expected_steps, start=1):
        candidate = candidate_by_step[step]
        adapter_path = project_path(
            ROOT,
            (
                run_dir
                / "checkpoints"
                / f"checkpoint_{step:06d}"
                / "consolidated"
                / "lora.safetensors"
            )
            .relative_to(ROOT)
            .as_posix(),
            f"candidate {step} adapter",
        )
        config_path = project_path(
            ROOT,
            (run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated" / "config.json")
            .relative_to(ROOT)
            .as_posix(),
            f"candidate {step} config",
        )
        print(
            f"runtime panel candidate {candidate_index}/{len(expected_steps)}: step={step}",
            flush=True,
        )
        static_validation = candidate_static_validation(
            candidate=candidate,
            adapter_path=adapter_path,
            config_path=config_path,
            expected_config=expected_config,
            expected_schema=expected_schema,
            ft_embed=lora_config.get("ft_embed") is True,
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
    report["status"] = "passed"
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
    report["status"] = "passed" if report["runtime_panel_evaluation_passes"] is True else "failed"
    write_report(out_path, report)
    if report["runtime_panel_evaluation_passes"] is not True:
        raise RuntimeError("v2 runtime-panel evaluation was incomplete")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
