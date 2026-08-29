#!/usr/bin/env python3
"""Run selected Moshi v2 on a deterministic final-test server panel."""

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
    run_candidate_server,
)
from scripts.validate_moshi_adapter import (  # noqa: E402
    json_object,
    project_path,
    sha256_file,
)


def evenly_spaced_indices(row_count: int, panel_size: int = 9) -> tuple[int, ...]:
    if row_count < panel_size or panel_size < 2:
        raise ValueError("row_count must cover a panel of at least two rows")
    return tuple(index * (row_count - 1) // (panel_size - 1) for index in range(panel_size))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("results/moshi_v2_adapter_validation.json"),
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/moshi_h100_v2.yaml"),
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
        default=Path("results/moshi_v2_final_runtime.json"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18998)
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--response-timeout", type=float, default=20.0)
    parser.add_argument("--input-seconds", type=float, default=5.0)
    parser.add_argument("--post-input-silence-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("final runtime evaluation is restricted to loopback")

    validation_path = (ROOT / args.validation).resolve()
    training_config_path = (ROOT / args.training_config).resolve()
    split_report_path = (ROOT / args.split_report).resolve()
    client_report_path = (ROOT / args.client_report).resolve()
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(
            f"refusing to repeat the v2 final-test runtime diagnostic: {out_path}"
        )
    validation = json_object(validation_path)
    split_report = json_object(split_report_path)
    client_report = json_object(client_report_path)
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if not isinstance(training_config, dict):
        raise ValueError("training config must be a mapping")
    adapter = validation.get("adapter")
    adapter_config = validation.get("config")
    selection_info = validation.get("selection")
    runtime_load = validation.get("runtime_load")
    if not all(
        isinstance(value, dict) for value in (adapter, adapter_config, selection_info, runtime_load)
    ):
        raise ValueError("v2 adapter validation report is incomplete")
    assert isinstance(adapter, dict)
    assert isinstance(adapter_config, dict)
    assert isinstance(selection_info, dict)
    assert isinstance(runtime_load, dict)
    adapter_path = project_path(ROOT, adapter.get("path"), "selected adapter")
    adapter_config_path = project_path(
        ROOT,
        adapter_config.get("path"),
        "selected adapter config",
    )
    selection_path = project_path(ROOT, selection_info.get("path"), "v2 selection")
    selection = json_object(selection_path)
    split_outputs = split_report.get("outputs")
    if not isinstance(split_outputs, dict):
        raise ValueError("v2 split report has no outputs")
    final_test = split_outputs.get("final_test")
    if not isinstance(final_test, dict):
        raise ValueError("v2 split report has no final-test output")
    test_manifest = project_path(
        ROOT,
        final_test.get("path"),
        "v2 final-test manifest",
    )
    test_rows = load_jsonl(test_manifest)
    panel_indices = evenly_spaced_indices(len(test_rows))

    moshi_paths = training_config.get("moshi_paths")
    if not isinstance(moshi_paths, dict):
        raise ValueError("training config has no moshi_paths mapping")
    base_model_path = project_path(ROOT, moshi_paths.get("moshi_path"), "base model")
    mimi_path = project_path(ROOT, moshi_paths.get("mimi_path"), "Mimi model")
    tokenizer_path = project_path(ROOT, moshi_paths.get("tokenizer_path"), "tokenizer")
    client_dist = client_report.get("dist") or {}
    static_dir = (ROOT / str(client_dist.get("path"))).resolve()
    static_files, static_tree_hash = tree_manifest(static_dir)
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("v2 selection has no selected candidate")

    preconditions = {
        "adapter_validation_passed": validation.get("validation_passes") is True,
        "official_cuda_loader_passed": (
            runtime_load.get("performed") is True and runtime_load.get("passed") is True
        ),
        "selection_passed": (
            selection.get("schema_version") == 3
            and selection.get("selection_passes") is True
            and selection.get("heldout_test_used_for_selection") is False
        ),
        "selection_hash_current": selection_info.get("sha256") == sha256_file(selection_path),
        "adapter_hash_current": (
            adapter.get("sha256") == selected.get("adapter_sha256") == sha256_file(adapter_path)
        ),
        "adapter_config_hash_current": (
            adapter_config.get("sha256")
            == selected.get("config_sha256")
            == sha256_file(adapter_config_path)
        ),
        "v2_split_passed": split_report.get("split_passes") is True,
        "final_test_manifest_current": (
            final_test.get("sha256") == sha256_file(test_manifest)
            and final_test.get("rows") == len(test_rows)
        ),
        "panel_rule_exact": panel_indices == (0, 92, 184, 276, 368, 460, 552, 644, 737),
        "client_build_valid": client_report.get("valid") is True,
        "client_tree_hash_current": static_tree_hash == client_dist.get("tree_sha256"),
        "client_file_count_current": len(static_files) == len(client_dist.get("files") or []),
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"v2 final-runtime preconditions failed: {preconditions}")

    runtime = run_candidate_server(
        step=int(selected["step"]),
        adapter_path=adapter_path,
        config_path=adapter_config_path,
        moshi_paths=moshi_paths,
        base_model_path=base_model_path,
        mimi_path=mimi_path,
        tokenizer_path=tokenizer_path,
        static_dir=static_dir,
        validation_manifest=test_manifest,
        validation_rows=test_rows,
        host=args.host,
        port=args.port,
        ready_timeout=args.ready_timeout,
        response_timeout=args.response_timeout,
        input_seconds=args.input_seconds,
        post_input_silence_seconds=args.post_input_silence_seconds,
        panel_indices=panel_indices,
        split_label="v2_final_test",
    )
    passed = all(preconditions.values()) and runtime["candidate_runtime_passes"] is True
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_evidence": True,
        "automatic_autoregressive_diagnostics_only": True,
        "used_for_checkpoint_selection": False,
        "human_perceptual_evidence": False,
        "panel_rule": {
            "method": "floor(i * (row_count - 1) / (panel_size - 1))",
            "panel_size": 9,
            "row_count": len(test_rows),
            "indices": list(panel_indices),
        },
        "preconditions": preconditions,
        "selected_step": selected["step"],
        "runtime": runtime,
        "final_runtime_passes": passed,
        "artifacts": {
            "adapter_validation_sha256": sha256_file(validation_path),
            "selection_sha256": sha256_file(selection_path),
            "split_report_sha256": sha256_file(split_report_path),
            "client_report_sha256": sha256_file(client_report_path),
            "adapter_sha256": sha256_file(adapter_path),
            "adapter_config_sha256": sha256_file(adapter_config_path),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "limitations": (
            "Automatic finite-audio, energy, text, and Persian-script diagnostics do not "
            "establish intelligibility, naturalness, relevance, or human preference."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("selected Moshi v2 failed final-test runtime diagnostics")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
