import hashlib
import json
from pathlib import Path

import pytest

from scripts.select_moshi_checkpoint import select_checkpoint


def _checkpoint(run_dir: Path, step: int) -> None:
    consolidated = run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"
    consolidated.mkdir(parents=True)
    (consolidated / "lora.safetensors").write_bytes(f"adapter-{step}".encode())
    (consolidated / "config.json").write_text(
        json.dumps({"step": step}),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reevaluation(run_dir: Path, path: Path, rows: list[dict], *, max_steps: int) -> None:
    candidates = []
    for row in rows:
        step = int(row["step"])
        consolidated = run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"
        candidates.append(
            {
                **row,
                "sample_count": 12,
                "validation_scope": "complete_fixed_manifest",
                "validation_manifest_sha256": "f" * 64,
                "heldout_test_used": False,
                "adapter_sha256": _sha256(consolidated / "lora.safetensors"),
                "config_sha256": _sha256(consolidated / "config.json"),
            }
        )
    report = {
        "schema_version": 1,
        "status": "passed",
        "reevaluation_passes": True,
        "heldout_test_used": False,
        "checkpoint_selection_performed": False,
        "selection_criterion_changed": False,
        "selection_input_corrected_after_training": True,
        "original_metrics_eligible_for_selection": False,
        "configured_max_steps": max_steps,
        "checkpoint_frequency": 500,
        "candidates": candidates,
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_checkpoint_selection_uses_only_complete_checkpoint_aligned_eval(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("max_steps: 1000\nckpt_freq: 500\n", encoding="utf-8")
    _checkpoint(run_dir, 500)
    _checkpoint(run_dir, 1000)
    (run_dir / "metrics.train.jsonl").write_text(
        '{"step": 500, "loss": 2.1}\n{"step": 1000, "loss": 1.6}\n',
        encoding="utf-8",
    )
    rows = [
        {"step": 500, "eval_loss": 2.0, "text_eval_loss": 1.0, "audio_eval_loss": 3.0},
        {"step": 1000, "eval_loss": 1.5, "text_eval_loss": 0.8, "audio_eval_loss": 2.2},
    ]
    reevaluation = tmp_path / "reevaluation.json"
    _reevaluation(run_dir, reevaluation, rows, max_steps=1000)
    report = select_checkpoint(run_dir, config, tmp_path / "selection.json", reevaluation)
    assert report["candidate_count"] == 2
    assert report["selected"]["step"] == 1000
    assert report["selected"]["adapter_sha256"]
    assert report["heldout_test_used_for_selection"] is False
    assert report["original_upstream_metrics_eligible_for_selection"] is False


def test_checkpoint_selection_rejects_incomplete_run(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("max_steps: 1000\nckpt_freq: 500\n", encoding="utf-8")
    _checkpoint(run_dir, 500)
    (run_dir / "metrics.train.jsonl").write_text(
        '{"step": 500, "loss": 2.0}\n',
        encoding="utf-8",
    )
    reevaluation = tmp_path / "reevaluation.json"
    _reevaluation(run_dir, reevaluation, [{"step": 500, "eval_loss": 2.0}], max_steps=1000)
    with pytest.raises(RuntimeError, match="ineligible"):
        select_checkpoint(run_dir, config, tmp_path / "must-not-exist.json", reevaluation)
