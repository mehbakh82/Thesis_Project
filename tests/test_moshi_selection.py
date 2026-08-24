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


def test_checkpoint_selection_uses_only_complete_checkpoint_aligned_eval(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("max_steps: 1000\nckpt_freq: 500\n", encoding="utf-8")
    _checkpoint(run_dir, 500)
    _checkpoint(run_dir, 1000)
    rows = [
        {"step": 250, "eval_loss": 3.0},
        {"step": 500, "eval_loss": 2.0, "text_eval_loss": 1.0, "audio_eval_loss": 3.0},
        {"step": 750, "eval_loss": 1.0},
        {"step": 1000, "eval_loss": 1.5, "text_eval_loss": 0.8, "audio_eval_loss": 2.2},
    ]
    (run_dir / "metrics.eval.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = select_checkpoint(run_dir, config, tmp_path / "selection.json")
    assert report["candidate_count"] == 2
    assert report["selected"]["step"] == 1000
    assert report["selected"]["adapter_sha256"]
    assert report["heldout_test_used_for_selection"] is False


def test_checkpoint_selection_rejects_incomplete_run(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("max_steps: 1000\nckpt_freq: 500\n", encoding="utf-8")
    _checkpoint(run_dir, 500)
    (run_dir / "metrics.eval.jsonl").write_text(
        json.dumps({"step": 500, "eval_loss": 2.0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        select_checkpoint(run_dir, config, tmp_path / "must-not-exist.json")
