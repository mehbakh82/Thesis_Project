from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts.evaluate_moshi_v2_final_runtime import evenly_spaced_indices
from scripts.select_moshi_v2_checkpoint import select_checkpoint


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_final_runtime_panel_is_deterministic_and_spans_manifest() -> None:
    assert evenly_spaced_indices(738) == (0, 92, 184, 276, 368, 460, 552, 644, 737)


def test_runtime_ineligible_lower_loss_cannot_win(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"max_steps": 200, "ckpt_freq": 100}),
        encoding="utf-8",
    )
    candidates = []
    runtime_candidates = []
    for step, loss, runtime_passes in ((100, 2.0, True), (200, 1.0, False)):
        artifact_dir = tmp_path / "checkpoints" / f"checkpoint_{step:06d}"
        artifact_dir.mkdir(parents=True)
        adapter = artifact_dir / "lora.safetensors"
        adapter.write_bytes(f"adapter-{step}".encode())
        adapter_config = artifact_dir / "config.json"
        adapter_config.write_text("{}", encoding="utf-8")
        candidates.append(
            {
                "step": step,
                "eval_loss": loss,
                "text_eval_loss": loss / 2,
                "audio_eval_loss": loss / 2,
                "sample_count": 7,
                "validation_scope": "complete_fixed_manifest",
                "validation_manifest_sha256": "validation-hash",
                "adapter_sha256": sha256(adapter),
                "config_sha256": sha256(adapter_config),
            }
        )
        runtime_candidates.append(
            {
                "step": step,
                "adapter_path": adapter.relative_to(tmp_path).as_posix(),
                "config_path": adapter_config.relative_to(tmp_path).as_posix(),
                "panel_count": 9,
                "panel_pass_count": 9 if runtime_passes else 8,
                "candidate_runtime_passes": runtime_passes,
                "static_adapter_validation": {
                    "adapter_sha256": sha256(adapter),
                    "config_sha256": sha256(adapter_config),
                    "passes": True,
                },
            }
        )

    reevaluation_path = tmp_path / "reevaluation.json"
    write_json(
        reevaluation_path,
        {
            "schema_version": 1,
            "status": "passed",
            "reevaluation_passes": True,
            "heldout_test_used": False,
            "configured_max_steps": 200,
            "checkpoint_frequency": 100,
            "candidates": candidates,
        },
    )
    runtime_path = tmp_path / "runtime.json"
    write_json(
        runtime_path,
        {
            "schema_version": 1,
            "status": "passed",
            "runtime_panel_evaluation_passes": True,
            "selection_performed": False,
            "artifacts": {"reevaluation_sha256": sha256(reevaluation_path)},
            "protocol": {"panel_indices": [0, 16, 32, 48, 64, 80, 96, 112, 130]},
            "candidates": runtime_candidates,
        },
    )

    report = select_checkpoint(
        config_path=config_path,
        reevaluation_path=reevaluation_path,
        runtime_path=runtime_path,
        out_path=tmp_path / "selection.json",
        root=tmp_path,
    )

    assert report["selection_passes"] is True
    assert report["eligible_steps"] == [100]
    assert report["selected"]["step"] == 100
    assert report["selected"]["eval_loss"] == 2.0
    assert report["candidates"][1]["autoregressive_eligible"] is False
