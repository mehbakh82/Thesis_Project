from __future__ import annotations

from pathlib import Path

from scripts.run_moshi_v5_posttraining import validation_commands


def test_v5_pipeline_stops_after_selection_and_uses_frozen_protocol() -> None:
    stages = validation_commands()
    names = [stage["name"] for stage in stages]
    assert names == [
        "complete_validation_reevaluation",
        "predeclared_complete_prompt_runtime_panel",
        "eligible_only_selection",
    ]
    assert all("final" not in name for name in names)
    selection = stages[-1]
    assert Path(selection["command"][0]).parent.parent.name == ".venv"
    assert selection["command"][-2:] == ["--protocol-mode", "predeclared-v5"]
    assert "evaluate_moshi_v5_runtime_candidates.py" in str(stages[1]["command"][-1])
    assert "moshi_fa_100s_v5_codebook10" in str(stages[0]["command"])
