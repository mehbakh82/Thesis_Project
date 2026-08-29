from __future__ import annotations

from scripts.run_moshi_v3_final import final_commands


def test_v3_final_pipeline_orders_adapter_validation_before_test_access() -> None:
    stages = final_commands()
    assert [stage["name"] for stage in stages] == [
        "selected_adapter_validation",
        "one_time_objective_final_test",
        "separate_complete_prompt_final_runtime_diagnostics",
    ]
    assert [stage["test_access"] for stage in stages] == [False, True, True]
    assert stages[1]["command"][-2:] == ["--experiment-label", "v3"]
    assert stages[2]["command"][-2:] == ["--experiment-label", "v3"]
