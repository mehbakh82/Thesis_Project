from __future__ import annotations

import pytest

from scripts.reevaluate_moshi_checkpoints import simulate_upstream_eval_consumption


def test_upstream_consumption_predicts_observed_exhaustion() -> None:
    steps = list(range(250, 8001, 250))

    rows = simulate_upstream_eval_consumption(682, steps)

    assert rows[0] == {
        "step": 250,
        "processed": 40,
        "discarded_by_off_by_one_break": 1,
        "consumed": 41,
        "remaining_after": 641,
    }
    assert rows[15]["step"] == 4000
    assert rows[15]["processed"] == 40
    assert rows[16]["step"] == 4250
    assert rows[16]["processed"] == 26
    assert rows[16]["remaining_after"] == 0
    assert rows[17]["step"] == 4500
    assert rows[17]["processed"] == 0
    assert [row["step"] for row in rows if row["processed"] == 0][0] == 4500


@pytest.mark.parametrize(
    ("chunks", "steps", "limit"),
    [(0, [1], 40), (1, [], 40), (1, [1], 0)],
)
def test_upstream_consumption_requires_positive_inputs(chunks, steps, limit) -> None:
    with pytest.raises(ValueError, match="positive chunks"):
        simulate_upstream_eval_consumption(chunks, steps, limit=limit)
