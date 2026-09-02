from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.moshi_text_input_dropout import configure_scheduled_text_input_dropout


class _FakeModel:
    text_padding_token_id = 3
    num_codebooks = 3

    def __init__(self) -> None:
        self.training = True
        self.received: list[torch.Tensor] = []

    def forward(self, codes: torch.Tensor, condition_tensors=None) -> torch.Tensor:
        self.received.append(codes.clone())
        return codes


def _configured_model(audit_path: Path | None = None) -> _FakeModel:
    model = _FakeModel()
    module = SimpleNamespace(get_fsdp_model=lambda args, checkpoint_info: model)
    configure_scheduled_text_input_dropout(
        module,
        torch,
        start_probability=0.5,
        end_probability=0.75,
        train_forwards=2,
        seed=20260902,
        audit_path=audit_path,
    )
    return module.get_fsdp_model(object(), object())


def _codes() -> torch.Tensor:
    text = torch.tensor([[0, 1, 2, 3] + list(range(4, 100))], dtype=torch.long)
    audio = torch.full((1, 2, 100), 17, dtype=torch.long)
    return torch.cat([text[:, None], audio], dim=1)


def test_dropout_changes_only_lexical_text_inputs_and_preserves_targets(tmp_path: Path) -> None:
    audit_path = tmp_path / "dropout.jsonl"
    model = _configured_model(audit_path)
    targets = _codes()
    original = targets.clone()

    corrupted = model.forward(codes=targets)

    assert torch.equal(targets, original)
    assert torch.equal(corrupted[:, 1:], original[:, 1:])
    assert torch.equal(corrupted[:, 0, :4], original[:, 0, :4])
    eligible = original[:, 0] > 3
    assert torch.all((corrupted[:, 0] == original[:, 0]) | (corrupted[:, 0] == 3))
    assert torch.any((corrupted[:, 0] == 3) & eligible)
    assert model.moshi_text_input_dropout_state["train_forward_calls"] == 1
    assert model.moshi_text_input_dropout_state["probability"] == pytest.approx(0.5)

    rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["targets_mutated"] is False
    assert rows[0]["evaluation_corrupted"] is False
    assert rows[0]["preserved_token_ids_at_most"] == 3


def test_dropout_schedule_is_deterministic_and_eval_is_clean() -> None:
    first = _configured_model()
    second = _configured_model()
    codes = _codes()

    first_output = first.forward(codes=codes)
    second_output = second.forward(codes=codes)
    assert torch.equal(first_output, second_output)

    first.forward(codes=codes)
    assert first.moshi_text_input_dropout_state["probability"] == pytest.approx(0.75)
    first.training = False
    assert torch.equal(first.forward(codes=codes), codes)
    assert first.moshi_text_input_dropout_state["train_forward_calls"] == 2

    first.training = True
    with pytest.raises(RuntimeError, match="exceeded frozen train forwards"):
        first.forward(codes=codes)


@pytest.mark.parametrize(
    ("start", "end", "forwards", "seed"),
    [
        (-0.1, 0.5, 2, 1),
        (0.8, 0.5, 2, 1),
        (0.0, 1.0, 2, 1),
        (0.1, 0.5, 0, 1),
        (0.1, 0.5, 2, -1),
        (float("nan"), 0.5, 2, 1),
    ],
)
def test_dropout_rejects_unsafe_controls(
    start: float,
    end: float,
    forwards: int,
    seed: int,
) -> None:
    module = SimpleNamespace(get_fsdp_model=lambda args, checkpoint_info: _FakeModel())
    with pytest.raises(RuntimeError):
        configure_scheduled_text_input_dropout(
            module,
            torch,
            start_probability=start,
            end_probability=end,
            train_forwards=forwards,
            seed=seed,
        )
