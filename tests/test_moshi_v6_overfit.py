from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.evaluate_moshi_v6_overfit import (
    MIN_NORMALIZED_TEXT_LENGTH,
    normalize_text,
    prefix_character_error_rate,
)
from scripts.moshi_train_entry import (
    PERSIAN_TEXT_PARAMETER_NAMES,
    _configure_audio_loss_weight,
    _configure_persian_text_adaptation,
)
from scripts.prepare_moshi_v6_overfit import (
    SELECTION_SIZE,
    persian_letter_fraction,
    select_rows,
    source_is_eligible,
)

ROOT = Path(__file__).resolve().parents[1]


class _Parameter:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class _PersianScopeModel:
    def __init__(self) -> None:
        self.parameters_by_name = {
            "transformer.0.lora_A.weight": _Parameter(True),
            "text_emb.weight": _Parameter(False),
            "depformer_text_emb.weight": _Parameter(False),
            "text_linear.frozen_W.weight": _Parameter(False),
            "emb.0.weight": _Parameter(False),
            "depformer_emb.0.weight": _Parameter(False),
        }

    def named_parameters(self):
        return list(self.parameters_by_name.items())


def test_persian_text_scope_enables_untied_output_head_only() -> None:
    model = _PersianScopeModel()
    module = SimpleNamespace(get_fsdp_model=lambda args, checkpoint_info: model)
    args = SimpleNamespace(
        full_finetuning=False,
        lora=SimpleNamespace(enable=True, ft_embed=False),
    )

    _configure_persian_text_adaptation(module)
    configured = module.get_fsdp_model(args, object())

    trainable_full = {
        name
        for name, parameter in configured.named_parameters()
        if parameter.requires_grad and "lora" not in name
    }
    assert trainable_full == PERSIAN_TEXT_PARAMETER_NAMES
    assert configured.parameters_by_name["emb.0.weight"].requires_grad is False
    assert configured.parameters_by_name["depformer_emb.0.weight"].requires_grad is False
    assert os.environ["MOSHI_PERSIAN_TEXT_ADAPTATION_EFFECTIVE"] == (
        "depformer_text_emb.weight,text_emb.weight,text_linear.frozen_W.weight"
    )


def test_persian_text_scope_rejects_broad_embedding_mode() -> None:
    module = SimpleNamespace(get_fsdp_model=lambda args, checkpoint_info: object())
    args = SimpleNamespace(
        full_finetuning=False,
        lora=SimpleNamespace(enable=True, ft_embed=True),
    )
    _configure_persian_text_adaptation(module)

    with pytest.raises(RuntimeError, match="ft_embed=false"):
        module.get_fsdp_model(args, object())


def test_audio_loss_weight_changes_only_audio_loss() -> None:
    module = SimpleNamespace(compute_loss_with_mask=lambda *args, **kwargs: torch.tensor(2.0))
    _configure_audio_loss_weight(module, 0.1)

    assert module.compute_loss_with_mask(None, None, None, "audio").item() == pytest.approx(0.2)
    assert module.compute_loss_with_mask(None, None, None, mode="text").item() == 2.0
    assert os.environ["MOSHI_AUDIO_LOSS_WEIGHT_EFFECTIVE"] == "0.1"


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.1, float("inf"), float("nan")])
def test_audio_loss_weight_rejects_unsafe_values(weight: float) -> None:
    module = SimpleNamespace(compute_loss_with_mask=lambda *args, **kwargs: torch.tensor(1.0))
    with pytest.raises(RuntimeError, match=r"finite and in \(0, 1\]"):
        _configure_audio_loss_weight(module, weight)


def _source(index: int) -> dict:
    return {
        "utt_id": f"pair-{index:03d}",
        "session_id": f"channel-{index % 9}/session-{index % 17}",
        "split": "train",
        "training_use_authorized": True,
        "noise_condition": "background-clean",
        "overlap_intervals": [],
        "interrupt_label": "none",
        "user_duration": 2.0,
        "assistant_duration": 2.0,
        "source_span_start": 0.0,
        "source_user_interval": [0.0, 2.0],
        "response_text": f"این پاسخ فارسی شماره {index} است",
        "snr": 30.0,
    }


def test_train_only_selection_is_deterministic_and_session_diverse(tmp_path: Path) -> None:
    sources = [_source(index) for index in range(35)]
    exports = []
    metadata = {}
    for index in range(35):
        wav = tmp_path / f"pair-{index:03d}.wav"
        wav.touch()
        exports.append({"path": str(wav), "duration": 5.0, "sha256": str(index)})
        metadata[wav.with_suffix(".json").resolve()] = {"source_pair_id": f"pair-{index:03d}"}

    def loader(path: Path) -> dict:
        return metadata[path.resolve()]

    first_exports, first_descriptors, eligible = select_rows(
        exports, sources, metadata_loader=loader
    )
    second_exports, second_descriptors, second_eligible = select_rows(
        exports, sources, metadata_loader=loader
    )

    assert eligible == second_eligible == 35
    assert len(first_exports) == len(first_descriptors) == SELECTION_SIZE
    assert [row["pair_id"] for row in first_descriptors] == [
        row["pair_id"] for row in second_descriptors
    ]
    assert first_exports == second_exports
    assert len({row["session_id"] for row in first_descriptors}) >= 8


def test_mechanical_filter_does_not_claim_human_cleanliness() -> None:
    source = _source(1)
    export = {"duration": 5.0}
    assert source_is_eligible(source, export) is True
    assert persian_letter_fraction(source["response_text"]) >= 0.95

    source["split"] = "validation"
    assert source_is_eligible(source, export) is False


def test_runtime_text_gate_has_meaningful_minimum_and_prefix_cer() -> None:
    generated = "این پاسخ"
    assert len(normalize_text(generated)) >= MIN_NORMALIZED_TEXT_LENGTH
    assert prefix_character_error_rate(generated, "این پاسخ فارسی است") == 0.0
    assert prefix_character_error_rate("متن غلط", "این پاسخ فارسی است") > 0.0


def test_frozen_v6_configuration_and_report_match() -> None:
    config = yaml.safe_load((ROOT / "configs/moshi_h100_v6_overfit.yaml").read_text())
    probe = yaml.safe_load((ROOT / "configs/moshi_h100_v6_overfit_probe.yaml").read_text())
    policy = json.loads((ROOT / "configs/moshi_v6_overfit_policy.json").read_text())
    report = json.loads((ROOT / "results/moshi_v6_overfit_data.json").read_text())

    assert config["data"]["train_data"] == config["data"]["eval_data"]
    assert config["duration_sec"] == probe["duration_sec"] == 12
    assert (
        config["lora"]
        == probe["lora"]
        == {
            "enable": True,
            "rank": 64,
            "scaling": 2.0,
            "ft_embed": False,
        }
    )
    assert config["max_steps"] == 200
    assert config["ckpt_freq"] == 50
    assert probe["max_steps"] == probe["ckpt_freq"] == 1
    assert policy["trainable_full_parameters"] == sorted(PERSIAN_TEXT_PARAMETER_NAMES)
    assert policy["expected_total_adapter_tensor_count"] == 677
    assert policy["audio_loss_weight"] == 0.1
    assert policy["estimated_adapter_bytes"] == 977_709_056
    assert policy["launcher_environment"]["MOSHI_DISTRIBUTED_BACKEND"] == "gloo"
    assert report["passes"] is True
    assert report["human_verified"] is False
    assert report["mechanically_filtered_not_human_clean"] is True
    assert report["final_test_accessed"] is False
    assert report["output"]["rows"] == SELECTION_SIZE
