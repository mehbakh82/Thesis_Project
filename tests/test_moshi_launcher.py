from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.moshi_train_entry import (
    _configure_low_peak_adamw,
    _configure_low_peak_checkpoints,
    _configure_repeatable_eval_loader,
)

ROOT = Path(__file__).resolve().parents[1]


class _FakeAdamW:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeCheckpointer:
    def retrieve_save_states(self, save_only_lora, save_dtype):
        return {"fallback": (save_only_lora, save_dtype)}


class _FakeParameter:
    requires_grad = True


class _FakeTrainableModule:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def parameters(self):
        return [_FakeParameter()]

    def state_dict(self):
        return {"weight": self.tensor}


class _FakeModel:
    def __init__(self, module: _FakeTrainableModule) -> None:
        self.module = module

    def modules(self):
        return []

    def named_modules(self):
        return [("text_emb", self.module)]


def test_low_peak_adamw_uses_fused_cuda_update() -> None:
    fake_torch = SimpleNamespace(optim=SimpleNamespace(AdamW=_FakeAdamW))

    _configure_low_peak_adamw(fake_torch)
    optimizer = fake_torch.optim.AdamW(["parameter"])

    assert optimizer.args == (["parameter"],)
    assert optimizer.kwargs["foreach"] is False
    assert optimizer.kwargs["fused"] is True


def test_low_peak_adamw_rejects_conflicting_request() -> None:
    fake_torch = SimpleNamespace(optim=SimpleNamespace(AdamW=_FakeAdamW))

    _configure_low_peak_adamw(fake_torch)

    with pytest.raises(RuntimeError, match="shared-H100 launcher requires AdamW foreach=False"):
        fake_torch.optim.AdamW([], foreach=True)

    with pytest.raises(RuntimeError, match="shared-H100 launcher requires AdamW fused=True"):
        fake_torch.optim.AdamW([], fused=False)


def test_low_peak_checkpoint_copies_adapter_state_to_cpu() -> None:
    checkpointing = SimpleNamespace(Checkpointer=_FakeCheckpointer, LoRALinear=type(None))
    distributed = SimpleNamespace(get_world_size=lambda: 1)
    source = torch.tensor([1.0], dtype=torch.float32)
    owner = _FakeCheckpointer()
    owner.full_finetuning = False
    owner.model = _FakeModel(_FakeTrainableModule(source))

    _configure_low_peak_checkpoints(checkpointing, distributed, torch)
    states = owner.retrieve_save_states(True, torch.float16)

    assert list(states) == ["text_emb.weight"]
    assert states["text_emb.weight"].device.type == "cpu"
    assert states["text_emb.weight"].dtype == torch.float16
    assert states["text_emb.weight"].data_ptr() != source.data_ptr()
    assert os.environ["MOSHI_CHECKPOINT_CPU_OFFLOAD_EFFECTIVE"] == "true"


def test_low_peak_checkpoint_preserves_upstream_fallbacks() -> None:
    checkpointing = SimpleNamespace(Checkpointer=_FakeCheckpointer, LoRALinear=type(None))
    distributed = SimpleNamespace(get_world_size=lambda: 1)
    owner = _FakeCheckpointer()

    _configure_low_peak_checkpoints(checkpointing, distributed, torch)

    assert owner.retrieve_save_states(False, torch.float16) == {"fallback": (False, torch.float16)}


def test_evaluation_loader_is_recreated_for_every_evaluation() -> None:
    calls = []

    def build_data_loader(*, is_eval):
        calls.append(is_eval)
        return iter([1, 2, 3])

    module = SimpleNamespace(build_data_loader=build_data_loader)
    _configure_repeatable_eval_loader(module)

    evaluation_loader = module.build_data_loader(is_eval=True)
    assert list(evaluation_loader) == [1, 2, 3]
    assert list(evaluation_loader) == [1, 2, 3]
    assert calls == [True, True]
    assert os.environ["MOSHI_REPEATABLE_EVAL_LOADER_EFFECTIVE"] == "true"

    training_loader = module.build_data_loader(is_eval=False)
    assert list(training_loader) == [1, 2, 3]
    assert calls == [True, True, False]


def test_exact_profile_is_bound_to_checkpoint_safe_launcher() -> None:
    launcher = ROOT / "scripts" / "moshi_train_entry.py"
    report = json.loads(
        (ROOT / "results" / "hardware" / "moshi_h100_profile_probe.json").read_text()
    )

    assert report["status"] == "passed"
    assert report["full_profile_gate_passes"] is True
    assert report["scientific_evidence"] is False
    assert report["requirements"]["checkpoint_save_completed"] is True
    assert report["requirements"]["project_launcher_offloads_single_gpu_adapter_save"] is True
    assert report["checkpoint_runtime"]["adapter_copy_device"] == "cpu"
    assert report["checkpoint_runtime"]["adapter_tensor_count"] == 699
    assert report["optimizer_runtime"] == {
        "algorithm": "AdamW",
        "foreach": False,
        "fused": True,
    }
    assert (
        report["artifacts"]["project_launcher_sha256"]
        == hashlib.sha256(launcher.read_bytes()).hexdigest()
    )
