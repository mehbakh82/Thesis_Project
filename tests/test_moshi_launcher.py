from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.moshi_train_entry import _configure_low_peak_adamw

ROOT = Path(__file__).resolve().parents[1]


class _FakeAdamW:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


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


def test_exact_profile_is_bound_to_fused_launcher_and_failed_attempts() -> None:
    launcher = ROOT / "scripts" / "moshi_train_entry.py"
    report = json.loads(
        (ROOT / "results" / "hardware" / "moshi_h100_profile_probe.json").read_text()
    )

    assert report["status"] == "passed"
    assert report["full_profile_gate_passes"] is True
    assert report["scientific_evidence"] is False
    assert report["optimizer_runtime"] == {
        "algorithm": "AdamW",
        "foreach": False,
        "fused": True,
    }
    assert (
        report["artifacts"]["project_launcher_sha256"]
        == hashlib.sha256(launcher.read_bytes()).hexdigest()
    )
