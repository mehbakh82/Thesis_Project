from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_asr_candidates_v4.py"
SPEC = importlib.util.spec_from_file_location("evaluate_asr_candidates_v4", SCRIPT)
assert SPEC and SPEC.loader
v4 = importlib.util.module_from_spec(SPEC)
sys.modules["evaluate_asr_candidates_v4"] = v4
SPEC.loader.exec_module(v4)


def arm(cer: float, senior: float, latency: float, paired: tuple[float, float] = (0.0, 0.0)) -> dict:
    return {
        "missing_predictions": 0,
        "micro_cer": cer,
        "latency_ms": {"short_clip_median": latency},
        "strata": {"senior": {"micro_cer": senior}},
        "paired_vs_incumbent": {"cer": {"point": sum(paired) / 2, "ci95_low": paired[0], "ci95_high": paired[1]}},
    }


def test_slow_arms_are_not_eligible_and_reruns_are_ignored() -> None:
    psrb = {
        "nemo_v4_ctc": arm(0.094, 0.156, 55),
        "nemo_v4_ctc_rerun_v4": arm(0.094, 0.156, 60),
        "nemotron_fa_la1120": arm(0.050, 0.10, 1200, (-0.05, -0.03)),
    }
    verdict = v4.replacement(psrb, {})
    assert verdict["latency_eligible_arms"] == ["nemo_v4_ctc"]
    assert verdict["replaces_nemo_v4"] is False


def test_replacement_needs_psrb_gain_seniors_and_no_fleurs_regression() -> None:
    psrb = {"nemo_v4_ctc": arm(0.094, 0.156, 55), "x": arm(0.080, 0.150, 60, (-0.02, -0.005))}
    better = {"x": arm(0.05, 0, 0, (-0.01, 0.002))}
    worse = {"x": arm(0.09, 0, 0, (0.001, 0.02))}
    assert v4.replacement(psrb, better)["replaces_nemo_v4"] is True
    assert v4.replacement(psrb, worse)["replaces_nemo_v4"] is False  # significant FLEURS regression
    assert v4.replacement(psrb, {})["replaces_nemo_v4"] is False  # no FLEURS evidence
    psrb["x"] = arm(0.080, 0.160, 60, (-0.02, -0.005))
    assert v4.replacement(psrb, better)["replaces_nemo_v4"] is False  # worse for seniors
    psrb["x"] = arm(0.080, 0.150, 60, (-0.02, 0.001))
    assert v4.replacement(psrb, better)["replaces_nemo_v4"] is False  # PSRB gain not significant
