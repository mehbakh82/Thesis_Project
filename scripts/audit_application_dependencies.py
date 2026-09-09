#!/usr/bin/env python3
"""Audit general application dependencies against the reviewed risk policy."""

from pathlib import Path

from audit_moshi_dependencies import main

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_policy=ROOT / "configs/application_dependency_risk_policy.json",
            default_lock=ROOT / "requirements.txt",
        )
    )
