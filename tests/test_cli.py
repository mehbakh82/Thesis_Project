from __future__ import annotations

import json

import pytest

from thesis_s2s.cli import main


def test_cli_help_registers_critical_workflows(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "audit-conversations",
        "export-moshi-data",
        "audit-moshi-data",
        "gpu-preflight",
        "verify-upstreams",
        "release-snapshot",
    ):
        assert command in output


def test_verify_upstreams_cli_dispatches_read_only(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkouts = tmp_path / "checkouts"
    checkouts.mkdir()

    main(["verify-upstreams", "--checkouts-root", str(checkouts)])

    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert len(payload["checks"]) >= 2
    assert all(check["checkout_present"] is False for check in payload["checks"])
