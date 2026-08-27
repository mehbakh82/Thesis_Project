from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.moshi_server_entry import configure_bound_host, requested_host
from scripts.prepare_moshi_v2_splits import heldout_session, split_rows
from scripts.validate_moshi_server_runtime import (
    decode_server_token_pieces,
    script_statistics,
)


def test_requested_host_defaults_to_loopback_and_parses_both_forms() -> None:
    assert requested_host([]) == "127.0.0.1"
    assert requested_host(["--host", "localhost"]) == "localhost"
    assert requested_host(["--host=::1"]) == "::1"


def test_requested_host_rejects_missing_or_empty_value() -> None:
    with pytest.raises(ValueError, match="requires a value"):
        requested_host(["--host"])
    with pytest.raises(ValueError, match="nonempty"):
        requested_host(["--host="])


def test_configured_server_honors_explicit_bind_host(monkeypatch) -> None:
    calls = []

    def run_app(app, *args, **kwargs):
        calls.append((app, args, kwargs))
        return "served"

    web = SimpleNamespace(run_app=run_app)
    configure_bound_host(web, "127.0.0.1")

    assert web.run_app("application", port=8998) == "served"
    assert calls == [("application", (), {"port": 8998, "host": "127.0.0.1"})]

    with pytest.raises(RuntimeError, match="conflicting"):
        web.run_app("application", host="0.0.0.0", port=8998)


def test_server_token_piece_decoder_recovers_persian_byte_fallback() -> None:
    decoded = decode_server_token_pieces(" <0xD8><0xB3>لام")

    assert decoded == " سلام"
    assert script_statistics(decoded) == {
        "persian_letters": 4,
        "latin_letters": 0,
        "persian_letter_fraction": 1.0,
    }
    assert script_statistics(" an English response")["persian_letter_fraction"] == 0.0


def test_v2_holdout_split_is_deterministic_and_group_disjoint() -> None:
    rows = [
        {"row": row, "session": f"session-{session}"} for session in range(40) for row in range(2)
    ]
    train, heldout, sessions = split_rows(rows, lambda row: row["session"])

    assert train and heldout
    assert sessions["train"].isdisjoint(sessions["heldout"])
    assert all(heldout_session(row["session"]) == (row in heldout) for row in rows)
