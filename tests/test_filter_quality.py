import json

import numpy as np
import pytest

from thesis_s2s.data import filter_corpus
from thesis_s2s.data.filter_corpus import filter_hours
from thesis_s2s.data.quality import conversational_ok, estimate_snr_db

PERSIAN = "این یک جمله فارسی مناسب است"


def _write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_quality_proxies_reject_nonfinite_values():
    assert estimate_snr_db(np.zeros(10, dtype=np.float32)) == 0.0
    assert conversational_ok(duration=float("nan"), text=PERSIAN, snr_db=10)[1] == (
        "invalid_duration"
    )
    assert conversational_ok(duration=3, text=PERSIAN, snr_db=float("nan"))[1] == (
        "invalid_snr"
    )


def test_filter_packs_rows_without_overshooting_hour_cap(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "filtered.jsonl"
    _write_rows(
        source,
        [
            {"duration": 4, "text": PERSIAN},
            {"duration": 3, "text": PERSIAN},
            {"duration": 1, "text": PERSIAN},
        ],
    )

    report = filter_hours(
        source, output, min_hours=0, max_hours=5 / 3600, compute_snr=False
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert [row["duration"] for row in rows] == [4, 1]
    assert report["max_hours_ok"] is True
    assert report["hours"] <= round(5 / 3600, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_hours": -1},
        {"min_hours": 2, "max_hours": 1},
        {"max_hours": float("inf")},
        {"min_snr_db": float("nan")},
    ],
)
def test_filter_rejects_invalid_thresholds(tmp_path, kwargs):
    source = tmp_path / "source.jsonl"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        filter_hours(source, tmp_path / "out.jsonl", **kwargs)


def test_filter_rejects_malformed_rows_before_replacing_output(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "filtered.jsonl"
    source.write_text('{"duration": 3, "text": "ok"}\n[]\n', encoding="utf-8")
    output.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        filter_hours(source, output)

    assert output.read_text(encoding="utf-8") == "preserve"


def test_atomic_manifest_write_preserves_previous_file_on_serialization_failure(
    tmp_path, monkeypatch
):
    output = tmp_path / "filtered.jsonl"
    output.write_text("preserve", encoding="utf-8")
    real_dumps = filter_corpus.json.dumps

    def fail_dumps(value, **kwargs):
        if value.get("fail"):
            raise TypeError("cannot serialize")
        return real_dumps(value, **kwargs)

    monkeypatch.setattr(filter_corpus.json, "dumps", fail_dumps)

    with pytest.raises(TypeError, match="cannot serialize"):
        filter_corpus._write_jsonl_atomic(output, [{"ok": True}, {"fail": True}])

    assert output.read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob("*.tmp"))
