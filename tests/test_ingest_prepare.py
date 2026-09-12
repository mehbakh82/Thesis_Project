import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from thesis_s2s.data import ingest, prepare_youtube
from thesis_s2s.data.channels import ChannelSpec


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "start_time,end_time,text\n" + "".join(
        f'"{start}","{end}","{text}"\n' for start, end, text in rows
    )
    path.write_text(body, encoding="utf-8")


def _episode(csv_path: Path, *, split: str = "val", channel: str = "Channel") -> dict:
    return {
        "channel": channel,
        "stem": csv_path.stem,
        "csv_path": str(csv_path),
        "chunks_remote": ":s3:bucket/chunks",
        "split": split,
    }


def test_rclone_wrappers_resolve_paths_and_build_safe_commands(tmp_path: Path, monkeypatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=" one.csv\n\ntwo.csv\n")

    monkeypatch.setattr(ingest, "rclone_prefix", lambda: ["rclone", "--config", "safe.conf"])
    monkeypatch.setattr(ingest, "rclone_path", lambda value: value.replace(":s3:", "named:"))
    monkeypatch.setattr(ingest, "rclone_process_env", lambda: {"SAFE": "1"})
    monkeypatch.setattr(ingest, "rclone_timeout_seconds", lambda: 123.0)
    monkeypatch.setattr(ingest.subprocess, "run", fake_run)

    result = ingest.rclone_run(["lsf", ":s3:bucket/path"], check=False)
    assert result.stdout.startswith(" one.csv")
    assert calls[0] == (
        ["rclone", "--config", "safe.conf", "lsf", "named:bucket/path"],
        {
            "check": False,
            "capture_output": True,
            "text": True,
            "env": {"SAFE": "1"},
            "timeout": 123.0,
        },
    )

    run_args: list[list[str]] = []
    monkeypatch.setattr(
        ingest,
        "rclone_run",
        lambda args, check=True: (
            run_args.append(args)
            or subprocess.CompletedProcess(args, 0, stdout=" one.csv\n\ntwo.csv\n")
        ),
    )
    ingest.rclone_copy(":s3:remote", tmp_path / "dest", include="*.wav")
    assert (tmp_path / "dest").is_dir()
    assert run_args[0] == [
        "copy",
        ":s3:remote",
        str(tmp_path / "dest"),
        "--transfers",
        "8",
        "--include",
        "*.wav",
    ]
    assert ingest.rclone_lsf(":s3:remote") == ["one.csv", "two.csv"]


def test_clock_duration_timeline_and_candidate_policies():
    assert ingest._parse_clock("") == 0.0
    assert ingest._parse_clock("12.5") == 12.5
    assert ingest._parse_clock("01:02") == 62.0
    assert ingest._parse_clock("1:02:03,5") == 3723.5
    assert ingest._parse_clock("bad") == 0.0
    assert ingest._parse_clock("1:bad") == 0.0

    rows = [
        {"start_time": "00:00", "end_time": "00:10"},
        {"start_time": "00:20", "end_time": "00:50"},
        {"start_time": "10", "end_time": "5"},
    ]
    assert ingest._csv_duration_hours(rows) == pytest.approx(40 / 3600)
    assert ingest._timeline_hours(rows) == pytest.approx(50 / 3600)
    assert ingest._timeline_hours([{"start_time": "5", "end_time": "2"}]) == 0.0

    podcast = ChannelSpec("pod", "csv", "chunks", expected_multi_speaker=True)
    mixed = ChannelSpec("mixed", "csv", "chunks", expected_multi_speaker=None)
    mono = ChannelSpec("mono", "csv", "chunks", expected_multi_speaker=False)
    assert ingest._conversation_candidate(podcast, "anything")[0]
    assert ingest._conversation_candidate(mixed, "anything")[0]
    assert ingest._conversation_candidate(mono, "interview episode 7")[0]
    assert not ingest._conversation_candidate(mono, "solo lesson")[0]


def test_inventory_csvs_is_fail_closed_and_records_channel_qualified_split(
    tmp_path: Path, monkeypatch
):
    csv_root = tmp_path / "csv-root"
    specs = (
        ChannelSpec(
            "Podcast",
            "pod/csv",
            "pod/chunks",
            source_kind="interview_podcast",
            conversation_priority=1,
            expected_multi_speaker=True,
            selection_weight=0.8,
        ),
        ChannelSpec(
            "Mono",
            "mono/csv",
            "mono/chunks",
            source_kind="monologue",
            expected_multi_speaker=False,
            selection_weight=0.2,
        ),
        ChannelSpec("Broken", "bad/csv", "bad/chunks"),
    )
    _write_csv(
        csv_root / "Podcast" / "csvs" / "episode.csv",
        [("0", "10", "سلام دوست"), ("10", "25", "حال شما")],
    )
    _write_csv(
        csv_root / "Podcast" / "csvs" / "short.csv",
        [("0", "1", "خیلی کوتاه")],
    )
    _write_csv(
        csv_root / "Podcast" / "csvs" / "no_captions_found.csv",
        [("0", "1", "نادیده")],
    )
    _write_csv(
        csv_root / "Mono" / "csvs" / "solo lesson.csv",
        [("0", "5", "درس اول"), ("5", "9", "درس دوم")],
    )

    def fake_copy(remote: str, _dest: Path, include: str | None = None):
        assert include == "*.csv"
        if "bad/csv" in remote:
            raise subprocess.CalledProcessError(1, ["rclone"], stderr="denied")

    monkeypatch.setattr(ingest, "rclone_copy", fake_copy)
    result = ingest.inventory_csvs(csv_root, channels=specs, min_episode_rows=2)

    assert result["stats"]["episodes_ok"] == 2
    assert result["stats"]["shorts_skipped"] == 1
    assert result["stats"]["csv_hours"] == pytest.approx(29 / 3600, abs=0.001)
    assert result["stats"]["channels"]["Broken"]["error"] == "denied"
    records = {row["channel"]: row for row in result["episodes"]}
    assert records["Podcast"]["selection_eligible"] is True
    assert records["Podcast"]["timeline_hours"] == round(25 / 3600, 4)
    assert records["Podcast"]["split"] == prepare_youtube.episode_split("Podcast/episode")
    assert records["Mono"]["selection_eligible"] is False
    assert records["Mono"]["license_verified"] is False


def test_manifest_progress_handles_absent_blank_and_partial_rows(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    assert ingest._manifest_progress(manifest) == (set(), set(), 0.0, 0)
    manifest.write_text(
        "\n"
        + json.dumps(
            {
                "utt_id": "Channel_episode_0001",
                "duration": 2.5,
                "channel": "Channel",
                "source_csv": "/private/episode.csv",
            }
        )
        + "\n"
        + json.dumps({"duration": 0})
        + "\n",
        encoding="utf-8",
    )
    seen, stems, hours, written = ingest._manifest_progress(manifest)
    assert seen == {"Channel_episode_0001"}
    assert stems == {("Channel", "episode")}
    assert hours == pytest.approx(2.5 / 3600)
    assert written == 2


def test_ingest_resumes_partial_episode_and_preserves_inventory_split(
    tmp_path: Path, monkeypatch
):
    csv_path = tmp_path / "episode.csv"
    _write_csv(
        csv_path,
        [("0", "2", "سلام اول"), ("2", "4", "سلام دوم")],
    )
    manifest = tmp_path / "manifests" / "youtube_all.jsonl"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "utt_id": "Channel_episode_0001",
                "duration": 2.0,
                "channel": "Channel",
                "source_csv": str(csv_path),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    remote_tmp = tmp_path / "remote"
    monkeypatch.setattr(ingest.tempfile, "mkdtemp", lambda prefix: str(remote_tmp))

    def fake_copy(_remote: str, dest: Path, include: str | None = None):
        dest.mkdir(parents=True, exist_ok=True)
        for index in (1, 2):
            (dest / prepare_youtube.chunk_name("episode", index)).touch()

    monkeypatch.setattr(ingest, "rclone_copy", fake_copy)
    monkeypatch.setattr(
        ingest,
        "load_wav_mono16k",
        lambda _path: (np.full(32000, 0.1, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(ingest, "estimate_snr_db", lambda _audio: 20.0)
    monkeypatch.setattr(
        ingest,
        "write_wav",
        lambda path, _audio, _sr: (path.parent.mkdir(parents=True, exist_ok=True), path.touch()),
    )

    stats = ingest.ingest_episodes(
        [_episode(csv_path, split="test")],
        tmp_path / "audio",
        manifest,
        max_hours=1,
        min_episode_rows=1,
        resume=True,
    )

    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert stats["written"] == 2
    assert stats["resumed_hours"] == pytest.approx(round(2 / 3600, 3))
    assert records[-1]["utt_id"] == "Channel_episode_0002"
    assert records[-1]["split"] == "test"
    assert records[-1]["transcript_caption"] == "سلام دوم"
    assert not remote_tmp.exists()


def test_ingest_counts_filter_decode_missing_mp3_and_remote_failures(
    tmp_path: Path, monkeypatch
):
    csv_path = tmp_path / "mixed.csv"
    _write_csv(
        csv_path,
        [
            ("0", "2", "[موسیقی]"),
            ("2", "4", "فایل گمشده"),
            ("4", "6", "خطای رمزگشایی"),
            ("6", "6.5", "بازه کوتاه"),
        ],
    )
    mp3_csv = tmp_path / "mp3.csv"
    _write_csv(mp3_csv, [("0", "2", "فایل ام پی سه")])
    bad_csv = tmp_path / "bad.csv"
    _write_csv(bad_csv, [("0", "2", "نمونه خراب")])
    tmp_dirs: list[Path] = []

    def fake_mkdtemp(prefix: str) -> str:
        path = tmp_path / f"remote-{len(tmp_dirs)}"
        path.mkdir()
        tmp_dirs.append(path)
        return str(path)

    def fake_copy(remote: str, dest: Path, include: str | None = None):
        if "bad" in remote:
            raise subprocess.CalledProcessError(1, ["rclone"])
        if include and include.startswith("mp3_chunk_") and include.endswith("*.mp3"):
            (dest / "mp3_chunk_0001.mp3").touch()
        elif include and include.startswith("mixed_chunk_"):
            (dest / "mixed_chunk_0003.wav").touch()
            (dest / "mixed_chunk_0004.wav").touch()

    def fake_load(path: Path):
        if "0003" in path.name:
            raise ValueError("decode")
        samples = 8000 if "0004" in path.name else 32000
        return np.full(samples, 0.1, dtype=np.float32), 16000

    monkeypatch.setattr(ingest.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(ingest, "rclone_copy", fake_copy)
    monkeypatch.setattr(ingest, "load_wav_mono16k", fake_load)
    monkeypatch.setattr(ingest, "estimate_snr_db", lambda _audio: 10.0)
    monkeypatch.setattr(
        ingest,
        "write_wav",
        lambda path, _audio, _sr: (path.parent.mkdir(parents=True, exist_ok=True), path.touch()),
    )

    episodes = [
        {**_episode(bad_csv, channel="Bad"), "chunks_remote": "bad"},
        _episode(csv_path, split="train"),
        _episode(mp3_csv, split="test"),
    ]
    stats = ingest.ingest_episodes(
        episodes,
        tmp_path / "audio",
        tmp_path / "manifest.jsonl",
        min_episode_rows=1,
        max_hours=1,
    )
    assert stats["errors"] == 2
    assert stats["missing_wav"] == 1
    assert stats["skipped_caption"] == 2
    assert stats["written"] == 1
    assert stats["episodes"] == 2
    assert all(not path.exists() for path in tmp_dirs)

    calls = 0

    def should_not_copy(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(ingest, "rclone_copy", should_not_copy)
    capped = ingest.ingest_episodes(
        episodes,
        tmp_path / "unused",
        tmp_path / "empty.jsonl",
        max_hours=0,
        max_episodes=0,
    )
    assert capped["written"] == 0
    assert calls == 0


def test_run_ingest_writes_inventory_and_delegates(tmp_path: Path, monkeypatch):
    csv_root = tmp_path / "csv"
    out_dir = tmp_path / "audio"
    manifest_dir = tmp_path / "manifests"
    inventory = {"stats": {"csv_hours": 12.3}, "episodes": [{"stem": "ep"}]}
    observed: dict = {}
    monkeypatch.setattr(ingest, "inventory_csvs", lambda path: inventory if path == csv_root else None)
    monkeypatch.setattr(
        ingest,
        "ingest_episodes",
        lambda episodes, output, manifest, **kwargs: observed.update(
            episodes=episodes, output=output, manifest=manifest, kwargs=kwargs
        )
        or {"written": 7},
    )

    result = ingest.run_ingest(
        max_hours=2,
        max_episodes=3,
        csv_root=csv_root,
        out_dir=out_dir,
        manifest_dir=manifest_dir,
        resume=False,
    )
    saved = json.loads((manifest_dir / "csv_inventory.json").read_text(encoding="utf-8"))
    assert saved == {"stats": inventory["stats"], "n_episodes": 1, "episodes": inventory["episodes"]}
    assert observed["manifest"] == manifest_dir / "youtube_all.jsonl"
    assert observed["kwargs"] == {"max_hours": 2, "max_episodes": 3, "resume": False}
    assert result == {"csv_inventory": inventory["stats"], "audio": {"written": 7}}


def test_prepare_local_covers_filtering_nested_audio_and_manifests(
    tmp_path: Path, monkeypatch
):
    csv_dir = tmp_path / "csv"
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    _write_csv(csv_dir / "no_captions_found.csv", [("0", "2", "نادیده گرفته شود")])
    _write_csv(csv_dir / "short.csv", [("0", "2", "قسمت کوتاه")])
    _write_csv(
        csv_dir / "episode.csv",
        [
            ("0", "2", "متن نخست"),
            ("2", "3", "[موسیقی]"),
            ("3", "5", "فایل گمشده"),
            ("5", "7", "خطای خواندن"),
            ("7", "9", "متن دوم"),
            ("9", "9.5", "خیلی کوتاه"),
        ],
    )
    (wav_dir / "episode_chunk_0001.wav").touch()
    (wav_dir / "episode_chunk_0004.wav").touch()
    (wav_dir / "nested").mkdir()
    (wav_dir / "nested" / "episode_chunk_0005.wav").touch()
    (wav_dir / "episode_chunk_0006.wav").touch()

    def fake_load(path: Path):
        if "0004" in path.name:
            raise ValueError("decode")
        samples = 8000 if "0006" in path.name else 32000
        return np.full(samples, 0.1, dtype=np.float32), 16000

    monkeypatch.setattr(prepare_youtube, "episode_split", lambda _stem: "val")
    monkeypatch.setattr(prepare_youtube, "load_wav_mono16k", fake_load)
    monkeypatch.setattr(
        prepare_youtube,
        "write_wav",
        lambda path, _audio, _sr: (path.parent.mkdir(parents=True, exist_ok=True), path.touch()),
    )
    manifest_dir = tmp_path / "manifests"
    stats = prepare_youtube.prepare(
        csv_dir,
        wav_dir,
        tmp_path / "out",
        manifest_dir,
        min_episode_rows=2,
    )
    assert stats == {
        "written": 2,
        "skipped_short_episode": 1,
        "skipped_caption": 2,
        "missing_wav": 2,
        "hours": round(4 / 3600, 3),
    }
    rows = [
        json.loads(line)
        for line in (manifest_dir / "val_manifest_youtube.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["utt_id"] for row in rows] == ["episode_0001", "episode_0005"]
    assert all(row["split"] == "val" and row["license_verified"] is False for row in rows)
    assert not (manifest_dir / "train_manifest_youtube.jsonl").read_text(encoding="utf-8")
    assert json.loads((manifest_dir / "prepare_stats.json").read_text()) == stats


def test_prepare_remote_cleanup_runs_on_success_and_failure(tmp_path: Path, monkeypatch):
    csv_dir = tmp_path / "csv"
    _write_csv(csv_dir / "episode.csv", [("0", "2", "سلام دنیا")])
    created: list[Path] = []

    def fake_mkdtemp(prefix: str) -> str:
        path = tmp_path / f"remote-{len(created)}"
        path.mkdir()
        created.append(path)
        return str(path)

    monkeypatch.setattr(prepare_youtube.tempfile, "mkdtemp", fake_mkdtemp)
    with pytest.raises(ValueError, match="need --wav-dir"):
        prepare_youtube.prepare(
            csv_dir,
            None,
            tmp_path / "out-none",
            tmp_path / "manifest-none",
            min_episode_rows=1,
        )
    assert created == []

    monkeypatch.setattr(
        prepare_youtube,
        "rclone_copy_episode",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("network")),
    )
    with pytest.raises(RuntimeError, match="network"):
        prepare_youtube.prepare(
            csv_dir,
            None,
            tmp_path / "out-fail",
            tmp_path / "manifest-fail",
            remote_chunks=":s3:chunks",
            min_episode_rows=1,
        )
    assert created and not created[-1].exists()

    def fake_copy(_remote: str, stem: str, dest: Path):
        (dest / prepare_youtube.chunk_name(stem, 1)).touch()

    monkeypatch.setattr(prepare_youtube, "rclone_copy_episode", fake_copy)
    monkeypatch.setattr(
        prepare_youtube,
        "load_wav_mono16k",
        lambda _path: (np.full(32000, 0.1, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(
        prepare_youtube,
        "write_wav",
        lambda path, _audio, _sr: (path.parent.mkdir(parents=True, exist_ok=True), path.touch()),
    )
    stats = prepare_youtube.prepare(
        csv_dir,
        None,
        tmp_path / "out-ok",
        tmp_path / "manifest-ok",
        remote_chunks=":s3:chunks",
        min_episode_rows=1,
    )
    assert stats["written"] == 1
    assert not created[-1].exists()


def test_prepare_rclone_loader_and_cli(tmp_path: Path, monkeypatch, capsys):
    csv_path = tmp_path / "rows.csv"
    _write_csv(csv_path, [("0", "2", "سلام دنیا")])
    assert prepare_youtube.iter_csv_rows(csv_path)[0]["text"] == "سلام دنیا"

    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(prepare_youtube, "rclone_prefix", None, raising=False)
    monkeypatch.setattr("thesis_s2s.data.s3_inventory.rclone_prefix", lambda: ["rclone"])
    monkeypatch.setattr("thesis_s2s.data.s3_inventory.rclone_process_env", lambda: {"SAFE": "1"})
    monkeypatch.setattr("thesis_s2s.data.s3_inventory.rclone_timeout_seconds", lambda: 123.0)
    monkeypatch.setattr(
        prepare_youtube.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs))
        or subprocess.CompletedProcess(cmd, 0),
    )
    prepare_youtube.rclone_copy_episode(":s3:chunks", "episode", tmp_path / "download")
    assert calls[0][0][-4:] == ["--include", "episode_chunk_*.wav", "--transfers", "8"]
    assert calls[0][1] == {"check": True, "env": {"SAFE": "1"}, "timeout": 123.0}

    monkeypatch.setattr(
        "thesis_s2s.audio.read_wav",
        lambda path, sample_rate: (np.zeros(10, dtype=np.float32), sample_rate),
    )
    audio, sample_rate = prepare_youtube.load_wav_mono16k(tmp_path / "anything.wav")
    assert len(audio) == 10 and sample_rate == 16000

    captured: dict = {}
    monkeypatch.setattr(
        prepare_youtube,
        "prepare",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs) or {"written": 3},
    )
    prepare_youtube.main(
        [
            "--csv-dir",
            str(tmp_path / "csv"),
            "--wav-dir",
            str(tmp_path / "wav"),
            "--out-dir",
            str(tmp_path / "out"),
            "--manifest-dir",
            str(tmp_path / "manifest"),
            "--min-episode-rows",
            "4",
        ]
    )
    assert captured["kwargs"]["min_episode_rows"] == 4
    assert json.loads(capsys.readouterr().out) == {"written": 3}
