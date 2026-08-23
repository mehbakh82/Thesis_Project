import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.audio import write_wav
from thesis_s2s.data.manual_qa import sample_manual_qa
from thesis_s2s.data.noise import annotate_noise_conditions, estimate_noise_condition


def _window(background_level: float) -> np.ndarray:
    rng = np.random.default_rng(4)
    audio = rng.normal(0.0, background_level, SAMPLE_RATE * 4).astype(np.float32)
    time = np.arange(SAMPLE_RATE * 2, dtype=np.float32) / SAMPLE_RATE
    audio[SAMPLE_RATE : SAMPLE_RATE * 3] += 0.2 * np.sin(2 * np.pi * 190 * time)
    return audio


def test_noise_estimate_distinguishes_clean_and_noisy_background():
    turns = [{"start": 1.0, "end": 3.0, "speaker": "A"}]

    clean = estimate_noise_condition(_window(0.001), SAMPLE_RATE, turns)
    noisy = estimate_noise_condition(_window(0.05), SAMPLE_RATE, turns)

    assert clean["noise_condition"] == "background-clean"
    assert clean["snr"] > 30.0
    assert noisy["noise_condition"] == "background-noisy"
    assert noisy["snr"] <= 20.0
    assert clean["noise_evidence"]["method"] == "speech_nonspeech_rms_v1"


def test_noise_annotation_writes_new_manifest_and_report(tmp_path: Path):
    wav = tmp_path / "conversation.wav"
    write_wav(wav, _window(0.02))
    source = tmp_path / "diarized.jsonl"
    source.write_text(
        json.dumps(
            {
                "window_id": "window-1",
                "audio_filepath": str(wav),
                "diarization_status": "complete",
                "exclusive_speaker_turns": [{"start": 1.0, "end": 3.0, "speaker": "A"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "noise.jsonl"
    report_path = tmp_path / "report.json"

    report = annotate_noise_conditions(source, out, report_path=report_path)
    row = json.loads(out.read_text(encoding="utf-8"))

    assert report["estimated_windows"] == 1
    assert row["noise_condition"].startswith("background-")
    assert row["snr"] is not None
    assert report_path.is_file()
    assert source.read_text(encoding="utf-8").find("noise_condition") == -1


def test_manual_qa_balances_noise_without_multiplying_stratum_size(tmp_path: Path):
    source = tmp_path / "noise-labeled.jsonl"
    rows = []
    for index, condition in enumerate(
        ("background-clean", "background-moderate", "background-noisy")
    ):
        rows.append(
            {
                "window_id": f"window-{index}",
                "episode_id": f"episode-{index}",
                "channel": "Podcast",
                "diarization_status": "complete",
                "automatic_multi_speaker_verified": True,
                "noise_condition": condition,
                "snr": 35.0 - index * 10.0,
            }
        )
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = sample_manual_qa(source, tmp_path / "qa.csv", per_stratum=3)

    assert report["sampled_windows"] == 3
    assert report["strata"] == {"Podcast/automatic_pass": 3}
    assert set(report["sampled_noise_conditions"]) == {
        "background-clean",
        "background-moderate",
        "background-noisy",
    }


def test_noise_annotation_concurrent_writers_use_unique_temporary_files(tmp_path: Path):
    wav = tmp_path / "conversation.wav"
    write_wav(wav, _window(0.02))
    source = tmp_path / "diarized.jsonl"
    source.write_text(
        json.dumps(
            {
                "window_id": "window-1",
                "audio_filepath": str(wav),
                "diarization_status": "complete",
                "exclusive_speaker_turns": [{"start": 1.0, "end": 3.0, "speaker": "A"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "noise.jsonl"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(annotate_noise_conditions, source, out) for _ in range(2)]
        reports = [future.result() for future in futures]

    assert [report["estimated_windows"] for report in reports] == [1, 1]
    assert json.loads(out.read_text(encoding="utf-8"))["noise_condition"].startswith("background-")
    assert not list(tmp_path.glob(".noise.jsonl.*.partial"))
