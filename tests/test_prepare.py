import json
from pathlib import Path

from thesis_s2s.data.channels import YOUTUBE_CHANNELS, remote_csv
from thesis_s2s.data.filter_corpus import filter_hours, write_synthetic_duplex
from thesis_s2s.data.prepare_youtube import caption_ok, chunk_name, episode_split
from thesis_s2s.data.quality import conversational_ok, looks_persian
from thesis_s2s.data.verbatim import s2s_text, verbatim_normalize
from thesis_s2s.runtime.tts import formant_synthesize, g2p


def test_chunk_join_rule():
    assert chunk_name("episodeA", 1) == "episodeA_chunk_0001.wav"
    assert episode_split("x") in {"train", "val", "test"}


def test_caption_filter_and_verbatim():
    assert caption_ok("[موسیقی]") is None
    assert caption_ok("سلام خوبی") is not None
    out = verbatim_normalize("سلام  كجايي")
    assert "  " not in out
    assert s2s_text(
        {
            "transcript_caption": "از سی اس وی",
            "text": "بازنویسی شده",
            "transcript_nemo": "فرضیه نمو",
        }
    ) == "از سی اس وی"
    assert s2s_text({"text": "فقط متن", "transcript_nemo": "نمو"}) == "فقط متن"


def test_channels_have_csv_and_chunks():
    assert len(YOUTUBE_CHANNELS) >= 4
    assert "CSVs" in remote_csv(YOUTUBE_CHANNELS[0])


def test_quality_persian_and_duration():
    assert looks_persian("سلام حالت چطوره")
    ok, reason = conversational_ok(duration=4.0, text="سلام خوبی هستی", snr_db=12.0)
    assert ok and reason == "ok"
    bad, why = conversational_ok(duration=0.2, text="سلام خوبی هستی", snr_db=12.0)
    assert not bad and why == "duration"


def test_filter_and_synthetic(tmp_path: Path):
    jsonl = tmp_path / "in.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "duration": 3.0,
                "text": "این یک جمله فارسی آزمایشی است",
                "transcript_caption": "این یک جمله فارسی آزمایشی است",
                "transcript_nemo": "این متن نباید برای S2S استفاده شود",
                "snr": 15.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    report = filter_hours(jsonl, out, min_hours=0.0, max_hours=1.0, compute_snr=False)
    assert report["n"] == 1
    kept = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert kept["text"] == "این یک جمله فارسی آزمایشی است"
    assert "نباید" not in kept["text"]
    synth = write_synthetic_duplex(tmp_path / "syn", tmp_path / "syn.jsonl", n_per_class=1, clip_seconds=1.2)
    assert synth["n"] == 4


def test_formant_tts_phones():
    phones = g2p("سلام")
    assert phones
    audio = formant_synthesize("سلام")
    assert len(audio) > 800
