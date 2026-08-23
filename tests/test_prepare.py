import json
from pathlib import Path

from thesis_s2s.data.channels import YOUTUBE_CHANNELS, remote_csv
from thesis_s2s.data.factory import refresh_dataset_card
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
    assert (
        s2s_text(
            {
                "transcript_caption": "از سی اس وی",
                "text": "بازنویسی شده",
                "transcript_nemo": "فرضیه نمو",
            }
        )
        == "از سی اس وی"
    )
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
    synth = write_synthetic_duplex(
        tmp_path / "syn", tmp_path / "syn.jsonl", n_per_class=1, clip_seconds=1.2
    )
    assert synth["n"] == 4


def test_formant_tts_phones():
    phones = g2p("سلام")
    assert phones
    audio = formant_synthesize("سلام")
    assert len(audio) > 800


def test_dataset_card_refresh_preserves_final_authorized_conversation_evidence(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "docs").mkdir()
    results = tmp_path / "results"
    results.mkdir()
    reports = {
        "conversation_selection_audit.json": {
            "inventory_stats": {"csv_hours": 775.887},
            "selected_candidate_hours": 180.043,
            "selected_episodes": 287,
            "planning_gate_passes": True,
        },
        "prepared_episode_audit.json": {
            "prepared_episodes": 287,
            "selected_episodes": 287,
            "windows": 947,
            "prepared_audio_hours": 201.576,
            "reconstruction_gate_passes": True,
        },
        "prepared_reserve_tabaghe16_audit.json": {
            "prepared_episodes": 22,
            "windows": 182,
            "prepared_audio_hours": 43.156,
        },
        "conversation_reserve_yield_selection.json": {
            "combined_candidate_hours": 196.546,
            "selected_reserve_episodes": 9,
        },
        "diarized_episode_audit_combined_authorized.json": {
            "episodes": 296,
            "windows": 1021,
            "prepared_hours": 219.403,
            "automatic_multi_speaker_hours": 181.824,
            "reference_aligned_hours": 217.115,
            "requirements": {"manual_qa_sample_present": False},
        },
        "conversation_yield_estimate_combined.json": {
            "estimated_pairs": 6017,
            "estimated_pair_hours": 105.727,
            "label_counts": {"none": 1230, "overlap_unattributed": 4787},
            "direct_interruption_pairs": 0,
        },
        "interaction_candidate_report.json": {
            "automatic_candidates": 712,
            "automatic_candidate_counts": {"interrupt": 669, "backchannel": 43},
            "sampled_candidates": 24,
        },
        "conversation_source_authorization_report_combined.json": {
            "counts": {"authorized_windows": 1021},
            "training_authorization_gate_passes": True,
        },
    }
    for name, report in reports.items():
        (results / name).write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr("thesis_s2s.data.factory.project_root", lambda: tmp_path)
    monkeypatch.setattr(
        "thesis_s2s.data.factory.SessionStore.export_manifest",
        lambda _self: {"hours": 0.0, "n": 0, "elderly_turns": 0},
    )

    card_path = refresh_dataset_card()
    card = card_path.read_text(encoding="utf-8")
    snapshot = json.loads((results / "dataset_card_snapshot.json").read_text())

    assert "Supervisor-approved internal thesis use" in card
    assert "196.546 candidate h / 296 episodes / 1021 windows" in card
    assert "105.727 estimated response-pair h / 6017 pairs" in card
    assert "overlap_unattributed" in card
    assert "712 conservative interaction candidates" in card
    assert "automatic candidates—not interruption claims" in card
    assert "pending explicit confirmation" not in card
    assert snapshot["conversation_training_authorization_gate"] is True
    assert snapshot["conversation_training_authorized_windows"] == 1021
    assert snapshot["conversation_direct_interruption_pairs"] == 0
    assert snapshot["conversation_automatic_interaction_candidates"] == 712
    assert snapshot["conversation_interaction_qa_sampled_candidates"] == 24
