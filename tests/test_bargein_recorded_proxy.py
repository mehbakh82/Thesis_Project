from __future__ import annotations

from thesis_s2s.audio import write_wav
from thesis_s2s.bargein.features import FeatureConfig
from thesis_s2s.bargein.recorded_proxy import (
    RecordedEvent,
    discover_events,
    extract_feature_rows,
    select_balanced_events,
    select_validation_threshold,
    stable_split,
    validate_event_split,
)


def _config() -> dict:
    return {
        "seed": 20260831,
        "split": {
            "train_end_exclusive": 70,
            "validation_end_exclusive": 85,
            "minimum_groups_per_split": 1,
            "minimum_events_per_class_per_split": 1,
        },
        "events": {
            "maximum_events_per_kind_per_session": 8,
            "raw_turn_match_ratio": 0.5,
            "boundary_tolerance_s": 0.5,
            "minimum_clean_gap_s": 0.2,
            "maximum_clean_gap_s": 2.0,
        },
    }


def _row(window: str, *, overlap: bool) -> dict:
    response_start = 0.7 if overlap else 1.3
    return {
        "window_id": window,
        "session_id": f"session-{window}",
        "episode_id": f"session-{window}",
        "channel": "Tabaghe16",
        "audio_filepath": "private.wav",
        "automatic_multi_speaker_verified": True,
        "reference_alignment_status": "complete",
        "internal_research_authorized": True,
        "segments": [
            {"start": 0.0, "end": 1.0, "speaker": "A", "text": "پرسش"},
            {"start": response_start, "end": 2.2, "speaker": "B", "text": "پاسخ"},
        ],
        "speaker_turns": [
            {"start": 0.0, "end": 1.0, "speaker": "A"},
            {"start": response_start, "end": 2.2, "speaker": "B"},
        ],
    }


def test_event_discovery_keeps_proxy_interrupt_and_clean_turn_separate() -> None:
    events = discover_events(
        [_row("overlap", overlap=True), _row("clean", overlap=False)], _config()
    )

    assert {event.kind for event in events} == {"interrupt", "clean_turn"}
    assert {event.label for event in events} == {0, 1}
    assert all(event.session_id for event in events)


def test_seeded_session_split_is_deterministic() -> None:
    first = stable_split("episode-1", seed=7, train_end=70, validation_end=85)
    second = stable_split("episode-1", seed=7, train_end=70, validation_end=85)

    assert first == second
    assert first in {"train", "validation", "test"}


def test_balancing_and_split_validation_are_group_disjoint() -> None:
    events = []
    for split in ("train", "validation", "test"):
        for label, kind in ((1, "interrupt"), (0, "clean_turn")):
            events.append(
                RecordedEvent(
                    event_id=f"{split}-{kind}",
                    session_id=f"{split}-session",
                    channel="channel",
                    audio_path="private.wav",
                    onset_s=1.0,
                    label=label,
                    kind=kind,
                    split=split,
                    source_window_id=f"{split}-window",
                )
            )
    selected = select_balanced_events(events, maximum_events_per_kind_per_session=8)

    assert len(selected) == 6
    assert all(validate_event_split(selected, _config()).values())


def test_threshold_selection_uses_frozen_tie_break() -> None:
    import numpy as np

    truth = np.asarray([0, 0, 1, 1], dtype=np.int32)
    probability = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    selected, rows = select_validation_threshold(truth, probability, [0.4, 0.5, 0.6])

    assert selected == 0.5
    assert len(rows) == 3


def test_feature_extraction_preserves_event_order_across_audio_groups(tmp_path) -> None:
    import numpy as np

    quiet = tmp_path / "a.wav"
    voiced = tmp_path / "z.wav"
    write_wav(quiet, np.zeros(16000, dtype=np.float32))
    write_wav(voiced, np.full(16000, 0.25, dtype=np.float32))
    events = [
        RecordedEvent(
            event_id="voiced-first",
            session_id="session-z",
            channel="channel",
            audio_path=str(voiced),
            onset_s=0.5,
            label=1,
            kind="interrupt",
            split="test",
            source_window_id="z",
        ),
        RecordedEvent(
            event_id="quiet-second",
            session_id="session-a",
            channel="channel",
            audio_path=str(quiet),
            onset_s=0.5,
            label=0,
            kind="clean_turn",
            split="train",
            source_window_id="a",
        ),
    ]

    vectors, labels = extract_feature_rows(
        events,
        root=tmp_path,
        feature_config=FeatureConfig(),
        pre_onset_s=0.4,
        post_onset_s=0.12,
    )

    assert labels.tolist() == [1, 0]
    assert vectors[0, 0] > vectors[1, 0]
