"""YouTube channel prefixes and conversation-selection priors on the 2TB bucket."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    csv_prefix: str
    chunks_prefix: str
    bucket: str = "asr"
    skip_shorts: bool = True
    source_kind: str = "mixed"
    conversation_priority: int = 2
    expected_multi_speaker: bool | None = None
    selection_weight: float = 0.0


# All sources are on the 2TB MinIO bucket. Do not embed credentials here.

TABAGHE16 = ChannelSpec(
    name="Tabaghe16",
    csv_prefix="STT/YT_PodCast_Chunks/CSVs/طبقه 16",
    chunks_prefix="STT/YT_PodCast_Chunks/Audio_Chunks/طبقه 16",
    source_kind="interview_podcast",
    conversation_priority=1,
    expected_multi_speaker=True,
    selection_weight=0.50,
)

# Weights are priors, not labels; diarization confirms speakers per episode.
YOUTUBE_CHANNELS: tuple[ChannelSpec, ...] = (
    TABAGHE16,
    ChannelSpec(
        "Mehran Rowshan Persian",
        "Mehran Rowshan Persian/CSVs",
        "Mehran Rowshan Persian/Audio_Chunks",
        source_kind="interview_podcast",
        conversation_priority=1,
        expected_multi_speaker=True,
        selection_weight=0.28,
    ),
    ChannelSpec(
        "Digiato",
        "Digiato/CSVs",
        "Digiato/Audio_Chunks",
        source_kind="technology_mixed",
        conversation_priority=2,
        expected_multi_speaker=None,
        selection_weight=0.10,
    ),
    ChannelSpec(
        "Zoomit",
        "Zoomit/CSVs",
        "Zoomit/Audio_Chunks",
        source_kind="technology_mixed",
        conversation_priority=2,
        expected_multi_speaker=None,
        selection_weight=0.10,
    ),
    ChannelSpec(
        "Kooshiar",
        "Kooshiar/CSVs",
        "Kooshiar/Audio_Chunks",
        source_kind="primarily_monologue",
        conversation_priority=3,
        expected_multi_speaker=False,
        selection_weight=0.02,
    ),
)


def remote_csv(spec: ChannelSpec) -> str:
    return f":s3:{spec.bucket}/{spec.csv_prefix}"


def remote_chunks(spec: ChannelSpec) -> str:
    return f":s3:{spec.bucket}/{spec.chunks_prefix}"
