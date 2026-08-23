"""YouTube channel prefixes on the 2TB `asr` bucket (and 1TB Tabaghe16)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    csv_prefix: str
    chunks_prefix: str
    bucket: str = "asr"
    skip_shorts: bool = True


# 2TB MinIO bucket `asr`. Do not embed credentials here.
YOUTUBE_CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("Digiato", "Digiato/CSVs", "Digiato/Audio_Chunks"),
    ChannelSpec("Zoomit", "Zoomit/CSVs", "Zoomit/Audio_Chunks"),
    ChannelSpec("Kooshiar", "Kooshiar/CSVs", "Kooshiar/Audio_Chunks"),
    ChannelSpec(
        "Mehran Rowshan Persian",
        "Mehran Rowshan Persian/CSVs",
        "Mehran Rowshan Persian/Audio_Chunks",
    ),
)

TABAGHE16 = ChannelSpec(
    name="Tabaghe16",
    csv_prefix="asr-gpu/Tabaghe16/Tabaghe16_CSVs",
    chunks_prefix="asr-gpu/Tabaghe16/Tabaghe16_Audio_Chunks",
    bucket="asr-gpu",
)


def remote_csv(spec: ChannelSpec) -> str:
    if spec.bucket == "asr-gpu":
        return f":s3:{spec.csv_prefix}"
    return f":s3:{spec.bucket}/{spec.csv_prefix}"


def remote_chunks(spec: ChannelSpec) -> str:
    if spec.bucket == "asr-gpu":
        return f":s3:{spec.chunks_prefix}"
    return f":s3:{spec.bucket}/{spec.chunks_prefix}"
