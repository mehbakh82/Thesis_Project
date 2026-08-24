"""Batch re-ASR via SpeechService.prepare + transcribe_chunks, not HTTP per file.

Falls back to Whisper-small as a local teacher when Triton/SpeechService cannot
be imported. S2S text is always fa-verbatim-2, never ASR normalise().
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from thesis_s2s.audio import read_wav
from thesis_s2s.data.verbatim import verbatim_normalize
from thesis_s2s.metrics import write_json


def _load_speech_service():
    configured = os.environ.get("THESIS_NEMO_ASR_API_ROOT", "").strip()
    if not configured:
        return None
    root = Path(configured)
    if not root.exists():
        return None
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from asr_engine import transcribe_chunks
        from service import SpeechService
        from speech_pipeline import audio_slice

        return SpeechService, transcribe_chunks, audio_slice
    except Exception:
        return None


class WhisperTeacher:
    def __init__(self, model_name: str = "openai/whisper-small"):
        from transformers import pipeline

        self.model_name = model_name
        device = 0
        try:
            import torch

            if not torch.cuda.is_available():
                device = -1
        except Exception:
            device = -1
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
        )

    def transcribe(self, wav_path: Path) -> dict:
        out = self.pipe(
            str(wav_path),
            generate_kwargs={"language": "persian", "task": "transcribe"},
        )
        text = verbatim_normalize(str(out.get("text") or ""))
        has_fa = any("\u0600" <= ch <= "\u06ff" for ch in text)
        return {
            "transcript_nemo": text,
            "language_status": "ok" if has_fa else "language_uncertain",
            "alignment": [],
            "teacher": self.model_name,
        }


def reasr_one(service, transcribe_chunks, audio_slice, wav_path: Path) -> dict:
    audio, sr = read_wav(wav_path)
    prepared = service.prepare(audio, enhance=False)
    if prepared.language_status != "ok" or not prepared.chunks:
        return {
            "transcript_nemo": None,
            "language_status": prepared.language_status,
            "alignment": [],
            "teacher": "nemo-soroush",
        }
    chunks = [audio_slice(prepared.asr_audio, c.start, c.end, sr) for c in prepared.chunks]
    starts = [c.start for c in prepared.chunks]
    result = transcribe_chunks(
        chunks, starts, sample_rate=sr, enable_alignment=True, use_ngram=False
    )
    text = verbatim_normalize(str(result.get("persian") or ""))
    return {
        "transcript_nemo": text,
        "language_status": prepared.language_status,
        "alignment": result.get("persian_alignment"),
        "enhancement": prepared.enhancement,
        "teacher": "nemo-soroush",
    }


class FasterWhisperTeacher:
    """CTranslate2 Whisper-small. Same teacher family, much faster than transformers."""

    def __init__(self, model_size: str = "small"):
        from faster_whisper import WhisperModel

        device = "cuda"
        compute = "float16"
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
                compute = "int8"
        except Exception:
            device = "cpu"
            compute = "int8"
        self.model_name = f"faster-whisper-{model_size}"
        self.model = WhisperModel(model_size, device=device, compute_type=compute)

    def transcribe(self, wav_path: Path) -> dict:
        segments, _info = self.model.transcribe(str(wav_path), language="fa", vad_filter=False)
        text = verbatim_normalize("".join(seg.text for seg in segments))
        has_fa = any("\u0600" <= ch <= "\u06ff" for ch in text)
        return {
            "transcript_nemo": text,
            "language_status": "ok" if has_fa else "language_uncertain",
            "alignment": [],
            "teacher": self.model_name,
        }


class HttpNemoTeacher:
    """NeMo serving stack over HTTP when local SpeechService cannot prepare audio."""

    def __init__(self):
        self.model_name = "nemo-soroush-http"

    def transcribe(self, wav_path: Path) -> dict:
        from thesis_s2s.runtime.cascade import _wav_bytes, asr_http

        audio, _ = read_wav(wav_path)
        asr = asr_http(_wav_bytes(audio))
        if asr.get("error"):
            raise RuntimeError(str(asr["error"]))
        text = verbatim_normalize(str(asr.get("persian") or asr.get("text") or ""))
        has_fa = any("\u0600" <= ch <= "\u06ff" for ch in text)
        return {
            "transcript_nemo": text or None,
            "language_status": "ok" if has_fa else "language_uncertain",
            "alignment": asr.get("persian_alignment") or [],
            "teacher": self.model_name,
        }


def _probe_teacher(fn, wav: Path) -> bool:
    try:
        extra = fn(wav)
        return bool(extra.get("transcript_nemo"))
    except Exception:
        return False


def _make_local_teacher(*, allow_whisper: bool, prefer_http_nemo: bool):
    """HTTP NeMo first (live serving stack). Whisper only if HTTP is down."""

    errors = {}
    if prefer_http_nemo:
        try:
            return HttpNemoTeacher(), errors
        except Exception as exc:
            errors["http_nemo"] = str(exc)[:200]
    if allow_whisper:
        try:
            return FasterWhisperTeacher(), errors
        except Exception as exc:
            errors["faster_whisper"] = str(exc)[:200]
        try:
            return WhisperTeacher(), errors
        except Exception as exc:
            errors["whisper"] = str(exc)[:200]
    return None, errors


def _transcribe_row(
    row: dict, service, transcribe_chunks, audio_slice, teacher
) -> tuple[dict, str | None]:
    wav = Path(row.get("audio_filepath") or row.get("audio_path") or "")
    try:
        if service is not None:
            extra = reasr_one(service, transcribe_chunks, audio_slice, wav)
        else:
            extra = teacher.transcribe(wav)
        return extra, None
    except Exception as exc:
        return {}, str(exc)[:300]


def run_manifest(
    in_jsonl: Path,
    out_jsonl: Path,
    limit: int | None = None,
    allow_whisper_fallback: bool = True,
    resume: bool = True,
    prefer_http_nemo: bool = True,
    workers: int = 8,
) -> dict:
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    loaded = None if prefer_http_nemo else _load_speech_service()
    stats_skip_silero = False
    if loaded is not None:
        try:
            import silero_vad  # noqa: F401
        except Exception:
            stats_skip_silero = True
            loaded = None
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if resume and out_jsonl.is_file():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row.get("utt_id") or row.get("audio_filepath") or "")
            if row.get("transcript_nemo"):
                done[uid] = row
    stats = {
        "n": 0,
        "ok": len(done),
        "skipped": 0,
        "resumed": len(done),
        "dry_run": False,
        "teacher": "nemo-soroush" if loaded else None,
        "workers": 1,
    }
    if stats_skip_silero:
        stats["nemo_skip"] = "silero_vad_missing"
        loaded = None
    service = transcribe_chunks = audio_slice = None
    teacher = None
    if loaded:
        SpeechService, transcribe_chunks, audio_slice = loaded
        try:
            service = SpeechService()
            service.initialize()
        except Exception as exc:
            stats["nemo_init_error"] = str(exc)[:400]
            loaded = None
            service = None
    # Probe local SpeechService; fall back to HTTP NeMo if prepare cannot run (missing silero/VoxLingua).
    probe_wav = None
    with in_jsonl.open("r", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            cand = Path(row.get("audio_filepath") or row.get("audio_path") or "")
            if cand.is_file():
                probe_wav = cand
                break
    if service is not None and probe_wav is not None:
        ok = _probe_teacher(
            lambda p: reasr_one(service, transcribe_chunks, audio_slice, p), probe_wav
        )
        if not ok:
            stats["nemo_prepare_unusable"] = True
            service = None
            loaded = None
    if loaded is None:
        teacher, errors = _make_local_teacher(
            allow_whisper=allow_whisper_fallback,
            prefer_http_nemo=prefer_http_nemo,
        )
        if errors:
            stats["teacher_errors"] = errors
        if teacher is None:
            stats["dry_run"] = True
            stats["teacher"] = None
        else:
            stats["teacher"] = teacher.model_name
        if (
            teacher is not None
            and probe_wav is not None
            and not _probe_teacher(teacher.transcribe, probe_wav)
        ):
            stats["teacher_probe_error"] = f"{teacher.model_name} failed the input-audio probe"
            teacher, fallback_errors = _make_local_teacher(
                allow_whisper=allow_whisper_fallback,
                prefer_http_nemo=False,
            )
            if fallback_errors:
                stats.setdefault("teacher_errors", {}).update(fallback_errors)
            stats["teacher"] = teacher.model_name if teacher is not None else None
            stats["dry_run"] = teacher is None
    use_pool = service is None and teacher is not None and workers > 1 and not stats["dry_run"]
    stats["workers"] = int(workers) if use_pool else 1
    pool = ThreadPoolExecutor(max_workers=stats["workers"]) if use_pool else None

    tmp_out = out_jsonl.with_suffix(".tmp.jsonl")
    checkpoint_every = 50
    dst = tmp_out.open("w", encoding="utf-8")

    def apply_result(row: dict, extra: dict, err: str | None) -> dict:
        if err:
            row["reasr_error"] = err
            stats["skipped"] += 1
            return row
        row.update(extra)
        if extra.get("transcript_nemo"):
            row["reasr_available"] = True
            stats["ok"] += 1
        else:
            stats["skipped"] += 1
        return row

    try:
        pending: list[dict] = []

        def flush_pending() -> None:
            if not pending:
                return
            if pool is None:
                for row in pending:
                    extra, err = _transcribe_row(
                        row, service, transcribe_chunks, audio_slice, teacher
                    )
                    dst.write(json.dumps(apply_result(row, extra, err), ensure_ascii=False) + "\n")
            else:
                futs = [
                    pool.submit(
                        _transcribe_row, row, service, transcribe_chunks, audio_slice, teacher
                    )
                    for row in pending
                ]
                for row, fut in zip(pending, futs, strict=True):
                    extra, err = fut.result()
                    dst.write(json.dumps(apply_result(row, extra, err), ensure_ascii=False) + "\n")
            pending.clear()

        with in_jsonl.open("r", encoding="utf-8") as src:
            for line in src:
                row = json.loads(line)
                uid = str(row.get("utt_id") or row.get("audio_filepath") or "")
                if uid in done:
                    dst.write(json.dumps(done[uid], ensure_ascii=False) + "\n")
                    continue
                if limit is not None and stats["n"] >= limit:
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue
                stats["n"] += 1
                if stats["dry_run"]:
                    row["transcript_nemo"] = None
                    row["reasr_note"] = "dry_run_speech_service_unavailable"
                    stats["skipped"] += 1
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    pending.append(row)
                    if len(pending) >= max(1, stats["workers"] * 4):
                        flush_pending()
                if stats["n"] % checkpoint_every == 0:
                    flush_pending()
                    dst.flush()
                    shutil.copyfile(tmp_out, out_jsonl)
                    write_json(
                        out_jsonl.with_name("reasr_stats.json"), {**stats, "in_progress": True}
                    )
            flush_pending()
    finally:
        dst.close()
        if pool is not None:
            pool.shutdown(wait=True)
    tmp_out.replace(out_jsonl)
    stats["in_progress"] = False
    write_json(out_jsonl.with_name("reasr_stats.json"), stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-whisper-fallback", action="store_true")
    parser.add_argument("--prefer-http-nemo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_manifest(
                args.in_jsonl,
                args.out_jsonl,
                args.limit,
                allow_whisper_fallback=not args.no_whisper_fallback,
                resume=not args.no_resume,
                prefer_http_nemo=args.prefer_http_nemo,
                workers=args.workers,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
