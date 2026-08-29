#!/usr/bin/env python3
"""Launch and attest the selected adapter in the official streaming server."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_moshi_client import tree_manifest  # noqa: E402
from scripts.moshi_runtime_panel import user_audio_bounds_seconds  # noqa: E402
from scripts.validate_moshi_adapter import (  # noqa: E402
    json_object,
    project_path,
    sha256_file,
)

ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(value: str) -> str:
    return ANSI_PATTERN.sub("", value)


BYTE_PIECE_PATTERN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def decode_server_token_pieces(value: str) -> str:
    """Decode SentencePiece byte-fallback markers emitted by the upstream server."""

    output: list[str] = []
    pending = bytearray()

    def flush_pending() -> None:
        if pending:
            output.append(bytes(pending).decode("utf-8", errors="replace"))
            pending.clear()

    position = 0
    for match in BYTE_PIECE_PATTERN.finditer(value):
        between = value[position : match.start()]
        if between:
            flush_pending()
            output.append(between)
        pending.append(int(match.group(1), 16))
        position = match.end()
    flush_pending()
    output.append(value[position:])
    return "".join(output)


def script_statistics(value: str) -> dict[str, float | int]:
    letters = [character for character in value if character.isalpha()]
    persian_letters = sum("\u0600" <= character <= "\u06ff" for character in letters)
    latin_letters = sum(character.isascii() for character in letters)
    denominator = persian_letters + latin_letters
    return {
        "persian_letters": persian_letters,
        "latin_letters": latin_letters,
        "persian_letter_fraction": persian_letters / denominator if denominator else 0.0,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("nonempty values and a quantile in [0, 1] are required")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def ensure_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))


def wait_for_http(url: str, process: subprocess.Popen[str], timeout: float) -> float:
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Moshi server exited before readiness: {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:  # noqa: S310
                if response.status == 200:
                    return time.monotonic() - started
        except OSError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Moshi server did not become ready within {timeout:.1f}s")


def process_gpu_memory_mib(pid: int) -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and int(fields[0]) == pid:
            return int(fields[1])
    return None


def listening_socket_report(port: int) -> dict[str, Any]:
    result = subprocess.run(
        ["ss", "-ltn", f"sport = :{port}"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    listener_lines = lines[1:]
    body = "\n".join(listener_lines)
    local_addresses = [fields[3] for line in listener_lines if len(fields := line.split()) >= 4]
    return {
        "output": body,
        "local_addresses": local_addresses,
        "loopback_only": bool(local_addresses)
        and all(address.startswith(("127.0.0.1:", "[::1]:")) for address in local_addresses),
    }


async def exercise_server(
    *,
    host: str,
    port: int,
    static_index: Path,
    input_audio: Path,
    input_seconds: float,
    response_timeout: float,
    post_input_silence_seconds: float,
) -> dict[str, Any]:
    import aiohttp
    import numpy as np
    import sphn

    base_url = f"http://{host}:{port}"
    async with aiohttp.ClientSession() as session:
        async with session.get(base_url + "/") as response:
            static_body = await response.read()
            static_status = response.status

        pcm, sample_rate = sphn.read(str(input_audio), sample_rate=24000)
        if pcm.ndim != 2 or pcm.shape[0] != 2:
            raise RuntimeError("runtime smoke input must be stereo assistant/user audio")
        source_user_start_seconds, source_user_end_seconds = user_audio_bounds_seconds(
            input_audio
        )
        input_samples = min(pcm.shape[-1], int(input_seconds * sample_rate))
        user_pcm = pcm[1, :input_samples]
        complete_user_audio_streamed = (
            source_user_end_seconds <= input_samples / sample_rate
        )
        post_silence_samples = int(post_input_silence_seconds * sample_rate)
        streamed_pcm = np.concatenate(
            [user_pcm, np.zeros(post_silence_samples, dtype=user_pcm.dtype)]
        )
        minimum_decoded_samples = int(0.8 * streamed_pcm.size)
        websocket_url = f"ws://{host}:{port}/api/chat"
        async with session.ws_connect(websocket_url, max_msg_size=16 * 1024 * 1024) as ws:
            handshake = await ws.receive(timeout=10)
            handshake_valid = (
                handshake.type == aiohttp.WSMsgType.BINARY and handshake.data == b"\x00"
            )
            counters = {
                "audio_messages": 0,
                "audio_payload_bytes": 0,
                "text_messages": 0,
                "text_utf8_bytes": 0,
                "unexpected_messages": 0,
            }
            decoded_chunks = []
            opus_reader = sphn.OpusStreamReader(sample_rate)
            text_fragments: list[str] = []

            async def receive_outputs() -> None:
                deadline = time.monotonic() + response_timeout
                while time.monotonic() < deadline:
                    try:
                        message = await ws.receive(timeout=1)
                    except TimeoutError:
                        continue
                    if message.type in {
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        return
                    if message.type != aiohttp.WSMsgType.BINARY or not message.data:
                        counters["unexpected_messages"] += 1
                        continue
                    kind = message.data[0]
                    payload = message.data[1:]
                    if kind == 1:
                        counters["audio_messages"] += 1
                        counters["audio_payload_bytes"] += len(payload)
                        opus_reader.append_bytes(payload)
                        decoded = opus_reader.read_pcm()
                        if decoded.size:
                            decoded_chunks.append(decoded)
                    elif kind == 2:
                        counters["text_messages"] += 1
                        counters["text_utf8_bytes"] += len(payload)
                        text_fragment = payload.decode("utf-8")
                        text_fragments.append(text_fragment)
                    else:
                        counters["unexpected_messages"] += 1
                    decoded_samples = sum(chunk.size for chunk in decoded_chunks)
                    if decoded_samples >= minimum_decoded_samples:
                        return

            receiver = asyncio.create_task(receive_outputs())
            opus_writer = sphn.OpusStreamWriter(sample_rate)
            for start in range(0, streamed_pcm.size, 1920):
                opus_writer.append_pcm(streamed_pcm[start : start + 1920])
                payload = opus_writer.read_bytes()
                if payload:
                    await ws.send_bytes(b"\x01" + payload)
                await asyncio.sleep(0.002)
            await receiver

        async with session.ws_connect(websocket_url) as reconnect:
            reconnect_handshake = await reconnect.receive(timeout=10)
            reconnect_valid = (
                reconnect_handshake.type == aiohttp.WSMsgType.BINARY
                and reconnect_handshake.data == b"\x00"
            )

    decoded_pcm = (
        np.concatenate([np.asarray(chunk).reshape(-1) for chunk in decoded_chunks])
        if decoded_chunks
        else np.array([], dtype=np.float32)
    )
    decoded_finite = bool(decoded_pcm.size) and bool(np.isfinite(decoded_pcm).all())
    decoded_rms = float(np.sqrt(np.mean(np.square(decoded_pcm)))) if decoded_pcm.size else 0.0
    decoded_peak = float(np.max(np.abs(decoded_pcm))) if decoded_pcm.size else 0.0
    raw_generated_text = "".join(text_fragments)
    decoded_generated_text = decode_server_token_pieces(raw_generated_text)
    generated_text_script = script_statistics(decoded_generated_text)
    return {
        "static_http_status": static_status,
        "static_index_bytes": len(static_body),
        "static_index_sha256": hashlib.sha256(static_body).hexdigest(),
        "static_index_exact": static_body == static_index.read_bytes(),
        "websocket_handshake_valid": handshake_valid,
        "reconnect_handshake_valid": reconnect_valid,
        "input_sample_rate": sample_rate,
        "input_samples": input_samples,
        "input_audio_seconds": input_samples / sample_rate,
        "source_audio_samples": int(pcm.shape[-1]),
        "source_audio_seconds": pcm.shape[-1] / sample_rate,
        "user_audio_start_seconds": source_user_start_seconds,
        "user_audio_end_seconds": source_user_end_seconds,
        "complete_user_audio_streamed": complete_user_audio_streamed,
        "post_input_silence_seconds": post_silence_samples / sample_rate,
        "total_streamed_samples": int(streamed_pcm.size),
        "total_streamed_seconds": streamed_pcm.size / sample_rate,
        **counters,
        "decoded_audio_samples": int(decoded_pcm.size),
        "decoded_audio_finite": decoded_finite,
        "decoded_audio_rms": decoded_rms,
        "decoded_audio_peak": decoded_peak,
        "decoded_audio_sha256": hashlib.sha256(decoded_pcm.astype("float32").tobytes()).hexdigest()
        if decoded_pcm.size
        else None,
        "generated_audio_retained": False,
        "generated_text_token_pieces": raw_generated_text,
        "generated_text": decoded_generated_text,
        "generated_text_sha256": hashlib.sha256(decoded_generated_text.encode("utf-8")).hexdigest(),
        "generated_text_script": generated_text_script,
        "generated_text_retained": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18998)
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--response-timeout", type=float, default=20.0)
    parser.add_argument("--validation-index", type=int, default=0)
    parser.add_argument("--input-seconds", type=float, default=5.0)
    parser.add_argument("--post-input-silence-seconds", type=float, default=3.0)
    parser.add_argument(
        "--base-only-diagnostic",
        action="store_true",
        help="Run the same smoke without the selected adapter for diagnosis.",
    )
    parser.add_argument(
        "--candidate-step",
        type=int,
        help="Run a saved validation candidate instead of the selected adapter.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_h100_runtime_smoke.json"),
    )
    args = parser.parse_args()
    if args.base_only_diagnostic and args.candidate_step is not None:
        raise ValueError("base-only and candidate-step modes are mutually exclusive")
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("development smoke is restricted to a loopback host")
    ensure_port_available("127.0.0.1", args.port)

    selection_path = ROOT / "results/moshi_checkpoint_selection.json"
    validation_path = ROOT / "results/moshi_adapter_validation.json"
    heldout_path = ROOT / "results/moshi_heldout_model_eval.json"
    client_report_path = ROOT / "results/hardware/moshi_client_build.json"
    training_report_path = ROOT / "results/hardware/moshi_h100_training.json"
    training_config_path = ROOT / "configs/moshi_h100.yaml"
    validation_manifest_path = ROOT / "data/processed/moshi_finetune/val.jsonl"
    selection = json_object(selection_path)
    validation = json_object(validation_path)
    heldout = json_object(heldout_path)
    client_report = json_object(client_report_path)
    training_report = json_object(training_report_path)
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if not isinstance(training_config, dict):
        raise ValueError("training config must be a mapping")
    selected = selection["selected"]
    selected_adapter_path = project_path(ROOT, selected["adapter_path"], "selected.adapter_path")
    selected_config_path = project_path(ROOT, selected["config_path"], "selected.config_path")
    runtime_candidate = selected
    if args.candidate_step is not None:
        matches = [
            candidate
            for candidate in selection["candidates"]
            if candidate["step"] == args.candidate_step
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown candidate step: {args.candidate_step}")
        runtime_candidate = matches[0]
    adapter_path = project_path(
        ROOT, runtime_candidate["adapter_path"], "runtime_candidate.adapter_path"
    )
    adapter_config_path = project_path(
        ROOT, runtime_candidate["config_path"], "runtime_candidate.config_path"
    )
    moshi_paths = training_config["moshi_paths"]
    base_config_path = project_path(ROOT, moshi_paths["config_path"], "config_path")
    base_model_path = project_path(ROOT, moshi_paths["moshi_path"], "moshi_path")
    mimi_path = project_path(ROOT, moshi_paths["mimi_path"], "mimi_path")
    tokenizer_path = project_path(ROOT, moshi_paths["tokenizer_path"], "tokenizer_path")
    static_dir = (ROOT / client_report["dist"]["path"]).resolve()
    static_files, static_tree_hash = tree_manifest(static_dir)
    validation_rows = [
        json.loads(line)
        for line in validation_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 0 <= args.validation_index < len(validation_rows):
        raise ValueError(f"validation index out of range: {args.validation_index}")
    input_row = validation_rows[args.validation_index]
    input_audio_path = Path(input_row["path"]).resolve()

    provenance_preconditions = {
        "selection_passed": selection.get("selection_passes") is True,
        "adapter_validation_passed": validation.get("validation_passes") is True,
        "official_cuda_loader_passed": validation.get("runtime_load", {}).get("passed") is True,
        "heldout_evaluation_passed": heldout.get("evaluation_passes") is True,
        "training_certificate_passed": training_report.get("training_run_passes") is True,
        "client_build_valid": client_report.get("valid") is True,
        "client_tree_hash_current": static_tree_hash == client_report["dist"]["tree_sha256"],
        "client_file_count_current": len(static_files) == len(client_report["dist"]["files"]),
        "selected_adapter_hash_current": sha256_file(selected_adapter_path)
        == selected["adapter_sha256"],
        "selected_config_hash_current": sha256_file(selected_config_path)
        == selected["config_sha256"],
        "runtime_adapter_hash_current": sha256_file(adapter_path)
        == runtime_candidate["adapter_sha256"],
        "runtime_config_hash_current": sha256_file(adapter_config_path)
        == runtime_candidate["config_sha256"],
        "base_hash_current": sha256_file(base_model_path)
        == heldout["provenance"]["base_model_sha256"],
        "mimi_hash_current": sha256_file(mimi_path) == heldout["provenance"]["mimi_sha256"],
        "tokenizer_hash_current": sha256_file(tokenizer_path)
        == heldout["provenance"]["tokenizer_sha256"],
        "validation_only_input": validation_manifest_path.name == "val.jsonl",
        "input_audio_hash_current": sha256_file(input_audio_path) == input_row["sha256"],
    }
    if not all(provenance_preconditions.values()):
        raise RuntimeError(f"runtime provenance preconditions failed: {provenance_preconditions}")

    runtime_config_path = base_config_path if args.base_only_diagnostic else adapter_config_path
    command = [
        sys.executable,
        "scripts/moshi_server_entry.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--static",
        str(static_dir),
        "--hf-repo",
        str(moshi_paths["hf_repo_id"]),
        "--moshi-weight",
        str(base_model_path),
        "--mimi-weight",
        str(mimi_path),
        "--tokenizer",
        str(tokenizer_path),
        "--config-path",
        str(runtime_config_path),
    ]
    if not args.base_only_diagnostic:
        command.extend(["--lora-weight", str(adapter_path)])
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    server_output = ""
    try:
        ready_seconds = wait_for_http(
            f"http://{args.host}:{args.port}/",
            process,
            args.ready_timeout,
        )
        socket_report = listening_socket_report(args.port)
        gpu_memory_mib = process_gpu_memory_mib(process.pid)
        exercise = asyncio.run(
            exercise_server(
                host=args.host,
                port=args.port,
                static_index=static_dir / "index.html",
                input_audio=input_audio_path,
                input_seconds=args.input_seconds,
                response_timeout=args.response_timeout,
                post_input_silence_seconds=args.post_input_silence_seconds,
            )
        )
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
        try:
            server_output, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            server_output, _ = process.communicate(timeout=10)

    clean_output = strip_ansi(server_output)
    frame_times_ms = [
        float(value) for value in re.findall(r"frame handled in ([0-9.]+)ms", clean_output)
    ]
    requirements = {
        **provenance_preconditions,
        "server_process_exited_cleanly": process.returncode == 0,
        "server_bound_loopback_only": socket_report["loopback_only"] is True,
        "official_server_loaded_mimi_and_moshi": all(
            marker in clean_output
            for marker in ("mimi loaded", "moshi loaded", "warming up the model")
        ),
        "attested_static_bundle_served_exactly": (
            exercise["static_http_status"] == 200 and exercise["static_index_exact"] is True
        ),
        "official_websocket_handshake_passed": exercise["websocket_handshake_valid"] is True,
        "reconnect_handshake_passed": exercise["reconnect_handshake_valid"] is True,
        "generated_audio_messages_received": (
            exercise["audio_messages"] > 0 and exercise["audio_payload_bytes"] > 0
        ),
        "generated_audio_decodes_finite": exercise["decoded_audio_samples"] > 0
        and exercise["decoded_audio_finite"] is True,
        "no_unexpected_websocket_messages": exercise["unexpected_messages"] == 0,
        "server_frame_processing_observed": bool(frame_times_ms),
        "real_model_gpu_allocation_observed": gpu_memory_mib is not None
        and gpu_memory_mib > 10_000,
        "no_cascade_or_tts_fallback_in_command": all(
            forbidden not in " ".join(command).lower() for forbidden in ("piper", "qwen", "formant")
        ),
    }
    output_requirements = {
        "speech_energy_observed": exercise["decoded_audio_rms"] >= 1e-3,
        "text_tokens_observed": exercise["text_messages"] > 0,
    }
    if not args.base_only_diagnostic:
        output_requirements["persian_script_observed"] = (
            exercise["generated_text_script"]["persian_letter_fraction"] >= 0.5
        )
    runtime_passed = all(requirements.values())
    output_passed = all(output_requirements.values())
    passed = runtime_passed and output_passed
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "development_runtime_evidence": True,
        "scientific_training_evidence": False,
        "official_4090_evidence": False,
        "browser_e2e_evidence": False,
        "human_perceptual_evidence": False,
        "purpose": (
            "H100 development smoke using the same pinned official streaming server and "
            "validation input; diagnostic only and not perceptual or target-hardware evidence"
        ),
        "model_variant": (
            "base-control"
            if args.base_only_diagnostic
            else f"candidate-step-{runtime_candidate['step']:06d}"
        ),
        "server": {
            "entry": "scripts/moshi_server_entry.py",
            "host": args.host,
            "port": args.port,
            "ready_seconds": ready_seconds,
            "process_gpu_memory_mib": gpu_memory_mib,
            "socket": socket_report,
            "exit_code": process.returncode,
            "fuse_lora": True,
            "adapter_loaded": not args.base_only_diagnostic,
            "frame_processing_ms": {
                "count": len(frame_times_ms),
                "mean": sum(frame_times_ms) / len(frame_times_ms),
                "p95": percentile(frame_times_ms, 0.95),
                "maximum": max(frame_times_ms),
                "official_latency_metric": False,
            },
        },
        "input": {
            "split": "validation",
            "validation_index": args.validation_index,
            "manifest": validation_manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(validation_manifest_path),
            "pair_id": input_audio_path.stem,
            "audio_sha256": sha256_file(input_audio_path),
            "channel": 1,
            "retained": False,
        },
        "stream": exercise,
        "runtime_requirements": requirements,
        "output_requirements": output_requirements,
        "speech_energy_threshold_rms": 1e-3,
        "runtime_smoke_passes": runtime_passed,
        "autoregressive_output_gate_passes": output_passed,
        "overall_passes": passed,
        "artifacts": {
            "selection_sha256": sha256_file(selection_path),
            "adapter_validation_sha256": sha256_file(validation_path),
            "training_certificate_sha256": sha256_file(training_report_path),
            "client_build_report_sha256": sha256_file(client_report_path),
            "client_tree_sha256": static_tree_hash,
            "adapter_sha256": sha256_file(adapter_path),
            "adapter_config_sha256": sha256_file(adapter_config_path),
            "runtime_config_sha256": sha256_file(runtime_config_path),
            "base_model_sha256": sha256_file(base_model_path),
            "mimi_sha256": sha256_file(mimi_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "server_entry_sha256": sha256_file(ROOT / "scripts/moshi_server_entry.py"),
            "runtime_validator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "limitations": (
            "Generated PCM is checked mechanically and not retained; generated token-piece text "
            "is retained. No claim is made about Persian intelligibility, naturalness, relevance, "
            "microphone continuity, browser cancellation, interruption context, or 4090 latency/fit."
        ),
    }
    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(
            f"Moshi runtime or autoregressive output gate failed: {requirements}, {output_requirements}"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
