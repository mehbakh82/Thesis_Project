#!/usr/bin/env python3
"""Exploratory: where does llama.cpp spend the first-token time of Gemma 4 12B Q5_K_M?

Responder comparison v3 (docs/RESPONDER_V3_RESULT.md) found the llama.cpp builds of Gemma 4 12B fit the 12 GiB
profile but take 169-214 ms to the first token, against about 40 ms for the vLLM builds. This script serves the
same build with the v3 harness's llama.cpp settings and with one setting changed at a time, sends the first 40
development rows one by one, and records the client first-token time and llama.cpp's own prompt and generation
timings. It changes nothing in the comparison; its receipt is exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_responder_v3 as v3  # noqa: E402

BASE_ARGS = ["-ngl", "999", "-fa", "on", "-c", "4096", "-np", "2", "-b", "2048", "-ub", "512", "--jinja", "--no-webui",
             "--fit", "off", "-lv", "4"]
# name: (server argument changes, request body changes); "base" is the v3 harness configuration. A server change
# maps a flag to its new value, None to remove it, or True to add it without a value.
VARIANTS: dict[str, tuple[dict[str, str | bool | None], dict[str, Any]]] = {
    "base": ({}, {}),
    "quiet_log": ({"-lv": None}, {}),
    "no_grammar": ({}, {"grammar": None}),
    "ubatch_2048": ({"-ub": "2048"}, {}),
    "one_slot": ({"-np": "1", "-c": "2048"}, {}),
    "no_prompt_cache": ({}, {"cache_prompt": False}),
    # Gemma 4 uses sliding-window attention; llama.cpp then saves and restores context checkpoints of the window
    # cache around every reused prompt. These two variants remove them in the two ways the server offers.
    "no_checkpoints": ({"--ctx-checkpoints": "0"}, {}),
    "swa_full": ({"--swa-full": True}, {}),
}


def server_args(changes: dict[str, str | bool | None]) -> list[str]:
    args = list(BASE_ARGS)
    for flag, value in changes.items():
        if flag not in args:
            args += [flag] if value is True else [flag, str(value)]
            continue
        i = args.index(flag)
        if value is None:
            del args[i:i + 2]
        else:
            args[i + 1] = str(value)
    return args


def gpu_memory_mib(pid: int) -> float | None:
    """GPU memory of the server process (llama.cpp allocates its buffers at start-up)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    used = [float(line.split(",")[1]) for line in out.splitlines() if line.split(",")[0].strip() == str(pid)]
    return used[0] if used else None


def request(row: dict[str, Any], system: str, body_changes: dict[str, Any], port: int) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "candidate", "messages": v3.messages_for(row, system), "max_tokens": v3.MAX_NEW_TOKENS,
                            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                            "chat_template_kwargs": {"enable_thinking": False}, "grammar": v3.GBNF, "top_k": 1,
                            "cache_prompt": True}
    for key, value in body_changes.items():
        if value is None:
            body.pop(key, None)
        else:
            body[key] = value
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    first = None
    timings: dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=300) as response:  # noqa: S310 - local server
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            event = json.loads(line[5:])
            timings = event.get("timings") or timings
            choices = event.get("choices") or []
            if first is None and choices and (choices[0].get("delta") or {}).get("content"):
                first = time.perf_counter() - started
    return {"first_token_ms": None if first is None else 1000 * first, "prompt_n": timings.get("prompt_n"),
            "prompt_ms": timings.get("prompt_ms"), "predicted_per_token_ms": timings.get("predicted_per_token_ms")}


def run_variant(name: str, model: Path, rows: list[dict[str, Any]], system: str, port: int, log: Path) -> dict[str, Any]:
    changes, body_changes = VARIANTS[name]
    binary = os.environ.get("THESIS_LLAMA_SERVER", "llama-server")
    cmd = [binary, "-m", str(model), *server_args(changes), "--host", "127.0.0.1", "--port", str(port)]
    with log.open("wb") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:  # noqa: S310 - local
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001 - not ready yet
                time.sleep(1)
        for row in rows[-2:]:  # warm-up, as in the v3 harness
            request(row, system, body_changes, port)
        results = [request(row, system, body_changes, port) for row in rows[:40]]
        memory = gpu_memory_mib(proc.pid)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    first = [r["first_token_ms"] for r in results if r["first_token_ms"] is not None]
    prompt = [r["prompt_ms"] for r in results if r["prompt_ms"] is not None]
    per_token = [r["predicted_per_token_ms"] for r in results if r["predicted_per_token_ms"] is not None]
    return {"server_changes": changes, "request_changes": {k: v for k, v in body_changes.items() if k != "grammar"} | (
                {"grammar": "removed"} if "grammar" in body_changes else {}),
            "rows": len(results), "first_token_median_ms": round(statistics.median(first), 1) if first else None,
            "prompt_ms_median": round(statistics.median(prompt), 1) if prompt else None,
            "prompt_tokens_median": statistics.median(r["prompt_n"] for r in results if r["prompt_n"] is not None)
            if prompt else None,
            "per_token_ms_median": round(statistics.median(per_token), 2) if per_token else None,
            "latin_free_note": "grammar removed: replies may contain Latin letters" if "grammar" in body_changes else None,
            "gpu_memory_mib": memory, "cpu_load1_at_end": os.getloadavg()[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="the Gemma 4 12B Q5_K_M GGUF file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--port", type=int, default=v3.LLAMA_PORT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    rows = v3.read_jsonl(args.work / "panel_dev.jsonl")
    system = v3.elderly_prompt()
    results = {}
    for name in args.variants:
        results[name] = run_variant(name, args.model, rows, system, args.port, args.work / "logs" / f"diagnose_llamacpp_{name}.log")
        print(name, json.dumps(results[name]), flush=True)
    receipt = {"schema_version": 1, "experiment": "llamacpp_first_token_diagnosis", "label": "exploratory",
               "protocol": "docs/RESPONDER_V3_RESULT.md", "model_file": args.model.name,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "panel_dev_sha256": hashlib.sha256((args.work / "panel_dev.jsonl").read_bytes()).hexdigest(),
               "base_server_args": BASE_ARGS, "variants": results, "text_retained": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
