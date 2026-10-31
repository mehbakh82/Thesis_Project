#!/usr/bin/env python3
"""Inventory of responder variants for responder comparison v3.

Enumerates, from the Hugging Face Hub, every repository that declares itself a
quantization of one of the in-scope official instruct checkpoints (or of
Google's QAT weights for Gemma 4), classifies each repository by quantization
method, applies the selection rule of docs/RESPONDER_V3_PROTOCOL.md, and
records the selected repository (or GGUF file set) per model and method family
with its weight size. Every repository that is not selected is counted with
the reason. The output is the frozen arm list of the comparison.

Usage:
  python scripts/inventory_responder_variants_v3.py --out results/eval/responder_v3_inventory.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

HUB = "https://huggingface.co"
GIB = 1024**3
GGUF_LIMIT_GIB = 21.5  # a GGUF file is what llama.cpp loads; 22 GiB minus a small KV cache
SAFETENSORS_LIMIT_GIB = 26.0  # files include vision and audio towers that text-only serving skips
FALLBACKS = 3  # repositories kept per method family, tried in order if one fails to load

BASES: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3.5-0.8B": {"family": "qwen3.5", "params_b": 0.8, "moe": False},
    "Qwen/Qwen3.5-2B": {"family": "qwen3.5", "params_b": 2.3, "moe": False},
    "Qwen/Qwen3.5-4B": {"family": "qwen3.5", "params_b": 4.7, "moe": False},
    "Qwen/Qwen3.5-9B": {"family": "qwen3.5", "params_b": 9.7, "moe": False},
    "Qwen/Qwen3.5-27B": {"family": "qwen3.5", "params_b": 27.8, "moe": False},
    "Qwen/Qwen3.5-35B-A3B": {"family": "qwen3.5", "params_b": 36.0, "moe": True},
    "Qwen/Qwen3.6-27B": {"family": "qwen3.6", "params_b": 27.8, "moe": False},
    "Qwen/Qwen3.6-35B-A3B": {"family": "qwen3.6", "params_b": 36.0, "moe": True},
    "Qwen/Qwen3.8-27B": {"family": "qwen3.8", "params_b": 27.8, "moe": False},
    "google/gemma-4-E2B-it": {"family": "gemma4", "params_b": 5.1, "moe": False},
    "google/gemma-4-E4B-it": {"family": "gemma4", "params_b": 8.0, "moe": False},
    "google/gemma-4-12B-it": {"family": "gemma4", "params_b": 12.0, "moe": False},
    "google/gemma-4-26B-A4B-it": {"family": "gemma4", "params_b": 26.0, "moe": True},
    "google/gemma-4-31B-it": {"family": "gemma4", "params_b": 31.0, "moe": False},
}

OUT_OF_SCOPE_OFFICIAL = {
    "Qwen/Qwen3.5-122B-A10B": "about 61 GB of weights at 4 bits; does not fit a 24 GB card",
    "Qwen/Qwen3.5-397B-A17B": "about 200 GB of weights at 4 bits",
    "Qwen/Qwen3.8-Flash-Next": "180B parameters; about 90 GB at 4 bits",
    "Qwen/Qwen3.8-2.4T-A95B": "2.4T parameters",
}

# Repositories whose name marks a fine-tune, merge, behaviour edit, pruning, or a
# packaging for another runtime are not plain quantizations of the official model.
NOT_PLAIN = re.compile(
    r"abliterat|uncensor|heretic|obliterat|distill|merge|fusion|finetun|-ft\b|lora|roleplay|"
    r"\brp\b|opus|qwopus|claude|coder|redteam|red-team|nsfw|prism-pro|artemis|zhtw|japanese|"
    r"typescript|extract|emperor|crow-|revised|instruct-gguf|reap|prun|xvita|turbo|dashq|"
    r"-local-|oq\d|super-|thinking|reasoning|vision-only|text-only-ft|sft|dpo|grpo|rl-|"
    r"toolcall|agent|math|medical|legal|persian|farsi",
    re.IGNORECASE,
)
OTHER_RUNTIME = re.compile(
    r"mlx|onnx|rkllm|mnn|litert|openvino|exl2|exl3|ninfer|coreml|tflite|ov-|qnn|sm8\d{3}|rdna|npu|hailo|axera|"
    r"rk35|ryzen|quark-rdna|permuted|ingest",
    re.IGNORECASE,
)
# Quantization-aware-trained ternary and binary builds that need the publisher's llama.cpp fork.
TERNARY_PUBLISHERS = {"prism-ml"}
TERNARY_FORMATS = ("PQ2_0", "PTQ1_0", "Q2_0", "Q1_0")
OFFICIAL_ORGS = {"Qwen", "google"}
# Publishers that quantize plainly and at scale; a GGUF ladder is taken from the
# most-downloaded of these for each model.
GGUF_PUBLISHERS = {"unsloth", "bartowski", "lmstudio-community", "ggml-org", "mradermacher", "QuantFactory"}
GGUF_LADDER = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M", "Q2_K")
GGUF_DYNAMIC = ("UD-Q4_K_XL", "UD-Q2_K_XL")


def _get(path: str, retries: int = 4) -> Any:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{HUB}{path}", timeout=120) as response:
                return json.loads(response.read())
        except Exception:  # noqa: BLE001 - network retries
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def children(parent: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {"filter": f"base_model:quantized:{parent}", "sort": "downloads", "direction": -1, "limit": 1000}
    )
    return _get(f"/api/models?{query}")


def tree(repo: str) -> list[dict[str, Any]]:
    return [f for f in _get(f"/api/models/{repo}/tree/main?recursive=true") if f.get("type") == "file"]


def config(repo: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{HUB}/{repo}/resolve/main/config.json", timeout=60) as response:
            return json.loads(response.read())
    except Exception:  # noqa: BLE001 - absent or gated config
        return {}


def _norm(text: str) -> str:
    text = text.lower().replace("_", "-")
    for prefix in ("qwen-", "google-"):
        if text.startswith(prefix) and not text.startswith(("qwen-qwen", "google-gemma")):
            continue
    return text.replace("qwen-qwen", "qwen").replace("google-gemma", "gemma")


def stem(base: str) -> str:
    return base.split("/")[1].lower().removesuffix("-it")


def method_family(repo: str, tags: list[str], cfg: dict[str, Any] | None) -> str | None:
    name = repo.lower()
    if "gguf" in tags or "gguf" in name:
        return "gguf"
    q = (cfg or {}).get("quantization_config") or (cfg or {}).get("text_config", {}).get("quantization_config") or {}
    method = str(q.get("quant_method") or "").lower()
    fmt = json.dumps(q).lower()
    if "nvfp4" in name or "nvfp4" in fmt or method == "modelopt_fp4":
        return "nvfp4"
    if "mxfp4" in name or "mxfp4" in fmt:
        return "mxfp4"
    if "bnb" in name or "nf4" in name or method == "bitsandbytes":
        return "bnb_4bit" if re.search(r"4bit|4-bit|nf4", name) or q.get("load_in_4bit") else "bnb_other"
    if "autoround" in name or method in {"auto-round", "autoround"}:
        return "autoround_int4" if not re.search(r"8bit|int8|mixed", name) else "autoround_other"
    if "awq" in name or method == "awq":
        return "awq_int4" if not re.search(r"8bit|6bit|int8", name) else "awq_other"
    if "gptq" in name or method == "gptq":
        return "gptq_int4" if not re.search(r"8bit|int8", name) else "gptq_int8"
    if "fp8" in name or method in {"fp8", "fbgemm_fp8"}:
        return "fp8"
    if re.search(r"w8a8|int8|w8a16", name) or '"num_bits": 8' in fmt:
        return "int8"
    if re.search(r"w4a16|int4", name) or '"num_bits": 4' in fmt:
        return "int4_compressed_tensors"
    return None


def weight_bytes(files: list[dict[str, Any]], suffix: str = ".safetensors") -> int:
    return sum(int(f.get("size") or 0) for f in files if f["path"].endswith(suffix))


def gguf_rungs(files: list[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    rungs: dict[str, dict[str, Any]] = {}
    ggufs = [f for f in files if f["path"].endswith(".gguf") and not re.search(r"mmproj|dspark|-f16|bf16", f["path"], re.IGNORECASE)]
    for label in labels:
        pattern = re.compile(rf"(^|[-_.]){re.escape(label)}(\.gguf|-\d{{5}}-of-\d{{5}}\.gguf)$", re.IGNORECASE)
        hits = [f for f in ggufs if pattern.search(Path(f["path"]).name)]
        if not label.startswith("UD-"):
            hits = [f for f in hits if "UD-" not in f["path"]]
        if hits:
            size = sum(int(f.get("size") or 0) for f in hits)
            rungs[label] = {"files": sorted(f["path"] for f in hits), "bytes": size, "fits_by_weights": size / GIB <= GGUF_LIMIT_GIB}
    return rungs


def official_releases(base: str) -> list[str]:
    org, name = base.split("/")
    query = urllib.parse.urlencode({"author": org, "search": stem(base), "limit": 200})
    keep = []
    for repo in (m["id"] for m in _get(f"/api/models?{query}")):
        tail = repo.split("/")[1]
        if not tail.lower().startswith(stem(base)):
            continue
        if re.search(r"-Base$|-pt$|assistant|SAE-|mobile|unquantized", tail):
            continue
        if base.startswith("google/") and "-it" not in tail:
            continue
        keep.append(repo)
    return sorted(set(keep))


def _official_arm(repo: str) -> tuple[str, dict[str, Any]]:
    tail = repo.split("/")[1].lower()
    files = tree(repo)
    if tail.endswith("-gguf"):
        return "official_qat_q4_0_gguf", {"repo": repo, "official": True, "rungs": gguf_rungs(files, ("Q4_0",))}
    key = "official_fp8" if tail.endswith("fp8") else "official_gptq_int4" if "gptq" in tail else "official_qat_w4a16" if "w4a16" in tail else "official_bf16"
    size = weight_bytes(files)
    return key, {"repo": repo, "official": True, "bytes": size, "fits_by_weights": size / GIB <= SAFETENSORS_LIMIT_GIB}


def inventory() -> dict[str, Any]:
    result: dict[str, Any] = {"models": {}, "out_of_scope_official": OUT_OF_SCOPE_OFFICIAL}
    for base, meta in BASES.items():
        parents = [base] + ([f"{base}-qat-q4_0-unquantized"] if base.startswith("google/") else [])
        seen: dict[str, dict[str, Any]] = {}
        for parent in parents:
            for m in children(parent):
                seen.setdefault(m["id"], {"id": m["id"], "downloads": m.get("downloads", 0), "tags": m.get("tags", [])})
        repos = sorted(seen.values(), key=lambda m: -m["downloads"])
        reasons: Counter[str] = Counter()
        by_family: dict[str, list[dict[str, Any]]] = {}
        ternary: list[dict[str, Any]] = []
        for m in repos:
            owner = m["id"].split("/")[0]
            if owner in TERNARY_PUBLISHERS and "gguf" in m["id"].lower():
                ternary.append(m)
                continue
            if stem(base) not in _norm(m["id"].split("/")[1]):
                reasons["name does not carry the model name (derived or renamed model)"] += 1
                continue
            if NOT_PLAIN.search(m["id"]):
                reasons["not a plain quantization (fine-tune, merge, behaviour edit, pruning)"] += 1
                continue
            if OTHER_RUNTIME.search(m["id"]) or {"mlx", "onnx"} & set(m["tags"]):
                reasons["packaged for another runtime or device (MLX, ONNX, EXL2/3, NPU)"] += 1
                continue
            by_family.setdefault(method_family(m["id"], m["tags"], None) or "unclassified", []).append(m)
        arms: dict[str, dict[str, Any]] = {}
        official = official_releases(base)
        for repo in official:
            key, arm = _official_arm(repo)
            arms[key] = arm
        for family, members in sorted(by_family.items()):
            if family == "gguf":
                ladder = [m for m in members if m["id"].split("/")[0] in GGUF_PUBLISHERS and not re.search(r"mtp|qat", m["id"], re.IGNORECASE)]
                rungs: dict[str, dict[str, Any]] = {}
                for m in ladder:  # complete the ladder from the next publisher if a rung is missing
                    if all(label in rungs for label in GGUF_LADDER + GGUF_DYNAMIC):
                        break
                    for label, rung in gguf_rungs(tree(m["id"]), GGUF_LADDER + GGUF_DYNAMIC).items():
                        rungs.setdefault(label, {"repo": m["id"], **rung})
                if rungs:
                    arms["gguf_ladder"] = {"publisher_order": [m["id"] for m in ladder], "rungs": rungs}
                reasons["GGUF repository not used for the ladder"] += len(members) - len({r["repo"] for r in rungs.values()})
                continue
            if family in {"unclassified", "awq_other", "gptq_int8", "bnb_other", "autoround_other"}:
                reasons[f"method family not in scope ({family})"] += len(members)
                continue
            official_key = {"fp8": "official_fp8", "gptq_int4": "official_gptq_int4", "int4_compressed_tensors": "official_qat_w4a16"}.get(family)
            if official_key and official_key in arms and arms[official_key].get("fits_by_weights"):
                reasons[f"{family}: official release preferred"] += len(members)
                continue
            picks: list[dict[str, Any]] = []
            for m in members[:12]:
                if len(picks) == FALLBACKS:
                    break
                cfg = config(m["id"])
                if method_family(m["id"], m["tags"], cfg) != family:
                    continue
                size = weight_bytes(tree(m["id"]))
                if size == 0:
                    continue
                picks.append({"repo": m["id"], "downloads": m["downloads"], "bytes": size, "fits_by_weights": size / GIB <= SAFETENSORS_LIMIT_GIB})
            fitting = [p for p in picks if p["fits_by_weights"]]
            if not picks:
                reasons[f"{family}: no repository with a matching quantization config"] += len(members)
                continue
            arms[family] = {**(fitting or picks)[0], "fallbacks": [p["repo"] for p in fitting[1:]]}
            reasons[f"{family}: less-downloaded repository of the same method"] += len(members) - 1
        if ternary:
            rungs = {}
            for m in ternary:
                for label, rung in gguf_rungs(tree(m["id"]), TERNARY_FORMATS).items():
                    rungs.setdefault(f"{m['id'].split('/')[1]}:{label}", {"repo": m["id"], **rung})
            arms["ternary_qat_gguf"] = {"engine": "PrismML llama.cpp fork", "rungs": rungs}
        result["models"][base] = {
            **meta,
            "quantized_repositories_declared": len(repos),
            "official_releases": official,
            "arms": arms,
            "not_selected": dict(reasons),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "protocol": "docs/RESPONDER_V3_PROTOCOL.md",
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "gguf_limit_gib": GGUF_LIMIT_GIB,
        "safetensors_limit_gib": SAFETENSORS_LIMIT_GIB,
        **inventory(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
