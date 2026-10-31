from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = _load("responder_v3", "scripts/evaluate_responder_v3.py")
inventory = _load("responder_v3_inventory", "scripts/inventory_responder_variants_v3.py")


def test_llama_cpp_memory_lines_are_summed() -> None:
    log = "\n".join([
        "load_tensors:        CUDA0 model buffer size =   497.40 MiB",
        "load_tensors:   CPU_Mapped model buffer size =   198.93 MiB",
        "llama_kv_cache: size =   48.00 MiB (  2048 cells,   6 layers,  2/2 seqs)",
        "llama_memory_recurrent: size =   38.53 MiB (     2 cells,  24 layers,  2 seqs)",
        "sched_reserve:      CUDA0 compute buffer size =    36.11 MiB",
    ])
    assert v3.parse_engine_memory("llamacpp", log) == {"kv_mib": 86.53, "weights_mib": 497.4, "compute_mib": 36.11}


def test_vllm_memory_lines_include_the_reported_need() -> None:
    log = ("Model loading took 17.60 GiB\nGPU KV cache size: 16,384 tokens\n"
           "Maximum concurrency for 2,048 tokens per request: 8.00x\n"
           "To serve at least one request (1.73 GiB KV cache is needed, which is larger than the available")
    parsed = v3.parse_engine_memory("vllm", log)
    assert parsed["weights_gib"] == 17.6
    assert parsed["kv_tokens"] == 16384
    assert parsed["max_concurrency"] == 8.0
    assert parsed["kv_needed_one_request_gib"] == 1.73


def test_judge_scores_require_every_label_in_range() -> None:
    text = 'noise {"A": {"relevance": 9, "coherence": 8, "appropriateness": 7, "overall": 8}, "B": {"relevance": 1, "coherence": 2, "appropriateness": 3, "overall": 1}} tail'
    scores = v3.parse_scores(text, ["A", "B"])
    assert scores["B"]["overall"] == 1
    with pytest.raises(KeyError):
        v3.parse_scores(text, ["A", "B", "C"])
    with pytest.raises(ValueError):
        v3.parse_scores('{"A": {"relevance": 11, "coherence": 8, "appropriateness": 7, "overall": 8}}', ["A"])


def test_judge_prompt_shows_history_and_every_label() -> None:
    row = {"text": "سلام", "history": [{"user": "قبلی", "assistant": "پاسخ قبلی"}]}
    prompt = v3.judge_prompt(row, [("A", "یک"), ("B", "دو")])
    assert "Earlier in the conversation" in prompt and "[A] یک" in prompt and "[B] دو" in prompt
    assert "labelled A–B" in prompt


def test_paired_gain_is_bootstrapped_on_common_rows() -> None:
    table = {
        "x": {str(i): {"overall": 8} for i in range(20)},
        "base": {str(i): {"overall": 6} for i in range(20)},
    }
    gain = v3.paired_gain(table, "x", "base", "overall", draws=200)
    assert gain["rows"] == 20 and gain["mean_gain"] == 2 and gain["ci95"] == [2, 2] and gain["win_rate"] == 1.0


def test_gbnf_grammar_matches_the_runtime_character_class() -> None:
    from thesis_s2s.runtime.streaming import PERSIAN_ONLY_REGEX

    assert PERSIAN_ONLY_REGEX.startswith("[\\u0600-\\u06FF\\u200c\\u200f ")
    assert v3.GBNF.startswith("root ::= [\\u0600-\\u06FF\\u200C\\u200F ")
    assert v3.GBNF.endswith("{1,400}")


def test_tiers_put_references_first_and_keep_each_models_best_fit() -> None:
    gib = v3.GIB
    arms = [
        {"arm": "ref", "model": "Qwen/Qwen3-4B", "family": "official_bf16", "engine": "vllm", "bytes": 8 * gib, "role": "reference"},
        {"arm": "m__official_bf16", "model": "Qwen/M", "family": "official_bf16", "engine": "vllm", "bytes": 30 * gib, "role": "candidate"},
        {"arm": "m__awq_int4", "model": "Qwen/M", "family": "awq_int4", "engine": "vllm", "bytes": 8 * gib, "role": "candidate"},
        {"arm": "m__gptq_int4", "model": "Qwen/M", "family": "gptq_int4", "engine": "vllm", "bytes": 8 * gib, "role": "candidate"},
        {"arm": "m__fp8", "model": "Qwen/M", "family": "fp8", "engine": "vllm", "bytes": 15 * gib, "role": "candidate"},
        {"arm": "m__gguf-q4_k_m", "model": "Qwen/M", "family": "gguf_ladder", "rung": "Q4_K_M", "engine": "llamacpp", "bytes": 5 * gib, "role": "candidate"},
        {"arm": "m__gguf-q2_k", "model": "Qwen/M", "family": "gguf_ladder", "rung": "Q2_K", "engine": "llamacpp", "bytes": 3 * gib, "role": "candidate"},
    ]
    v3.assign_tiers(arms)
    tiers = {a["arm"]: a["tier"] for a in arms}
    assert tiers["ref"] == "A1"
    assert tiers["m__fp8"] == "A3"  # highest precision that fits the 24 GB weight budget
    assert tiers["m__awq_int4"] == "A3"  # highest precision that fits the 12 GiB budget
    assert tiers["m__gguf-q4_k_m"] == "A2"
    assert tiers["m__gptq_int4"] == "D" and tiers["m__gguf-q2_k"] == "D"


def test_inventory_method_family_reads_the_quantization_config() -> None:
    assert inventory.method_family("org/Model-AWQ", [], None) == "awq_int4"
    assert inventory.method_family("org/Model-AWQ-8bit", [], None) == "awq_other"
    assert inventory.method_family("org/Model", [], {"quantization_config": {"quant_method": "modelopt_fp4"}}) == "nvfp4"
    assert inventory.method_family("org/Model-GGUF", ["gguf"], None) == "gguf"
    assert inventory.method_family("org/Model-bnb-4bit", [], None) == "bnb_4bit"


def test_inventory_gguf_rungs_do_not_confuse_k_quant_suffixes() -> None:
    files = [{"path": name, "size": 1} for name in ("m-Q2_K.gguf", "m-Q2_K_L.gguf", "m-UD-Q2_K_XL.gguf", "m-Q4_K_M-00001-of-00002.gguf", "m-Q4_K_M-00002-of-00002.gguf")]
    rungs = inventory.gguf_rungs(files, ("Q2_K", "Q4_K_M", "UD-Q2_K_XL"))
    assert rungs["Q2_K"]["files"] == ["m-Q2_K.gguf"]
    assert len(rungs["Q4_K_M"]["files"]) == 2
    assert rungs["UD-Q2_K_XL"]["files"] == ["m-UD-Q2_K_XL.gguf"]


def test_frozen_plan_and_inventory_are_consistent() -> None:
    plan = json.loads((ROOT / "results/eval/responder_v3_plan.json").read_text(encoding="utf-8"))
    assert plan["panels"]["dev_rows"] == 100 and plan["panels"]["test_rows"] == 252
    assert plan["panels"]["test_elderly_by_author"] == {"claude": 63, "gpt-6-astra": 63}
    inv = json.loads((ROOT / "results/eval/responder_v3_inventory.json").read_text(encoding="utf-8"))
    assert len(inv["models"]) == 14
    requests = json.loads((ROOT / "configs/responder_v3_elderly_requests.json").read_text(encoding="utf-8"))["requests"]
    assert len(requests) == 176 and {r["author"] for r in requests} == {"claude", "gpt-6-astra"}
