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
    files = [{"path": name, "size": 1} for name in ("m-Q2_K.gguf", "m-Q2_K_L.gguf", "m-UD-Q2_K_XL.gguf", "m-Q4_K_M-00001-of-00002.gguf",
                                                     "m-Q4_K_M-00002-of-00002.gguf", "m-Q8_0.gguf", "MTP/mtp-m-Q8_0.gguf")]
    rungs = inventory.gguf_rungs(files, ("Q2_K", "Q4_K_M", "UD-Q2_K_XL", "Q8_0"))
    assert rungs["Q2_K"]["files"] == ["m-Q2_K.gguf"]
    assert rungs["Q8_0"]["files"] == ["m-Q8_0.gguf"]  # not the speculative-decoding drafter
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


def test_history_turns_are_cut_to_their_last_words() -> None:
    long_turn = " ".join(f"w{i}" for i in range(100))
    row = {"text": "سلام", "history": [{"user": long_turn, "assistant": "کوتاه"}]}
    messages = v3.messages_for(row, "system")
    assert messages[1]["content"].split() == [f"w{i}" for i in range(60, 100)]
    assert messages[2]["content"] == "کوتاه" and messages[-1]["content"] == "سلام"
    assert "w59" not in v3.judge_prompt(row, [("A", "x")])


def test_stages_screen_every_model_and_probe_sensitivity_on_two() -> None:
    gib = v3.GIB
    arms = [
        {"arm": "ref", "model": "Qwen/Qwen3-4B", "family": "official_bf16", "engine": "vllm", "bytes": 8 * gib, "role": "reference", "tier": "A1"},
        {"arm": "big__gguf-q4_k_m", "model": "Qwen/Qwen3.6-27B", "family": "gguf_ladder", "engine": "llamacpp", "bytes": 16 * gib, "role": "candidate", "tier": "A2"},
        {"arm": "big__awq_int4", "model": "Qwen/Qwen3.6-27B", "family": "awq_int4", "engine": "vllm", "bytes": 17 * gib, "role": "candidate", "tier": "A3"},
        {"arm": "qwen3.5-4b__gptq_int4", "model": "Qwen/Qwen3.5-4B", "family": "gptq_int4", "engine": "vllm", "bytes": 4 * gib, "role": "candidate", "tier": "B"},
        {"arm": "gemma-4-e4b__official_bf16", "model": "google/gemma-4-E4B-it", "family": "official_bf16", "engine": "vllm", "bytes": 15 * gib, "role": "candidate", "tier": "A3"},
        {"arm": "fp8", "model": "Qwen/Qwen3.8-27B", "family": "official_fp8", "engine": "remote", "bytes": 29 * gib, "role": "reference_precision", "tier": "-"},
        {"arm": "x__bnb_4bit", "model": "Qwen/Qwen3.6-27B", "family": "bnb_4bit", "engine": "vllm", "bytes": 17 * gib, "role": "excluded", "tier": "-"},
    ]
    v3.assign_stages(arms)
    stages = {a["arm"]: a["stage"] for a in arms}
    assert stages == {"ref": "S1", "big__gguf-q4_k_m": "S1", "big__awq_int4": "S3?", "qwen3.5-4b__gptq_int4": "S2",
                      "gemma-4-e4b__official_bf16": "S2", "fp8": "S1", "x__bnb_4bit": "-"}


def test_hub_snapshots_mount_the_whole_cache_repo(tmp_path: Path) -> None:
    snapshot = tmp_path / "hub/models--org--m/snapshots/abc123"
    assert v3.container_mount(snapshot) == (tmp_path / "hub/models--org--m", "/model/snapshots/abc123")
    assert v3.container_mount(tmp_path / "org__m") == (tmp_path / "org__m", "/model")


def test_failed_kv_profiles_are_not_used_for_deployment(tmp_path: Path) -> None:
    arm = {"model": "org/M", "engine": "vllm"}
    rows = [{"model": "org/M", "engine": "vllm", "kv": "auto", "error": {"2048": {"error": "engine exited during start"}}}]
    (tmp_path / "kvprofile.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert v3.deployment_kv_bytes(tmp_path, arm, "auto") is None
    rows.append({"model": "org/M", "engine": "vllm", "kv": "auto", "per_seq_const_bytes": 0, "per_seq_bytes_per_token": 1000})
    (tmp_path / "kvprofile.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert v3.deployment_kv_bytes(tmp_path, arm, "auto") == int(1.03 * 2 * 1000 * 2048) + 16 * v3.MIB


def test_judge_calls_show_at_most_twelve_replies_and_share_the_anchors() -> None:
    names = ["ref1", "ref2"] + [f"arm{i}" for i in range(18)]
    groups = v3.split_with_anchors(names, ["ref1", "ref2"])
    assert len(groups) == 2 and all(len(g) + 1 <= 12 for g in groups)  # + the planted control
    assert all(g[:2] == ["ref1", "ref2"] for g in groups)
    assert sorted({a for g in groups for a in g}) == sorted(names)
    assert groups[0][2:4] == ["arm0", "arm2"]  # the rest alternate, so both calls span every size
    assert v3.split_with_anchors(names[:11], ["ref1"]) == [names[:11]]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _judgments(scores_by_arm: dict[str, float], rows: list[str]) -> list[dict]:
    out = []
    for row in rows:
        labels = {chr(65 + i): [arm] for i, arm in enumerate(scores_by_arm)}
        scores = {chr(65 + i): dict.fromkeys(("relevance", "coherence", "appropriateness", "overall"), s)
                  for i, s in enumerate(scores_by_arm.values())}
        out.append({"row_id": row, "group": 0, "labels": labels, "scores": scores})
    return out


def test_anchor_scores_are_averaged_over_calls(tmp_path: Path) -> None:
    _write(tmp_path / "judgments_s_j2.jsonl", _judgments({"ref": 8, "a": 6}, ["r1"]) + _judgments({"ref": 6, "b": 7}, ["r1"]))
    table, _ = v3.arm_scores(tmp_path, "s", "j2")
    assert table["ref"]["r1"]["overall"] == 7 and table["a"]["r1"]["overall"] == 6


def test_contenders_follow_amendment_one(tmp_path: Path) -> None:
    gib = v3.GIB

    def arm(name: str, model: str, family: str, engine: str, size: float, rung: str | None = None, role: str = "candidate") -> dict:
        return {"arm": name, "model": model, "family": family, "engine": engine, "bytes": size * gib, "rung": rung, "role": role, "stage": "S1"}

    q4b, q9b, g31 = "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "google/gemma-4-31B-it"
    _write(tmp_path / "arms.jsonl", [
        arm("qwen3-4b-instruct-2507__deployed", "Qwen/Qwen3-4B-Instruct-2507", "official_bf16", "vllm", 8, role="reference"),
        arm("qwen3.5-4b__official_bf16", q4b, "official_bf16", "vllm", 8.7, role="reference"),
        arm("qwen3.5-4b__gguf-q4_k_m", q4b, "gguf_ladder", "llamacpp", 2.6, "Q4_K_M"),
        arm("qwen3.5-4b__gguf-q8_0", q4b, "gguf_ladder", "llamacpp", 4.5, "Q8_0"),
        arm("qwen3.5-4b__gguf-q3_k_m", q4b, "gguf_ladder", "llamacpp", 2.0, "Q3_K_M"),
        arm("qwen3.5-4b__awq_int4", q4b, "awq_int4", "vllm", 3.0),
        arm("qwen3.5-9b__gguf-q4_k_m", q9b, "gguf_ladder", "llamacpp", 5.3, "Q4_K_M"),
        arm("qwen3.5-9b__official_bf16", q9b, "official_bf16", "vllm", 18),
        arm("gemma-4-31b__gguf-q4_k_m", g31, "gguf_ladder", "llamacpp", 18, "Q4_K_M"),
        arm("gemma-4-31b__gguf-q2_k", g31, "gguf_ladder", "llamacpp", 11, "Q2_K"),
        arm("gemma-4-31b__official_qat_w4a16", g31, "official_qat_w4a16", "vllm", 19),
    ])
    _write(tmp_path / "panel_dev.jsonl", [{"row_id": "r1", "panel": "natural_turns"}, {"row_id": "r2", "panel": "elderly_requests", "author": "claude"}])
    _write(tmp_path / "judgments_stage1_j2.jsonl", _judgments({
        "qwen3-4b-instruct-2507__deployed": 9, "qwen3.5-4b__official_bf16": 8, "qwen3.5-4b__gguf-q4_k_m": 7.5,
        "qwen3.5-9b__gguf-q4_k_m": 7.0, "gemma-4-31b__gguf-q4_k_m": 7.8}, ["r1", "r2"]))
    _write(tmp_path / "judgments_variants_j2.jsonl", _judgments({
        "qwen3.5-4b__official_bf16": 8, "qwen3.5-4b__gguf-q8_0": 8, "qwen3.5-4b__awq_int4": 7.9,
        "qwen3.5-4b__gguf-q4_k_m": 7.8, "qwen3.5-4b__gguf-q3_k_m": 7.0}, ["r1", "r2"]))
    result = v3.contenders(tmp_path)
    assert result["delta_q"]["delta_q"] == 0.2 and result["delta_q"]["delta"] == 0.5
    assert result["q3_k_m_non_inferior"] is False and result["best_4bit_methods"]["Qwen"] == ["awq_int4", "q4_k_m"]
    small, large = result["profiles"]["12gib"]["models"], result["profiles"]["24gb"]["models"]
    assert "Qwen/Qwen3-4B-Instruct-2507" not in small  # the deployed responder is a reference, not a candidate model
    assert small[q4b]["contender"] and not small[q9b]["contender"] and not small[g31]["has_fitting_build"]
    assert small[q4b]["stage3_builds"] == ["qwen3.5-4b__official_bf16", "qwen3.5-4b__gguf-q8_0", "qwen3.5-4b__awq_int4"]
    assert large[g31]["contender"] and large[g31]["stage3_builds"] == ["gemma-4-31b__official_qat_w4a16", "gemma-4-31b__gguf-q4_k_m"]
    assert result["untested"]["gemma-4-31b__gguf-q2_k"].startswith("3 bits or fewer")
    assert result["untested"]["qwen3.5-9b__official_bf16"].startswith("model outside")


def test_judging_stops_at_a_usage_limit_and_keeps_what_was_judged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = [{"call_id": f"c{i}", "row_id": f"r{i}", "group": 0, "prompt": "p", "labels": {"A": ["x"]}} for i in range(6)]
    monkeypatch.setattr(v3, "build_calls", lambda *args: calls)
    made: list[int] = []

    def fake_j2(prompt: str, scratch: Path, effort: str = "medium") -> tuple[str, float]:
        made.append(1)
        if len(made) > 2:
            raise RuntimeError("ERROR: You've hit your usage limit. Try again at 2:34 PM.")
        return '{"A": {"relevance": 5, "coherence": 5, "appropriateness": 5, "overall": 5}}', 0.0

    monkeypatch.setattr(v3, "call_j2", fake_j2)
    result = v3.judge(tmp_path, "j2", "s", [], {}, "dev", workers=1)
    assert result["judged"] == 2 and result["failures"] == 0 and "usage limit" in result["stopped"]
    assert len(made) == 3  # no retries after the limit


def test_drafter_files_are_never_served() -> None:
    arm = {"files": ["MTP/mtp-gemma-4-E4B-it-Q8_0.gguf", "gemma-4-E4B-it-Q8_0.gguf"]}
    assert v3.model_files(arm) == ["gemma-4-E4B-it-Q8_0.gguf"]


def test_a_file_that_fails_verification_is_fetched_again_without_the_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    good = b"weights"
    files = [{"type": "file", "path": "m-Q5_K_M.gguf", "lfs": {"oid": hashlib.sha256(good).hexdigest()}}]
    monkeypatch.setattr(v3, "_api", lambda path: files)
    monkeypatch.setattr(v3, "_hf_headers", lambda: {})
    passes: list[bool] = []

    def fake_download(arm: dict, dest: Path, todo: list, header: dict, use_mirror: bool) -> None:
        passes.append(use_mirror)
        (dest / "m-Q5_K_M.gguf").write_bytes(b"mixed" if use_mirror else good)

    monkeypatch.setattr(v3, "_download", fake_download)
    result = v3.fetch(tmp_path, {"arm": "x", "repo": "org/m", "files": ["m-Q5_K_M.gguf"], "engine": "llamacpp"})
    assert passes == [True, False] and result["mirror_retry"] is True


def test_rate_limits_back_off_without_using_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = [{"call_id": "c0", "row_id": "r0", "group": 0, "prompt": "p", "labels": {"A": ["x"]}}]
    monkeypatch.setattr(v3, "build_calls", lambda *args: calls)
    waits: list[float] = []
    monkeypatch.setattr(v3.time, "sleep", waits.append)
    answers = iter([RuntimeError("ERROR: exceeded retry limit, last status: 429 Too Many Requests")] * 4
                   + ['{"A": {"relevance": 5, "coherence": 5, "appropriateness": 5, "overall": 5}}'])

    def fake_j2(prompt: str, scratch: Path, effort: str = "medium") -> tuple[str, float]:
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer, 0.0

    monkeypatch.setattr(v3, "call_j2", fake_j2)
    result = v3.judge(tmp_path, "j2", "s", [], {}, "dev", workers=1)
    assert result["judged"] == 1 and result["failures"] == 0 and result["stopped"] is None
    assert waits == [60, 120, 240, 480]


def test_rungs_of_one_repository_download_to_separate_directories(tmp_path: Path) -> None:
    q4 = {"arm": "m__gguf-q4_k_m", "repo": "org/M-GGUF", "local": None}
    q5 = {"arm": "m__gguf-q5_k_m", "repo": "org/M-GGUF", "local": None}
    assert v3.model_dir(tmp_path, q4) != v3.model_dir(tmp_path, q5)
    assert v3.model_dir(tmp_path, q4) != v3.model_dir(tmp_path, {**q4, "repo": "other/M-GGUF"})  # a fallback repository


def test_usage_limit_messages_give_the_wait_until_the_stated_time() -> None:
    from datetime import datetime

    message = "ERROR: You've hit your usage limit. ... or try again at 5:19 PM.\n"
    assert v3.limit_wait_seconds(message, datetime(2026, 11, 2, 17, 3, 37)) == 15 * 60 + 23 + 180
    assert v3.limit_wait_seconds("try again at 12:05 AM", datetime(2026, 11, 2, 23, 0)) == 65 * 60 + 180  # after midnight
    assert v3.limit_wait_seconds("rate limited 13 times in a row", datetime(2026, 11, 2, 12, 0)) == 1800


def test_judging_resumes_after_a_usage_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = iter([{"calls": 4, "judged": 1, "stopped": "try again at 5:19 PM", "failures": 0},
                    {"calls": 4, "judged": 4, "stopped": None, "failures": 0}])
    monkeypatch.setattr(v3, "judge", lambda *args: next(results))
    waits: list[float] = []
    monkeypatch.setattr(v3.time, "sleep", waits.append)
    assert v3.judge_until_done(tmp_path, "j2", "s", [], {}, "dev", 2)["judged"] == 4
    assert len(waits) == 1 and waits[0] > 0


def test_a_lapsed_codex_sign_in_stops_the_run_instead_of_failing_every_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = [{"call_id": f"c{i}", "row_id": f"r{i}", "group": 0, "prompt": "p", "labels": {"A": ["x"]}} for i in range(3)]
    monkeypatch.setattr(v3, "build_calls", lambda *args: calls)
    made: list[int] = []

    def fake_j2(prompt: str, scratch: Path, effort: str = "medium") -> tuple[str, float]:
        made.append(1)
        raise RuntimeError("codex rc=1: ERROR: workspace routing discovery unauthorized (401)")

    monkeypatch.setattr(v3, "call_j2", fake_j2)
    result = v3.judge(tmp_path, "j2", "s", [], {}, "dev", workers=1)
    assert result["failures"] == 0 and "401" in result["stopped"] and len(made) == 1
    assert v3.limit_wait_seconds(result["stopped"], __import__("datetime").datetime(2026, 11, 3, 12, 0)) == 1800


def test_finalists_prefer_configurations_that_meet_the_latency_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    def lat(ttft: float, decode: float) -> dict:
        return {"single": {"first_token_s": {"median_ms": ttft}}, "decode_tok_s_median": decode}

    fit = {
        "m__gguf-q5_k_m": {"deployment_fit": {"12gib": None, "24gb": "f16"}, "latency": lat(300, 60)},
        "m__awq_int4": {"deployment_fit": {"12gib": "fp8", "24gb": "auto"}, "latency": lat(60, 90)},
    }
    contenders = {"profiles": {p: {"models": {"org/M": {"contender": True}, "org/N": {"contender": False}}} for p in ("12gib", "24gb")}}
    variants = {"m__gguf-q5_k_m": {"means": {"overall": 7.2}}, "m__awq_int4": {"means": {"overall": 7.0}}}
    engines = {"m__gguf-q5_k_m": "llamacpp", "m__awq_int4": "vllm"}
    monkeypatch.setattr(v3, "fits", lambda work: fit)
    monkeypatch.setattr(v3, "contenders", lambda work: contenders)
    monkeypatch.setattr(v3, "screen", lambda work, stage: {"controls": {}, "models": {"org/M": {"variants": variants}}})
    monkeypatch.setattr(v3, "arm_by_id", lambda work, arm: {"engine": engines[arm]})
    kv_rows = {"m__awq_int4@fp8": {str(i): {"overall": 7} for i in range(10)}, "m__awq_int4@auto": {str(i): {"overall": 7} for i in range(10)}}

    monkeypatch.setattr(v3, "arm_scores", lambda work, stage, judge: ({}, {}))
    result = v3.finalists(Path("."))
    assert result["per_profile"]["24gb"]["org/M"]["arm"] == "m__awq_int4"  # the faster build, although it scores 0.2 lower
    assert "org/M" not in result["per_profile"]["12gib"] and result["pending_kv_checks"] == ["m__awq_int4@fp8"]

    monkeypatch.setattr(v3, "arm_scores", lambda work, stage, judge: (kv_rows, {}))
    result = v3.finalists(Path("."))
    assert result["per_profile"]["12gib"]["org/M"]["kv"] == "fp8" and result["pending_kv_checks"] == []


def test_kv_variants_are_named_with_an_at_sign(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v3, "arm_by_id", lambda work, arm: {"engine": "llamacpp" if "gguf" in arm else "vllm"})
    assert v3.kv_label(Path("."), "m__awq_int4@fp8") == "fp8"
    assert v3.kv_label(Path("."), "m__gguf-q4_k_m") == "f16" and v3.kv_label(Path("."), "m__awq_int4") == "auto"


def test_kv_profiles_ignore_memory_lines_from_engines_that_crashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class CrashingServer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("engine exited during start")

        def stop(self) -> str:  # vLLM logs its KV size before the attention backend asserts
            return "GPU KV cache size: 16,384 tokens\nMaximum concurrency for 2,048 tokens per request: 8.00x\nAssertionError: fp8 tensor core"

    monkeypatch.setattr(v3, "Server", CrashingServer)
    rows = v3.kvprofile(tmp_path, tmp_path, {"arm": "g__qat", "model": "google/g", "engine": "vllm"})
    assert all("error" in r and "per_seq_const_bytes" not in r for r in rows)


def test_judging_refuses_a_missing_reply_set(tmp_path: Path) -> None:
    (tmp_path / "replies").mkdir()
    (tmp_path / "panel_dev.jsonl").write_text(json.dumps({"row_id": "r1", "text": "سلام", "history": []}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no reply"):
        v3.build_calls(tmp_path, "dev", [["a@fp8", "a@auto"]], {"a@fp8": "fp8", "a@auto": "auto"})


def test_later_runs_use_the_repository_that_served_the_arm(tmp_path: Path) -> None:
    _write(tmp_path / "arms.jsonl", [{"arm": "g__nvfp4", "repo": "first/G-NVFP4", "fallbacks": ["second/G-NVFP4"], "local": None}])
    _write(tmp_path / "attempts.jsonl", [{"arm": "g__nvfp4", "repos_tried": ["first/G-NVFP4", "second/G-NVFP4"], "status": "ok"}])
    arm = v3.serving_arm(tmp_path, "g__nvfp4")
    assert arm["repo"] == "second/G-NVFP4" and arm["fallbacks"] == []


def _lat(ttft: float, decode: float) -> dict:
    return {"single": {"first_token_s": {"median_ms": ttft}}, "decode_tok_s_median": decode}


def test_decision_latency_follows_amendment_three(tmp_path: Path) -> None:
    _write(tmp_path / "arms.jsonl", [{"arm": "g__gguf-q5_k_m", "engine": "llamacpp"}])
    dev = {"arm": "g__gguf-q5_k_m", "kv": "q4_0", "tag": "kvquality", "status": "ok", "latency": _lat(120, 60)}
    test = {"arm": "g__gguf-q5_k_m", "kv": "q4_0", "tag": "test", "status": "ok", "latency": _lat(300, 25)}
    _write(tmp_path / "runs.jsonl", [dev, test])
    lat = v3.decision_latency(tmp_path, "g__gguf-q5_k_m", "q4_0")
    assert lat["source"] == "test run" and lat["third_needed"] is True  # 300 ms vs 120 ms: more than a factor of two
    _write(tmp_path / "runs.jsonl", [dev, test, {**test, "tag": "latency3", "latency": _lat(130, 58)}])
    lat = v3.decision_latency(tmp_path, "g__gguf-q5_k_m", "q4_0")
    assert lat["source"] == "median of three runs" and lat["ttft_ms"] == 130 and lat["decode_tok_s"] == 58 and not lat["third_needed"]


def test_the_decision_considers_only_each_profiles_finalists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fin = {"per_profile": {"12gib": {"org/S": {"arm": "s__q5", "kv": "f16"}}, "24gb": {"org/B": {"arm": "b__awq", "kv": "auto"}}}}
    (tmp_path / "finalists.json").write_text(json.dumps(fin), encoding="utf-8")
    engines = {"s__q5": "llamacpp", "b__awq": "vllm", "ref": "vllm", "ref2": "vllm"}
    monkeypatch.setattr(v3, "arm_by_id", lambda work, arm: {"engine": engines[arm]})
    rows = [str(i) for i in range(30)]

    def table(score: dict[str, float]) -> dict:
        return {arm: {r: dict.fromkeys(v3.SCORE_KEYS, s) for r in rows} for arm, s in score.items()}

    scores = {"s__q5": 8.0, "b__awq": 7.0, "ref": 5.0, "ref2": 3.0}
    monkeypatch.setattr(v3, "arm_scores", lambda work, stage, judge: (table(scores), {}))
    monkeypatch.setattr(v3, "fits", lambda work: {"s__q5": {"deployment_fit": {"12gib": "f16", "24gb": "f16"}},
                                                  "b__awq": {"deployment_fit": {"12gib": None, "24gb": "auto"}}})
    monkeypatch.setattr(v3, "subset_rows", lambda work, panel: {"all": rows})
    lats = {"s__q5": _lat(200, 100), "b__awq": _lat(100, 90)}
    monkeypatch.setattr(v3, "decision_latency", lambda work, arm, kv: {"ttft_ms": lats[arm]["single"]["first_token_s"]["median_ms"],
                                                                        "decode_tok_s": lats[arm]["decode_tok_s_median"], "third_needed": False})
    result = v3.decide(tmp_path, references=("ref", "ref2"))
    assert result["profiles"]["12gib"]["winner"] is None  # its only finalist misses the latency gate
    assert result["profiles"]["24gb"]["winner"] == "b__awq"  # s__q5 scores higher but is not a 24 GB finalist
    assert result["profiles"]["24gb"]["versus"]["ref"]["beats"] is True


def test_an_exploratory_tuning_replaces_or_adds_vllm_flags() -> None:
    base = ["--max-num-batched-tokens", "2048", "--language-model-only"]
    assert v3.apply_tuning(base, "") == base
    assert v3.apply_tuning(base, "b512") == ["--max-num-batched-tokens", "512", "--language-model-only"]
    assert v3.apply_tuning(base, "eager_b512") == ["--max-num-batched-tokens", "512", "--language-model-only", "--enforce-eager"]
    assert v3.apply_tuning(base + ["--enforce-eager"], "eager").count("--enforce-eager") == 1


def test_a_tuned_run_keeps_its_kv_and_stores_replies_under_the_tuned_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeServer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            seen.update(kwargs)
            self.peak_mib = 10_000

        def start(self) -> None:
            pass

        def stop(self) -> str:
            return ""

    monkeypatch.setattr(v3, "Server", FakeServer)
    monkeypatch.setattr(v3, "deployment_kv_bytes", lambda work, arm, kv: 300 * v3.MIB)
    monkeypatch.setattr(v3, "latency", lambda server, panel, system: {})
    monkeypatch.setattr(v3, "generate_panel", lambda server, panel, system: [{"row_id": "r1", "reply": "سلام", "latin": False}])
    monkeypatch.setattr(v3, "system_for", lambda arm: "")
    _write(tmp_path / "panel_test.jsonl", [{"row_id": "r1"}])
    _write(tmp_path / "panel_dev.jsonl", [{"row_id": "r1"}])
    arm = {"arm": "g__qat", "model": "google/g", "engine": "vllm"}
    record = v3.run_arm(tmp_path, tmp_path, arm, kv="auto+eager", panel_name="test", tag="explore")
    assert seen["kv"] == "auto" and seen["tuning"] == "eager"
    assert record["kv"] == "auto" and record["tuning"] == "eager" and record["status"] == "ok"
    assert (tmp_path / "replies" / "test__g__qat__auto+eager.jsonl").exists()
    with pytest.raises(ValueError, match="tuning"):
        v3.run_arm(tmp_path, tmp_path, {**arm, "engine": "llamacpp"}, kv="f16+eager")


def test_the_queue_runs_each_tuning_of_one_arm_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path / "arms.jsonl", [{"arm": "g__qat", "model": "google/g", "engine": "vllm", "repo": "google/g-qat", "fallbacks": [],
                                      "bytes": 1, "local": "hub/g", "stage": "S1"}])
    _write(tmp_path / "kvprofile.jsonl", [{"model": "google/g", "engine": "vllm", "kv": "auto", "via_arm": "g__qat", "per_seq_const_bytes": 1}])
    _write(tmp_path / "runs.jsonl", [{"arm": "g__qat", "tag": "explore", "kv": "auto", "tuning": "b512", "status": "ok"}])
    (tmp_path / "list.txt").write_text("g__qat@auto+b512\ng__qat@auto+eager\ng__qat@auto+eager_b512\n", encoding="utf-8")
    ran: list[str | None] = []

    def fake_run(work: Path, models: Path, arm: dict, *, kv: str | None = None, panel_name: str = "dev", tag: str = "run",
                 generate: bool = True) -> dict:
        ran.append(kv)
        return {"arm": arm["arm"], "kv": kv, "status": "ok"}

    monkeypatch.setattr(v3, "fetch", lambda models, arm: {"arm": arm["arm"], "local": True})
    monkeypatch.setattr(v3, "guarded_run", fake_run)
    v3.queue(tmp_path, tmp_path, None, None, arm_list=tmp_path / "list.txt", panel_name="test", tag="explore")
    assert ran == ["auto+eager", "auto+eager_b512"]


def test_exploration_chooses_the_first_tuning_that_fits_and_meets_the_gate(tmp_path: Path) -> None:
    def run(tuning: str, peak: int, ttft: float) -> dict:
        return {"arm": "g__qat", "tag": "explore", "kv": "auto", "tuning": tuning, "status": "ok", "peak_mib": peak,
                "latency": {"single": {"first_token_s": {"median_ms": ttft}}, "decode_tok_s_median": 60.0}}

    _write(tmp_path / "runs.jsonl", [run("b512", 10_700, 50.0), run("eager", 10_500, 160.0), run("eager_b512", 10_400, 60.0)])
    result = v3.explore(tmp_path, arm="g__qat", compared=())
    assert result["chosen"] == "eager_b512" and not result["tunings"]["b512"]["fits_12gib"]
    assert not result["tunings"]["eager"]["meets_gate"] and "means" not in result


def test_the_receipt_keeps_hashes_of_private_files_not_their_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v3, "fits", lambda work: {})
    monkeypatch.setattr(v3, "contenders", lambda work: {"profiles": {}})
    (tmp_path / "replies").mkdir()
    _write(tmp_path / "replies" / "test__a__auto.jsonl", [{"row_id": "r1", "reply": "متن خصوصی"}])
    (tmp_path / "decide.json").write_text(json.dumps({"profiles": {"24gb": {"winner": "a"}}}), encoding="utf-8")
    result = v3.receipt(tmp_path)
    assert result["decision"]["profiles"]["24gb"]["winner"] == "a" and result["exploratory"] is None
    assert list(result["data_sha256"]) == ["replies/test__a__auto.jsonl"] and "متن" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("keep", [False, True])
def test_the_queue_deletes_a_downloaded_model_unless_asked_to_keep_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep: bool) -> None:
    arm = {"arm": "g__nvfp4", "model": "google/g", "engine": "vllm", "repo": "org/g-nvfp4", "fallbacks": [], "bytes": 1,
           "local": None, "stage": "S1"}
    _write(tmp_path / "arms.jsonl", [arm])
    _write(tmp_path / "kvprofile.jsonl", [{"model": "google/g", "engine": "vllm", "kv": "auto", "via_arm": "x", "per_seq_const_bytes": 1}])
    (tmp_path / "list.txt").write_text("g__nvfp4\n", encoding="utf-8")
    weights = v3.model_dir(tmp_path / "models", arm) / "model.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    monkeypatch.setattr(v3, "fetch", lambda models, arm: {"arm": arm["arm"]})
    monkeypatch.setattr(v3, "guarded_run", lambda work, models, arm, **kw: {"arm": arm["arm"], "kv": "auto", "status": "ok"})
    v3.queue(tmp_path, tmp_path / "models", None, None, arm_list=tmp_path / "list.txt", keep=keep)
    assert weights.exists() is keep


def test_pmon_lines_split_into_this_engine_and_other_processes() -> None:
    text = "# gpu pid type sm mem\n# Idx # C/G % %\n    0  111  C  40  10 - - - - x\n    0  222  C  -  - - - - - y\n    0  333  C  25  5 - - - - z\n"
    assert v3.parse_pmon(text, {333}) == (40, 25)


def test_spearman_handles_ties_and_order() -> None:
    assert v3.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert v3.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert v3.spearman([1, 2], [1, 2]) is None


def test_more_judge_reasoning_is_compared_on_identical_calls(tmp_path: Path) -> None:
    def row(call_id: str, a: int, b: int, control: int) -> dict:
        return {"call_id": call_id, "row_id": call_id, "labels": {"A": ["x"], "B": ["y"], "C": ["__control__"]},
                "scores": {k: {"relevance": s, "coherence": s, "appropriateness": s, "overall": s} for k, s in (("A", a), ("B", b), ("C", control))}}

    _write(tmp_path / "judgments_test_j2.jsonl", [row("c1", 8, 5, 1), row("c2", 7, 6, 1), row("c3", 9, 2, 1)])
    _write(tmp_path / "judgments_effort_j2high.jsonl", [row("c1", 8, 4, 7), row("c2", 6, 6, 0)])
    result = v3.effort(tmp_path, pairs=(("j2", "j2high"),))["j2high"]
    assert result["calls"] == 2 and result["variant_control_failure_rate"] == 0.5
    assert result["overall_means"]["x"] == {"base": 7.5, "variant": 7.0} and result["ranking_variant"] == ["x", "y"]
    assert result["reply_scores"]["exact"] == 0.5 and result["reply_scores"]["within_one"] == 1.0


def test_an_identical_prompt_in_a_new_group_reuses_the_judged_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "replies").mkdir()
    _write(tmp_path / "panel_dev.jsonl", [{"row_id": "r1", "text": "سلام", "history": []}, {"row_id": "r2", "text": "خوبی", "history": []}])
    _write(tmp_path / "replies" / "dev__a__auto.jsonl", [{"row_id": "r1", "reply": "یک"}, {"row_id": "r2", "reply": "دو"}])
    _write(tmp_path / "replies" / "dev__b__auto.jsonl", [{"row_id": "r1", "reply": "یک"}, {"row_id": "r2", "reply": "دو"}])
    calls = {"n": 0}

    def fake_j2(prompt: str, scratch: Path, effort: str = "medium") -> tuple[str, float]:
        calls["n"] += 1
        return json.dumps({k: {"relevance": 7, "coherence": 7, "appropriateness": 7, "overall": 7} for k in "ABC"}), 0.0

    monkeypatch.setattr(v3, "call_j2", fake_j2)
    v3.judge(tmp_path, "j2", "s", [["a"]], {"a": "auto"}, "dev", 1)
    assert calls["n"] == 2
    result = v3.judge(tmp_path, "j2", "s", [["a", "b"]], {"a": "auto", "b": "auto"}, "dev", 1)
    # b repeats a word for word, so every prompt is one already judged: no new call, the scores are reused.
    assert calls["n"] == 2 and result["reused_identical_prompts"] == 2 and result["judged"] == 2
    table, controls = v3.arm_scores(tmp_path, "s", "j2")
    assert set(table["b"]) == {"r1", "r2"} and controls["calls"] == 2
    assert v3.judge(tmp_path, "j2", "s", [["a", "b"]], {"a": "auto", "b": "auto"}, "dev", 1)["reused_identical_prompts"] == 0
