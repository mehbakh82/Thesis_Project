# Frozen cascade intelligibility-proxy protocol

Protocol frozen before producing `cascade_intelligibility_proxy.json`.

## Question and claim boundary

This evaluation asks only whether the already validated Piper waveforms preserve
their intended generated Persian reply text well enough for the existing NeMo
ASR service to recover it. It is an automatic content-controlled round-trip
proxy. It is not a human listening test and cannot establish pronunciation
quality, naturalness, semantic relevance, conversational usefulness, or
population-level intelligibility.

## Locked inputs

- Existing mechanics result:
  `results/eval/cascade_real_service_validation_panel.json`, SHA-256
  `ae515381db2b72e9ed0b8dcb491c0765990732ed8bb8c46383499b9d60f89ced`.
- Existing group-disjoint validation manifest:
  `data/processed/moshi_finetune/val.jsonl`, 131 rows, SHA-256
  `a2762495830c82ce12d3dc4cc8469e2110ae21a77ea91f9b70498a6812e187a7`.
- The same predeclared panel indices: `0, 16, 32, 48, 65, 81, 97, 113, 130`.
- Existing exact backends: NeMo Soroush HTTP ASR, cached
  `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, and Piper Mana model SHA-256
  `e390c0e74ba71fd97c49ba662ee0c6e1724b462ba2d4561698af4f564840f126`.
- The sealed Moshi final-test manifest is forbidden and must not be read.

## Execution and validity gates

For each of the nine rows, rerun the exact deterministic cascade, then send the
generated Piper waveform to the same real NeMo ASR service. The measurement is
valid only if:

1. all locked artifact hashes and panel indices match;
2. the parent mechanics report is passed and says the final test was not
   accessed;
3. regenerated input-transcript and reply-text hashes exactly match the parent
   report for all nine rows;
4. Qwen loads at the exact backend/revision without rule fallback, Piper is the
   exact required backend/model, and both user and reply ASR calls succeed;
5. exactly nine round-trip measurements exist; and
6. the evaluator records that the final test was not accessed.

A failed validity gate invalidates the measurement; it must not be silently
excluded or replaced.

## Metrics fixed before execution

Both reference and hypothesis use the project's deterministic Persian verbatim
normalization. Word tokens split on whitespace and treat ZWNJ as a boundary.
Character tokens are Unicode code points excluding whitespace and ZWNJ.

- Per-row word error rate (WER) and character error rate (CER).
- Macro mean, median, p95, and maximum across all nine rows.
- Micro WER/CER from summed edit counts and reference-token counts.
- Exact-match row counts and ASR-failure counts.

No quality threshold is introduced after observing the values. The result is a
descriptive automatic proxy with `N=9`. The same ASR family participates in the
system and the proxy, token errors are not independent observations, and one
synthetic voice/domain cannot support population claims. Therefore no
confidence interval or human-intelligibility pass/fail claim is reported.

## Privacy and output

The committed report contains content SHA-256 values, token counts, error
counts/rates, backend identities, artifact hashes, and limitations. It must not
contain input transcripts, generated reply text, round-trip ASR text, audio, or
host-specific paths.
