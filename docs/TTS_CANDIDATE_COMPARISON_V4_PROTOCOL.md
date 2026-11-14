# TTS candidate comparison v4: protocol

Status: frozen on 2026-11-14, before any v4 model was downloaded or any v4
audio was generated.

## Why a v4

On 2026-11-14 the user asked whether the deployed voice (Mana, Persian Piper)
is really the best available. A search on the same day found Persian voices
released after or missed by TTS v3 (`docs/TTS_CANDIDATE_COMPARISON_V3_PROTOCOL.md`):

1. **MOSS-TTS-Nano-Persian** (`nimaaaAI/MOSS-TTS-Nano-Persian`, 117M, Apache
   2.0, fine-tuned on Common Voice Persian). The earlier audit excluded it for
   lack of clean reference speech; the CC0 ManaTTS references used for
   ParsVoice-XTTS in v3 remove that reason.
2. **Gooya v1** (`Reza2kn/gooya-v1`, a LoRA fine-tune of a MOSS-TTS-Nano
   Persian model, licence "other", trained on Gemini-generated speech), which
   takes phonemes from its published G2P (`Reza2kn/negara-g2p-clean-v7`).
3. **Chatterbox Persian** (`Thomcles/Chatterbox-TTS-Persian-Farsi`, 500M,
   CC BY-NC 4.0, gated). It is evaluated through `Reza2kn/Gooya-Bozorg-v1.5`,
   which its card states is a byte-identical, ungated repackaging of upstream
   snapshot `4e9f6b70…`; the hashes in its `MODEL_MANIFEST.json` are recorded,
   but the gated upstream files are not compared.

**Outcome boundary fixed now.** Under the v3 deployment rule, a voice replaces
Mana only if listening-study participants rate it more natural **and** its
median first-four-words synthesis time is at most 100 ms. The listening study
was not conducted (`results/eval/listening_study_status.json`), so **no v4
result can change the deployed voice**. v4 answers a narrower question: is
Mana still the most intelligible tested voice, and would any new voice have
qualified for the listening study.

Licences: Gooya v1 ("other", Gemini-generated training data) and Chatterbox
Persian (non-commercial) are evaluated for research only, as ParsVoice-XTTS was
in v3.

## Texts, reference and synthesis

- **Texts:** the same 60 assistant replies as TTS v2 and v3
  (`tts_v2_texts.jsonl`).
- **Reference voice:** `mana_ref_0.wav` from the v3 reference set (CC0
  ManaTTS); models that accept several references also get `mana_ref_3.wav`,
  as XTTS did in v3.
- **Clause chunking:** every reply is split with the live server's
  `ClauseChunker` (default limits) and each fragment is synthesized
  separately, then concatenated without added silence. This is how the live
  system feeds its voice, and it keeps fragments within MOSS's documented
  ~5 s range. Mana in v2/v3 synthesized whole replies; that difference is
  reported.
- **Settings:** the authors' recommended settings, fixed seed 20261017.
  MOSS: voice-clone mode with the card's default decoding. Gooya v1:
  `audio_temperature` 0.6, `audio_top_p` 0.9, phonemes from the v7 G2P.
  Chatterbox: `exaggeration` 0.5, `cfg_weight` 0.5.
- **Pinned revisions:** MOSS `dcfd7f2b…`; Gooya v1 `9320f582…`; G2P and
  audio tokenizer at the revisions resolved on download (recorded); Gooya
  Bozorg v1.5 `8844b3a6…`.

Remote code that these models execute is read before it is run.

## Measures (unchanged from v3)

Round-trip intelligibility by two ASR listeners (NeMo v4 CTC and Seamless
v5ft), two-listener mean CER with a paired bootstrap interval against
`piper_mana`; DNSMOS P.835 OVRL (an automatic signal-quality proxy, not a
Persian naturalness MOS); median time to synthesize the first four words on
the H100; real-time factor; peak GPU memory. Scored by
`scripts/score_tts_candidates_v2.py` together with the v2 and v3 arms.

## Rules (fixed now)

1. **Would qualify for the listening study (v3 rule):** two-listener CER not
   significantly worse than Mana's (paired 95% interval includes zero or lies
   below it) and DNSMOS OVRL at least Mana's.
2. **Latency:** whether the median first-four-words time is at most 100 ms
   (the v3 deployment bound) is reported.
3. **Deployment:** none, because the listening study was not run. The thesis
   reports the outcome as an automatic screen.

If a model cannot be run as its card describes (missing component, runtime
failure), it is reported as not run with the reason; no debugging beyond the
documented usage.

## Claim boundary

Automatic listeners and an automatic quality proxy on 60 texts, one cloned
reference voice, H100 timing; no human rating.
