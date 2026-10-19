# Minimal remote listening study — protocol

Status: frozen before the page was sent to any participant (student decision
in `docs/SUPERVISOR_DECISIONS.md`; no supervisor approval is claimed).

## Instrument

One self-contained, mobile-friendly Persian HTML page built by
`scripts/build_listening_study.py` (SHA-256 of the distributed page:
`08f44bcd164ef0972b7cb353a93ee2cc29556eb27544f4e2e6811cc5d9259e0d` after the amendment below; the first build was `ca6ece09…`). Nothing is
recorded from the participant and no name or contact detail is asked. The page
asks for consent, an age group (under 40 / 40–59 / 60+), first language, and
listening device, then:

1. **Naturalness (absolute category rating, 1–5)** of 23 clips in a fresh
   random order per participant:
   - 8 × `tts_mana` — the deployed voice, reading 8 assistant replies;
   - 8 × `tts_ganji_adabi` — the runner-up voice of TTS v2, reading the **same**
     8 replies (paired design);
   - 6 × `natural_human` — clean single-speaker human Persian (PSRB-public
     lecture/audiobook clips, 3 male incl. one senior, 3 female), the anchor;
   - 1 attention check (a clip asking for option 2).
   Clip identifiers are opaque hashes, so the condition cannot be read from the
   page.
2. **Three recorded interactions with the live system** (the formal-loopback
   stack, recorded by `scripts/record_live_demo.py` as actual system output, not
   re-synthesis), each rated for responsiveness, interruption handling (where
   applicable), and satisfaction (1–5), plus would-use (yes/no):
   - i1: a question and the reply;
   - i2: a request, and a backchannel («بله بله») during the reply — the
     assistant ducks and continues;
   - i3: a question, and a real interruption with a new question — the
     assistant stops and answers the new question.
   The user's voice is synthesized (male Piper voice, 0.85× tempo) and the page
   says so.

The participant sends back the short code shown at the end; codes are decoded by
`scripts/ingest_listening_study.py`.

**Disclosure about the demos.** i1 was recorded twice: the first recording's
punctuated user text paused long enough for the endpointer to split the question,
so unpunctuated user text was used. i2 needed three attempts: «اوهوم آره»
(transcribed «وم») and «آره آره» (transcribed «کار», its first phoneme clipped)
were both wrongly treated as interruptions; «بله بله» was correctly ignored and
is the recording used. The measured error rates are in the simulation and
loopback receipts, not in these demos.

## Participants

5–10 native Persian listeners, at least two aged 60 or older (the requirement of
the project definition). Participants with a failed attention check, without
consent, or with incomplete ratings are excluded and counted.

## Analysis (fixed now)

- Naturalness MOS per condition, mean of per-participant means, with a
  participant-bootstrap 95% interval; also per age group.
- The proposal's criterion **MOS ≥ 3.5** is evaluated on `tts_mana` (the
  deployed voice). `tts_ganji_adabi` versus `tts_mana` is reported as the paired
  per-participant difference; the human anchor shows the scale's ceiling for
  these listeners.
- Interaction ratings: mean per item and the share answering "would use".
- With at most ten participants, intervals are wide; the result is reported as
  a small-sample listening study, not as a powered comparison.

## Claim boundary

A small remote study with self-reported age, on headphones or phone speakers,
rating recorded clips and recorded interactions (not live use). It is the only
human evidence of the project.

## Amendment before distribution (2026-10-17)

No participant had received the page. Two changes:

1. **Voices.** The naturalness section follows the TTS v3 rule
   (`docs/TTS_CANDIDATE_COMPARISON_V3_PROTOCOL.md`): Mana (deployed) plus up to
   two other voices that are not significantly less intelligible and score at
   least Mana's DNSMOS. The same 8 texts are used for every voice.
2. **Reply preference (human tie-breaker for the responder judges).** The two
   automatic judges disagree about the deployed responder
   (`docs/RESPONDER_CANDIDATE_COMPARISON_V2_RESULT.md`). Eight elderly requests
   (seeded random choice among the 52 not used for the voice clips) are shown
   as text with two replies in random order — the deployed Qwen3-4B prompt-v2
   reply and the Qwen3.5-4B elderly-prompt reply — and the participant picks
   the better reply for an older person, or "no difference". Analysis: the
   per-participant share of picks for each responder, with a participant
   bootstrap 95% interval, and the tie share. It is reported as human evidence
   on the judge disagreement; it does not by itself change the deployed
   responder.

Outcome of the voice rule: ParsVoice-XTTS was significantly less intelligible
than Mana in TTS v3 and is not included; the lab VITS model could not be run
(missing front-end checkpoint). The study therefore keeps Mana, ganji_adabi and
the human anchor, and adds the reply-preference section.

Update (2026-10-18, before any participant received the page): once its
front-end checkpoint was available, the lab VITS model was evaluated as served
and after inference optimization (`docs/TTS_LAB_VITS_OPTIMIZATION_RESULT.md`).
Both versions are significantly less intelligible than Mana (two-listener CER
9.89% and 9.19% against 5.82%), so the voice rule excludes it and the study page
is unchanged.
