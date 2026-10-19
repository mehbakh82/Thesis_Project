# TTS candidate comparison v3 — protocol

Status: written before any v3 audio was synthesized. Purpose: test the two
Persian TTS models that became available after TTS v2 — ParsVoice-XTTS (a
public XTTS-v2 fine-tune on 2,200 h of Persian, reported naturalness MOS 3.6)
and the lab's own VITS model (`star.text_to_speech`) — against the deployed
voice, and decide which voices the listening study compares.

## Arms

| Arm | Model | Runs on |
|---|---|---|
| `piper_mana` (deployed) | Piper VITS, Mana voice | CPU |
| `piper_ganji_adabi` | Piper VITS, best DNSMOS voice of TTS v2 | CPU |
| `parsvoice_xtts` | ParsVoice-XTTS, zero-shot clone of the ManaTTS speaker (two CC0 ManaTTS recordings as the reference, 7.8 s and 11.1 s) | GPU |
| `star_vits` | lab VITS model (`star.text_to_speech`, Coqui VITS, 22.05 kHz) | CPU |

Cloning the ManaTTS speaker gives XTTS the same voice identity as Piper Mana
(Piper's Mana voice is trained on ManaTTS), so the comparison isolates the
synthesis method. ManaTTS is released under CC0. XTTS decodes with a fixed seed.
ParsVoice-XTTS is under the non-commercial Coqui Public Model License and is
evaluated for research only.

## Texts and measures

The same 60 assistant replies as TTS v2. Per arm: round-trip intelligibility
by two ASR listeners (NeMo v4 CTC and Seamless v5ft), two-listener mean CER and
a paired bootstrap interval against `piper_mana`; DNSMOS P.835 (an automatic
signal-quality proxy, not a Persian naturalness MOS); time to synthesize the
first four words (the live server's first clause) and real-time factor on the
arm's device; GPU memory where used.

## Rules (fixed now)

**Listening study.** The study keeps `piper_mana` (the deployed voice, on which
the proposal's MOS ≥ 3.5 criterion is evaluated) and adds up to two other
voices. A candidate qualifies if its two-listener CER is not significantly worse
than Mana's (the paired 95% interval of the difference includes zero or lies
below it) and its DNSMOS OVRL is at least Mana's; if more than two qualify, the
two with the highest DNSMOS are used.

**Deployment.** A candidate replaces Mana only if, in the listening study, its
naturalness MOS exceeds Mana's with a participant-bootstrap 95% interval of the
paired difference above zero, **and** its median first-four-words synthesis time
is at most 100 ms on the deployment hardware (Mana: 84 ms), so that the measured
end-to-end budget is not broken. Otherwise the result is reported as a
naturalness/latency trade-off.

## Claim boundary

Automatic proxies on 60 texts; naturalness is decided only by the listening
study.
