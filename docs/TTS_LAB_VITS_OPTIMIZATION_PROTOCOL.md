# Lab VITS inference optimization — protocol

Status: written before any tuning run. Scope: everything that can be improved
in the lab VITS model (`star.text_to_speech`) **without retraining**: the text
front-end, the inference settings, the sentence handling and the serving path.
The acoustic weights are not changed.

## Data

- **Development:** 339 assistant replies written by the other responder arms
  of responder v2 for the same requests (none is one of the 60 evaluation
  texts); 26 contain digits.
- **Evaluation:** the 60 TTS v2/v3 texts, used **once**, after the final
  configuration is frozen.
- **Normalization unit cases:** hand-written Persian sentences with times,
  years, dates, phone numbers, prices, percentages and dosages, checked against
  the expected spoken form (correctness tests, not a tuning target).

## Steps (each kept only if it helps on the development set)

1. **Serving speed (must not change the audio):** keep the espeak-ng process
   persistent instead of starting one per call (phoneme strings must be
   identical on all development texts); resample with soxr instead of
   `kaiser_best` (difference to `kaiser_best` must be inaudible: SNR ≥ 40 dB);
   run the acoustic model on the GPU; cache all front-end models.
2. **Number and time reading:** fix readings that are wrong or unnatural in
   Persian (e.g. years read as «یک هزار و …», clock times, phone numbers),
   verified by the unit cases.
3. **Inference settings:** `noise_scale` ∈ {0.667 (model default), 0.333, 0},
   stochastic-duration noise `noise_scale_dp` ∈ {1.0 (model default), 0.5, 0},
   `length_scale` ∈ {1.0, 1.1}. Selection by
   the lowest two-listener CER (NeMo v4 CTC + Seamless v5ft) on the
   development texts without digits; among settings within 0.5 percentage
   points of the best, the highest DNSMOS OVRL.
4. **Sentence handling:** the service's split (at «.» and «،», 10-word chunks,
   no pause) versus punctuation-aware splitting (also «؟ ! ؛ :», chunks of up to
   25 words) with a 120 ms pause between chunks; kept if development CER is not
   worse and DNSMOS is not lower.

## Evaluation and decision

The frozen configuration is synthesized once for the 60 evaluation texts and
scored like TTS v3 (two-listener CER with a paired interval against Mana,
DNSMOS, first-four-words latency on GPU and CPU). The TTS v3 rules then apply
unchanged: it enters the listening study if it is not significantly less
intelligible than Mana and its DNSMOS is at least Mana's; it can replace Mana
only if the study shows it more natural **and** its median first-four-words
time is at most 100 ms on the deployment hardware.

## Claim boundary

Automatic proxies; the acoustic model is as trained; naturalness is decided by
the listening study.

## Amendment 1 (2026-10-18, during the step 3 grid run, before step 4)

Findings from reading the lab front-end, added before they are run:

- **Step 1, serving:** the fork builds a new `ESpeak` object for every
  sentence, and each construction runs `espeak-ng --voices` and
  `espeak-ng --version` in subprocesses (about 14 ms); both answers are cached
  (same phonemes). The espeak-ng engine runs in-process through libespeak-ng
  instead of as a persistent process (same intent; phoneme strings verified
  identical on all development texts). soxr did not meet the 40 dB criterion
  (minimum 37.6 dB; the difference lies above 7 kHz), so `kaiser_best` is kept
  and is instead computed exactly, with resampy's own filter table expanded
  into a polyphase filter (criterion re-checked on the development texts).
- **Step 4b, Persian punctuation:** «؟ ؛ ،» are not in the fork's punctuation
  set, so espeak-ng ends a clause there and the fork joins the clause outputs
  without a space: «خوردید؟ بله» is phonemized as one word. Mapping them to
  «? ; ,» (which are in the model's symbol set) before the front-end is tested
  after step 4, on top of the step 4 winner, and kept under the step 4 rule
  (development CER not worse and DNSMOS OVRL not lower).
