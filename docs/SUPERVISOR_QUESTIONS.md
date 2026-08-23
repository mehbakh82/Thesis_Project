# Questions for the supervisor

Send the project definition and current evidence status with these questions. Ask for the answers in writing so the thesis can cite the agreed interpretation.

[نسخه فارسی](SUPERVISOR_QUESTIONS_FA.md)

## Scope and data

1. Does “collect and label 100–200 hours” allow existing, properly licensed natural Persian conversations, or must any portion be newly recorded by me?
2. The definition does not specify 8–15 hours of new recording. Is any minimum newly recorded duration actually required? If yes, what is the exact minimum and purpose?
3. Does the university have LDC access to CALLFRIEND Farsi (`LDC2014S01`) and MATERIAL Farsi-English (`LDC2024S13`), or funding to obtain them?
4. May restricted LDC audio remain internal while aggregate results and derived model artifacts are reported in the thesis?
5. **Resolved 2026-08-23:** the supervisor approved private thesis training on the student-crawled public YouTube interview/podcast archive; this does not approve redistribution.
6. What proportion of automatically generated ASR/diarization/overlap labels is acceptable, and how many hours or rows must be manually verified?
7. May read speech and synthetic overlap/noise be used only as supplements, while the 100–200-hour requirement is counted from natural conversational material?

### Existing YouTube archive: resolved and remaining decisions

- **Resolved:** supervisor-approved private thesis training, automatic annotation, aggregate reporting, and internal derived checkpoints cover the selected and reserve channels. Raw audio/captions and credentials remain non-redistributable.
- The measured final plan has **196.546 candidate h / 296 episodes**, **181.824 automatic multi-speaker h**, and **6,017 estimated response pairs / 105.727 pair h**. Which hour measure is binding for the 100–200-hour definition: selected candidate speech, verified multi-speaker windows, aligned staging context, or exported user/response spans?
- The generated 40-row listening sample covers every channel × automatic pass/reject stratum. Is five windows per stratum sufficient, and what rejection/error threshold is acceptable?
- The existing caption CSVs are the response text; NeMo is only a baseline/quality screen. Is that acceptable after the stratified listening sample passes?
- The conservative caption labels leave 4,787 pairs as `overlap_unattributed`, but retained raw speaker boundaries recover **712 strict automatic candidates** (669 interruption-like / 43 backchannel-like). A generated 24-row sheet samples six per channel and contains only **168.3 seconds** of excerpt audio. Is that pair-level listening sample sufficient, what minimum precision/acceptance threshold should apply, and may passing reviewed archive pairs satisfy interruption supervision without new recording?
- May the supervisor, a lab member, or another approved annotator complete the archive listening QA without the student recording new speech?

## Human study without stored voice

8. May I run the required 5–10-person study with raw audio processed live but never stored, retaining only timing, stop events, ratings, age bin, and pseudonymous IDs?
9. May I retain lossy aggregate energy/F0/MFCC features for speaker-held-out detector evaluation, or must the mode be metrics-only?
10. Does feature-only or metrics-only processing require ethics-board approval or a revised consent form at our institution?
11. Can the lab/supervisor help recruit or host 5–10 Persian speakers, including at least two aged 60+, for short live sessions?
12. Are supervised remote browser sessions acceptable if no raw audio leaves the live processing machine?
13. If recruitment is impossible, will you approve a written scope change replacing the interaction study with a listening study, expert review, or an explicitly incomplete protocol/future-work section?
14. If a listening study is accepted, which outcomes may it support? It cannot directly prove live barge-in or elderly turn-taking behavior.

## Model requirement

15. Must the final system include a genuinely adapted direct speech-to-speech language model, or is an honest low-latency ASR→LLM→TTS full-duplex cascade plus a documented direct-model feasibility experiment acceptable?
16. The implemented engineering path is Moshika 7B with the official Apache-2.0 Moshi-Finetune LoRA trainer, which supervises response text and assistant Mimi audio tokens. Is this direct full-duplex training objective acceptable for the thesis?
17. May the final run use the exact pinned Moshika/Moshi/Moshi-Finetune revisions recorded in `third_party/UPSTREAMS.lock.json`, train on the H100, and reserve the physical RTX 4090 for final fit and latency evaluation?
18. What minimum evidence makes the model “adapted”: Persian speech conditioning, Persian speech-token generation, runtime-loadable checkpoint, held-out intelligibility, and no external TTS substitution?

## Metrics and hardware

19. Is a physical RTX 4090 (24 GB) acceptable for the required 12–24 GB evaluation?
20. The Persian definition says “maximum response time ≤500 ms,” while the current engineering protocol uses median first-audio ≤500 ms. Should the binding gate be every turn/max, p95, or p50? May I report all three and use the agreed one as primary?
21. Is the >80% barge-in requirement event-level interrupt-vs-other accuracy, or frame-level classification accuracy?
22. Must detector evaluation be speaker/session-held-out, and should the thesis report 95% confidence intervals for accuracy, F1, false-accept rate, and false-reject rate?
23. Is `T_barge_in` measured from acoustic onset to confirmed client playback stop, and what p95 threshold should be used?

## Release and project management

24. May project code be released under Apache-2.0 while datasets, model weights, and third-party components retain their own terms or remain excluded?
25. Which artifacts must be submitted: code snapshot, exact upstream commits, environment hashes, model checksums, restricted-data provenance, evaluation JSON, and demo video?
26. What is the final deadline and priority order if genuine S2S adaptation, the 100–200-hour conversational corpus, and the human study compete for time?

## Short message to send

“The internal archive-training approval is recorded. The final plan has 196.546 candidate hours and 105.727 estimated response-pair hours. Raw speaker boundaries recover 712 automatic interaction candidates, and the generated four-channel pair review is only 24 rows / 168.3 seconds. Please confirm which hour measure is binding, whether the 40-row window QA and this pair-level QA are sufficient, and the required candidate-precision threshold. I also need decisions on feature/metrics-only human sessions, the 500 ms statistic, RTX 4090 target evaluation, and the pinned Moshika + official Moshi-Finetune path.”
