# Questions for the supervisor

Send the project definition and current evidence status with these questions. Ask for the answers in writing so the thesis can cite the agreed interpretation.

## Scope and data

1. Does “collect and label 100–200 hours” allow existing, properly licensed natural Persian conversations, or must any portion be newly recorded by me?
2. The definition does not specify 8–15 hours of new recording. Is any minimum newly recorded duration actually required? If yes, what is the exact minimum and purpose?
3. Does the university have LDC access to CALLFRIEND Farsi (`LDC2014S01`) and MATERIAL Farsi-English (`LDC2024S13`), or funding to obtain them?
4. May restricted LDC audio remain internal while aggregate results and derived model artifacts are reported in the thesis?
5. May the existing university-held Persian interviews/podcasts be used for training after license review, ASR, diarization, and response-pair construction?
6. What proportion of automatically generated ASR/diarization/overlap labels is acceptable, and how many hours or rows must be manually verified?
7. May read speech and synthetic overlap/noise be used only as supplements, while the 100–200-hour requirement is counted from natural conversational material?

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
16. The inspected official LLaMA-Omni2 release has no complete training entrypoint and does not train on target speech units in its released forward path. Is rebuilding and validating a trainer within the BSc scope?
17. Which backbone/revision should be approved before GPU work: LLaMA-Omni2-0.5B, Mini-Omni2, Freeze-Omni, or another model with an actually released trainer and suitable license?
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

“I cannot create or retain new raw voice recordings, but I can implement no-WAV live processing and use licensed conversational corpora. May I use licensed/archive natural Persian conversation for the 100–200-hour corpus and run the 5–10-person study with feature-only or metrics-only retention? If live recruitment is impossible, which written scope amendment do you approve? Please also clarify whether the 500 ms gate means max, p95, or median, whether an RTX 4090 is acceptable, and whether a genuine direct S2S adaptation is mandatory despite the selected upstream release lacking a complete trainer.”
