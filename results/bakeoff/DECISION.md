# Bake-off result

**Evidence status: component survey; no end-to-end model winner.**

This command did not fine-tune or compare the candidate speech-language models.

Encodec proxy: True. Mimi proxy: False. CosyVoice2 available: False.

Selected implementation path: Moshika 7B with the official Moshi-Finetune LoRA trainer. This is an engineering selection, not an empirical winner.

Current deployable baseline: NeMo ASR → local Qwen/rules → Piper. Promote the direct model only after controlled Persian end-to-end evaluation.
