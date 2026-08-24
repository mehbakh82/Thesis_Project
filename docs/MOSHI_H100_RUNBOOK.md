# Moshi Persian adaptation on the H100

This is the genuine direct/full-duplex training path. It replaces the earlier
LLaMA-Omni2 reconstruction ablation, whose Qwen model was not in the forward
path and whose target was the input waveform.

Training and official evaluation use different hardware contracts:

- **H100:** diarization, response-pair preparation, LoRA training, development
  inference, and ablations;
- **physical 12–24 GB GPU:** final fit and browser-observed latency evidence.

An H100 memory cap is useful as a fit smoke test but cannot reproduce the
compute throughput or memory bandwidth of the target card.

## What this training is—and is not

The earlier NeMo fine-tune and this run solve different tasks:

| Run | Input | Supervision | Learned output |
|---|---|---|---|
| Earlier NeMo ASR | one Persian speech clip | its reference transcript | Persian text |
| Required Moshi adaptation | a timed two-channel dialogue | the next assistant response text and Mimi speech tokens | the next spoken Persian response, including turn timing |

No additional NeMo ASR fine-tune is required here. NeMo remains the cascade
baseline and an optional transcript-quality screen. The genuine training run is
a LoRA adaptation of the pinned Moshika 7B causal dialogue model with the
official Moshi-Finetune trainer. It learns assistant text/audio-token losses
from adjacent user-response pairs; it is not speech-to-transcript training.

Moshi does not create a special category of permission absent from ASR. The
same source-use question applied to the earlier NeMo run. This repository
requires an explicit internal-training authorization because no reusable open
license was independently verified; it never infers permission from public
availability or data possession. The recorded supervisor decision satisfies
that internal thesis-training gate but does not assert redistribution rights.

## Why Moshi

Kyutai publishes an Apache-2.0 LoRA trainer for Moshi. The trainer consumes
stereo conversations, jointly predicts text and Mimi assistant-audio tokens,
and saves adapters loadable by the official streaming server. The Moshika base
weights are CC BY 4.0. This is materially different from ASR:

```text
channel 1: Persian user speech  ─┐
                                 ├─> Moshi causal dialogue model
channel 0: prior assistant audio ┘      └─> next Persian assistant audio/text
```

The English SentencePiece model represents Persian through UTF-8 byte tokens
without unknown tokens, but inefficiently. `ft_embed: true` is therefore part
of the Persian LoRA profile and Persian held-out intelligibility is mandatory.

Moshi's published instruction-tuning design keeps the system voice consistent
while varying user voices. The primary export follows that design: it preserves
natural YouTube user audio, response timing, and the approved reference response
text, but renders the assistant channel deterministically with the pinned
single-speaker Mana-Persian-Piper voice. Original podcast response audio is a
multi-voice ablation, not the primary run. This also avoids treating a changing
set of podcast guests as one assistant identity.

## Pinned upstreams

Use the revisions in `third_party/UPSTREAMS.lock.json`. Do not silently train
against moving branches.

```bash
mkdir -p third_party/checkouts
git clone https://github.com/kyutai-labs/moshi.git \
  third_party/checkouts/moshi
git -C third_party/checkouts/moshi checkout \
  061cc4c630d9e11722e08b7d02b1836ba58f30e8
git clone https://github.com/kyutai-labs/moshi-finetune.git \
  third_party/checkouts/moshi-finetune
git -C third_party/checkouts/moshi-finetune checkout \
  2acc879fe7c48f885a18f6cc9548bccb2674d87b
```

Install into an isolated environment because the official trainer uses its own
Torch/Triton stack. The checked-in environment lock records every installed
version. Both local upstream packages are installed with `--no-deps`, which
prevents the trainer's moving Git dependency from being resolved.

```bash
python3 -m venv .venv-moshi
export PIP_CACHE_DIR="$PWD/hf_cache/pip"
export HF_HOME="$PWD/hf_cache"
.venv-moshi/bin/python -m pip install -r requirements-moshi.lock
.venv-moshi/bin/python -m pip install --no-deps \
  -e third_party/checkouts/moshi/moshi \
  -e third_party/checkouts/moshi-finetune
.venv-moshi/bin/python scripts/validate_moshi_environment.py
.venv-moshi/bin/python scripts/prepare_moshi_base.py
```

The last command downloads only the exact Hugging Face revision in the
upstream lock and verifies the byte length and SHA-256 of every training-time
blob. `configs/moshi_h100.yaml` then uses those local files and the tracked
legacy architecture config; the trainer cannot silently resolve a later
version of the model from `main`. Recheck offline at any time with
`--verify-only`.

## Prepare final training data

Do not train unless either the strict human-QA path or the validated documented
waiver path passes. Automatic processing is complete: the candidate-capped
primary + reserve plan estimates 6,017 pairs / 105.727 pair h, and all 1,021
staging windows are internally authorized. The original strict data command
sequence starts with the reviewer:

```bash
# A reviewer completes the generated 40-row window sheet and 24-row
# conversation_interruption_qa.csv (168.3 seconds of short excerpts).
.venv/bin/python -m thesis_s2s.cli apply-conversation-qa \
  --in-jsonl data/processed/manifests/conversation_episode_windows_noise_labeled_combined_authorized.jsonl \
  --out-jsonl data/processed/manifests/conversation_episode_windows_reviewed.jsonl
.venv/bin/python -m thesis_s2s.cli apply-interruption-qa \
  --in-jsonl data/processed/manifests/conversation_episode_windows_reviewed.jsonl \
  --out-jsonl data/processed/manifests/conversation_episode_windows_interactions_reviewed.jsonl
.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes \
  --manifest data/processed/manifests/conversation_episode_windows_interactions_reviewed.jsonl \
  --min-hours 100 --max-hours 240
.venv/bin/python -m thesis_s2s.cli build-conversations \
  --in-jsonl data/processed/manifests/conversation_episode_windows_interactions_reviewed.jsonl
.venv/bin/python -m thesis_s2s.cli audit-conversations
.venv/bin/python -m thesis_s2s.cli export-moshi-data \
  --assistant-audio-mode piper
```

For the current time-constrained limited run, leave the blank QA sheets intact
and use the student waiver instead of running either `apply-*-qa` command. This
is not represented as supervisor approval of the waiver.

```bash
.venv/bin/python -m thesis_s2s.cli build-conversations \
  --in-jsonl data/processed/manifests/conversation_episode_windows_noise_labeled_combined_authorized.jsonl \
  --qa-waiver configs/conversation_qa_waiver.yaml
.venv/bin/python -m thesis_s2s.cli audit-conversations \
  --qa-waiver configs/conversation_qa_waiver.yaml
.venv/bin/python -m thesis_s2s.cli export-moshi-data \
  --assistant-audio-mode piper \
  --qa-waiver configs/conversation_qa_waiver.yaml
```

Only `training_ready_under_qa_waiver=true` permits this limited run;
`final_training_ready`, strict coverage, and human-verification claims remain
false. The complete policy and thesis disclosure are in `docs/QA_WAIVER.md`.

The builder excludes every explicitly reviewed failure. `export-moshi-data`
writes channel 0 as the deterministic Persian assistant response and channel 1
as the natural user turn, preserves the aligned response timing, creates the
official adjacent transcript JSON, keeps session-level splits, rejects
unauthorized rows, and reports whether the 100–200 h and manual-QA gates pass.
It pins and hashes the assistant voice.

Caption-aligned turns remain conservative: 4,787 pairs are still
`overlap_unattributed`. The retained raw speaker boundaries now recover 712
stricter review candidates (669 interruption-like and 43 backchannel-like) using
a 50% speaker-turn match, a 0.5 s caption-boundary tolerance, and at least 0.2 s
of cross-speaker overlap. The 24-row, four-channel listening sheet is only 168.3
seconds in total. Automatic candidates never pass the interruption gate. A
complete passing row is attached to the source window, and the builder then
uses its reviewed raw boundaries so clip/stereo timing preserves the overlap.
If the sample has unacceptable precision, use an approved supplement or narrow
the claim; never promote the remaining candidates automatically.

These files remain internal and gitignored. Use `--assistant-audio-mode source`
only for the disclosed original multi-voice ablation.

## Infrastructure smoke test

Exercise the complete official data/model/loss path with one deliberately
non-scientific, project-generated harmonic stereo example. No unreviewed source
media enters this test. The smoke profile disables checkpoints and embedding
fine-tuning and uses only five seconds to minimize shared-H100 pressure:

```bash
.venv/bin/python scripts/prepare_moshi_smoke_fixture.py
MOSHI_DISTRIBUTED_BACKEND=gloo \
  .venv-moshi/bin/torchrun --standalone --nproc-per-node 1 \
  scripts/moshi_train_entry.py configs/moshi_h100_smoke.yaml
.venv/bin/python scripts/record_moshi_smoke_result.py
```

The fixture's `REPORT.json` records `scientific_evidence: false`. A successful
smoke proves wiring only. It is not evidence of Persian model
quality, convergence, corpus coverage, or target-GPU performance.

On 2026-08-23 the shared H100 later became idle with 30,431 MiB free, so the
one-step official smoke was run without stopping any unrelated service. It
completed in 21 seconds with loss 4.614331 and 15.258 GB peak allocated memory.
`results/hardware/moshi_h100_smoke.json` validates and hashes the fixture,
resolved arguments, metrics, source config, and model-environment report. This
proves the real pinned model/Mimi/data/loss/backward/optimizer wiring, not
Persian quality or convergence.

The pinned Torch 2.6 environment bundles NCCL 2.21.5. On this host, both the
trainer and an isolated one-rank probe reproducibly SIGSEGV inside `libnccl`
during communicator initialization; the kernel log confirms it occurs before
model loading. A one-rank Gloo CUDA all-reduce probe passed. The project-owned
`scripts/moshi_train_entry.py` therefore supports an explicit
`MOSHI_DISTRIBUTED_BACKEND=gloo` fallback **only for world size 1**; it leaves
the pinned checkout unchanged and fails if that fallback is requested for
multiple ranks. Multi-GPU work still requires a healthy NCCL installation.

## Train

The official one-H100 example peaks near 39.6 GB at 100 seconds × batch 16.
The project profile uses 20 seconds × batch 1, gradient checkpointing, and four
microbatches so it can coexist more safely on a shared H100. Never terminate
unrelated GPU jobs to make room; wait for headroom or reduce `duration_sec` for
a smoke test.

After the final reviewed Moshi export exists, first run one step with the exact
full-training shape. Unlike the earlier five-second rank-8 smoke, this probe
uses 20 seconds, rank 64, embedding tuning, and four microbatches. It records a
memory certificate but is still explicitly non-scientific:

```bash
MOSHI_DISTRIBUTED_BACKEND=gloo \
  .venv-moshi/bin/torchrun --standalone --nproc-per-node 1 \
  scripts/moshi_train_entry.py configs/moshi_h100_profile_probe.yaml
.venv/bin/python scripts/record_moshi_profile_probe.py
.venv/bin/python -m thesis_s2s.cli gpu-preflight \
  --out results/hardware/current_preflight.json
```

Launch only when `adaptation_run_ready`, `adaptation_launch_safe_now`, and the
profile probe's hash checks all pass. The launch gate requires the measured
peak plus 4 GB of currently free headroom. This scheduling check is transient;
it does not make the H100 a target-hardware result.

```bash
MOSHI_DISTRIBUTED_BACKEND=gloo \
  .venv-moshi/bin/torchrun --standalone --nproc-per-node 1 \
  scripts/moshi_train_entry.py configs/moshi_h100.yaml
```

The pinned upstream trainer writes periodic adapters but does not persist the
optimizer, scheduler, data-loader, and `TrainState` needed for exact resume.
Run the full job in a persistent terminal/service, keep `overwrite_run_dir: false`, and do not describe an interrupted restart as a resume. If a run fails,
archive its run directory and restart from the immutable base/config; never
delete the only adapter evidence.

A final run is acceptable only when:

- the Moshi export report passes every gate;
- training and validation losses are finite and periodic adapters are runtime-loadable;
- changing assistant speech targets changes the audio-token loss;
- the adapter produces intelligible Persian on held-out sessions;
- the checkpoint loads in the official Moshi server without Piper substitution.

## Development inference and target-GPU evaluation

Build the browser client from the same pinned Moshi checkout. `npm ci` uses its
committed lockfile; do not omit `--static`, because the server otherwise
retrieves a separate moving web bundle.

```bash
cd third_party/checkouts/moshi/client
npm ci
npm run build
cd ../../../..
.venv-moshi/bin/python -m moshi.server \
  --hf-repo kyutai/moshika-pytorch-bf16 \
  --moshi-weight hf_cache/pinned/moshika-pytorch-bf16-a49141e/model.safetensors \
  --mimi-weight hf_cache/pinned/moshika-pytorch-bf16-a49141e/tokenizer-e351c8d8-checkpoint125.safetensors \
  --tokenizer hf_cache/pinned/moshika-pytorch-bf16-a49141e/tokenizer_spm_32k_3.model \
  --lora-weight checkpoints/moshi_fa/checkpoints/checkpoint_008000/consolidated/lora.safetensors \
  --config-path checkpoints/moshi_fa/checkpoints/checkpoint_008000/consolidated/config.json \
  --static third_party/checkouts/moshi/client/dist
```

The explicit local model paths prevent the runtime from resolving its default
Moshiko repository or a moving Moshika branch. First validate on the H100. Then
copy the immutable adapter and base-model
hashes to the 4090 and repeat live browser timing and memory measurement there.
The H100 result is scientific training/development evidence; only the physical
12–24 GB run proves the target-hardware requirement.
