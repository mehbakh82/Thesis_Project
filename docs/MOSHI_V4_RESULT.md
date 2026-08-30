# Moshi v4 result: selective text embeddings did not pass runtime gates

Status: finalized scientific negative on 2026-08-30. No v4 final-test row was
accessed, no adapter was selected, and no deployment claim is allowed.

## Controlled question

V4 changed one factor from v3: rank-128 LoRA remained enabled and the broad
upstream `ft_embed` switch remained false, but gradients were enabled for
exactly `text_emb.weight` and `depformer_text_emb.weight`. All 23 audio-codebook
embeddings stayed frozen. Data, split, context, batch/microbatch shape,
optimizer, seed, 500-step horizon, candidate steps, complete-prompt panel, and
all eligibility thresholds were unchanged.

The frozen hypothesis was that v2's limited Persian behavior came from text
embedding adaptation while its later speech collapse came from updating every
audio embedding. V4 tested whether retaining only the text part would preserve
speech and produce reliable Persian.

## Execution integrity

- Launch commit: `5db5a6e9b1542298392afbd7be46a20fd6080817`.
- Training service invocation: `979a468626ef4bc4b89e5630bc38d525`.
- One-step probe: finite loss 3.573359, 24.057 GB peak, exact 676-tensor
  checkpoint (674 LoRA plus two text embeddings).
- Scientific run: 500/500 steps in 39m53s, 28.640 GB maximum logged peak, 50
  finite logged training losses, five retained exact 1,103,525,992-byte
  adapters, and a clean service exit.
- Every checkpoint contained the same 674 LoRA tensors and only the two declared
  embeddings. Shapes, BF16 dtypes, values, saved configs, and hashes passed.
- The complete fixed validation scope was 202 chunks. The held-out final test
  was not used for loss, generation, selection, or diagnosis.

Canonical evidence is in:

- `results/hardware/moshi_v4_preflight.json`
- `results/hardware/moshi_h100_v4_profile_probe.json`
- `results/moshi_v4_validation_reevaluation.json`
- `results/moshi_v4_runtime_candidates.json`
- `results/moshi_v4_checkpoint_selection.json`
- `results/hardware/moshi_v4_validation_pipeline.json`
- `results/hardware/moshi_h100_v4_training.json`

## Results

| Step | Complete-scope total loss | Speech rows | Nonempty-text rows | Persian-gate rows | Fully passing rows |
|---:|---:|---:|---:|---:|---:|
| 100 | 2.082125 | 9/9 | 9/9 | 1/9 | 1/9 |
| 200 | 1.948315 | 7/9 | 7/9 | 0/9 | 0/9 |
| 300 | 1.899640 | 4/9 | 4/9 | 0/9 | 0/9 |
| 400 | 1.881632 | 5/9 | 5/9 | 1/9 | 1/9 |
| 500 | 1.879061 | 5/9 | 5/9 | 0/9 | 0/9 |

The step-100 passing row mixed an English greeting with malformed Persian-like
text at exactly the 0.5 script threshold. The step-400 passing row emitted only
`اینا �`. These isolated rows do not establish relevant, intelligible Persian
conversation. Most nonempty generations remained English; silence/nonempty-text
failures increased after step 100 even while teacher-forced loss improved.

The predeclared rule required all nine rows to pass. Therefore eligible steps
were `[]`, the selected checkpoint was `null`, and the selector failed closed.
The validation pipeline and training certificate passed because they correctly
attest a negative experiment. Selected-adapter validation, objective final-test
evaluation, final runtime diagnostics, and a final-pipeline receipt do not
exist.

## Interpretation and next constraint

The narrow hypothesis is not supported strongly enough for promotion. Freezing
audio embeddings preserved valid adapter structure and initially preserved
speech, but selective text-embedding tuning did not produce reliable Persian
and did not prevent later autoregressive degradation. Broad audio-embedding
drift is therefore not a sufficient explanation of v2/v3 failure.

No additional run should be justified by trying more checkpoints from the same
recipe. A defensible v5 would need a new predeclared hypothesis about the data
or training objective—such as language conditioning, target quality, or the
relative text/audio supervision—not a post-hoc threshold change. It must reuse
only validation evidence for design and keep the untouched final test sealed.
