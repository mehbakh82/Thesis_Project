# Moshi v5 result: reduced semantic-codebook weight did not pass runtime gates

Status: finalized scientific negative on 2026-08-31. No v5 final-test row was
accessed, no adapter was selected, and no deployment claim is allowed.

## Controlled question

V5 changed one factor from v4: `first_codebook_weight_multiplier` was reduced
from 100.0 to 10.0. Rank-128 LoRA, the exact two trainable text embeddings,
frozen audio-codebook embeddings, data, source-session split, 100-second
context, effective batch shape, optimizer, seed, 500-step horizon, candidate
steps, complete-prompt panel, and every eligibility threshold were unchanged.

The frozen hypothesis was that v4's later autoregressive speech collapse was
caused by excessive pressure on the first semantic audio codebook. V5 tested
whether a tenfold reduction would preserve speech while the text embeddings
and LoRA learned reliable Persian responses.

## Execution integrity

- Launch commit: `8a8c18185c3e6586fe3ebc50a1b6df95ccc6300b`.
- Training service invocation: `4c3e1c27782b479397100c6b52b06461`.
- One-step probe: finite loss 3.597103, 24.057 GB peak, exact 676-tensor
  checkpoint (674 LoRA plus two text embeddings).
- Scientific run: 500/500 steps in 42m03s, 28.640 GB maximum logged peak, 50
  finite logged training losses, five exact 1,103,525,992-byte adapters present
  at certification time, and a clean service exit.
- Every checkpoint contained the same 674 LoRA tensors and only the two declared
  embeddings. Shapes, BF16 dtypes, values, saved configs, and hashes passed.
- The complete fixed validation scope was 202 chunks. The held-out final test
  was not used for loss, generation, selection, or diagnosis.

Canonical evidence is in:

- `results/hardware/moshi_v5_preflight.json`
- `results/hardware/moshi_h100_v5_profile_probe.json`
- `results/moshi_v5_validation_reevaluation.json`
- `results/moshi_v5_runtime_candidates.json`
- `results/moshi_v5_checkpoint_selection.json`
- `results/hardware/moshi_v5_validation_pipeline.json`
- `results/hardware/moshi_h100_v5_training.json`

## Results

| Step | Complete-scope total loss | Speech rows | Nonempty-text rows | Persian-gate rows | Fully passing rows |
|---:|---:|---:|---:|---:|---:|
| 100 | 2.082676 | 9/9 | 9/9 | 0/9 | 0/9 |
| 200 | 1.943459 | 7/9 | 7/9 | 0/9 | 0/9 |
| 300 | 1.889862 | 6/9 | 6/9 | 0/9 | 0/9 |
| 400 | 1.870719 | 5/9 | 5/9 | 1/9 | 1/9 |
| 500 | 1.867859 | 4/9 | 4/9 | 1/9 | 1/9 |

The step-400 passing row mixed an English greeting with malformed Persian-like
text. The step-500 passing row did the same. Other nonempty generations were
predominantly generic English greetings or responses. Speech/nonempty-text
coverage again degraded from 9/9 at step 100 to 4/9 at step 500 while
teacher-forced loss improved.

The predeclared rule required all nine rows to pass. Therefore eligible steps
were `[]`, the selected checkpoint was `null`, and the selector failed closed.
The validation pipeline and training certificate passed because they correctly
attest a negative experiment. Selected-adapter validation, objective final-test
evaluation, final runtime diagnostics, and a final-pipeline receipt do not
exist.

## Post-finalization retention

After certification and the null selection were committed, disk-pressure
cleanup retained the step-400 best-runtime-tied and step-500 minimum-loss
tensors. Steps 100–300 were non-promoted negative intermediates and were
removed only after their exact hashes, losses, runtime outputs, schemas,
configurations, and certificate were committed. Recreating those tensor
payloads requires rerunning the frozen v5 recipe; the scientific result does
not depend on their continued local presence. The chained verified receipt is
`results/hardware/storage_cleanup_20260831.json`.

## Interpretation and next constraint

The v5 hypothesis is not supported strongly enough for promotion. Reducing the
semantic-codebook multiplier from 100 to 10 did not prevent later speech
collapse and did not produce reliable Persian. Excessive first-codebook weight
is therefore not a sufficient explanation of the v4 failure.

No additional run should be justified by trying more checkpoints or adjusting
the frozen thresholds. Any subsequent experiment needs a new predeclared
hypothesis about language conditioning, target construction, tokenizer/model
language support, or the training objective, while reusing validation evidence
only and keeping the untouched final test sealed.
