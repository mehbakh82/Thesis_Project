# Third-party notices and boundaries

This project does not relicense third-party source code, model weights, datasets, or generated media. Their original terms remain controlling. `third_party/UPSTREAMS.lock.json` records inspected repository revisions; a pin is not permission to copy or redistribute.

## Model repositories inspected

- LLaMA-Omni2, revision `c8afa9061a9c2d2c1919f7293f5492d946869752`: no repository-level license file was observed at inspection time. Do not copy, redistribute, or build a release dependency on its code until terms are clarified.
- Mini-Omni2, revision `75f450ea4e7dc1e52a7fc0e4dcf45b4c3d45ab98`: repository code declares MIT; model-weight and dependency terms must still be checked.
- BayLing-Speech, revision `c63fd7221d1e6c2d22c85e781bc61e842835ed53`: repository code declares Apache-2.0; model/data/dependency terms remain separate.
- Freeze-Omni, revision `163a24880e533b2a07038fb8dcfe02dbbb8457e6`: custom license limited to academic, research, and educational use and excluding commercial/production use.
- Moshi runtime, revision `061cc4c630d9e11722e08b7d02b1836ba58f30e8`: Python/client code is MIT and Rust code is Apache-2.0; the selected Moshika weights declare CC BY 4.0.
- Moshi-Finetune, revision `2acc879fe7c48f885a18f6cc9548bccb2674d87b`: trainer code is Apache-2.0; trained adapters remain subject to base-weight and dataset terms.
- Moshika-PyTorch-BF16 weights, revision `a49141e28b3d9c947cf9aa5314431e1b11cbd2f5`: CC BY 4.0; all three training-time blob hashes and sizes are pinned in `third_party/UPSTREAMS.lock.json`.
- Mana-Persian-Piper, revision `ad9dd8518bedf517bd7cbc9f63b8e5c844bf5bc0`: the model card declares MIT and Mana-TTS declares CC0; the pinned ONNX SHA-256 is `e390c0e74ba71fd97c49ba662ee0c6e1724b462ba2d4561698af4f564840f126`. It is used as a deterministic single-speaker assistant target and is not redistributed here.

No code from these repositories is vendored in this project.

## Models and services used at runtime

NeMo, Qwen, Piper, Hugging Face Transformers, PyTorch, scikit-learn, SciPy, NumPy, FastAPI, and their models/dependencies retain their own licenses. Before distribution, record the exact model card, weight revision, license, and checksum in the release snapshot.

The local NeMo/diarization services are separate codebases and are not covered by this project's eventual code license.

## Data

- Existing YouTube-derived media stays research-internal and is not released by this project.
- LDC corpora are proposals only and may be used only after institutional access and license confirmation.
- Participant audio/features/metrics follow the approved consent and retention mode. No public release is implied.
- Synthetic harmonic fixtures are test artifacts and not representations of natural Persian speech.

## Project license status

The repository author and supervisor must approve the project-code license before public release. The supervisor question sheet proposes Apache-2.0. Until that decision is recorded, no license is granted merely by access to this source tree.
