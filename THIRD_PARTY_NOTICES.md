# Third-party notices and boundaries

This project does not relicense third-party source code, model weights, datasets, or generated media. Their original terms remain controlling. `third_party/UPSTREAMS.lock.json` records inspected repository revisions; a pin is not permission to copy or redistribute.

## Model repositories inspected

- LLaMA-Omni2, revision `c8afa9061a9c2d2c1919f7293f5492d946869752`: no repository-level license file was observed at inspection time. Do not copy, redistribute, or build a release dependency on its code until terms are clarified.
- Mini-Omni2, revision `75f450ea4e7dc1e52a7fc0e4dcf45b4c3d45ab98`: repository code declares MIT; model-weight and dependency terms must still be checked.
- BayLing-Speech, revision `c63fd7221d1e6c2d22c85e781bc61e842835ed53`: repository code declares Apache-2.0; model/data/dependency terms remain separate.
- Freeze-Omni, revision `163a24880e533b2a07038fb8dcfe02dbbb8457e6`: custom license limited to academic, research, and educational use and excluding commercial/production use.

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
