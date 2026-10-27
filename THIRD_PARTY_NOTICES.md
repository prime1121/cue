# Third-party notices

The repository's first-party license does not replace upstream licenses.
Model weights are downloaded separately and are not included in this source
repository. See [vendor/PROVENANCE.md](vendor/PROVENANCE.md) for pinned sources,
SHA-256 checksums, input contracts, and reference validation details.

| Component | Source | Terms and retained notice |
| --- | --- | --- |
| Smart Turn v3.2 CPU | [Pipecat / Daily](https://huggingface.co/pipecat-ai/smart-turn-v3) | BSD-2-Clause; [LICENSE.pipecat](vendor/LICENSE.pipecat) |
| Silero VAD | [Silero Team](https://github.com/snakers4/silero-vad) | MIT; [LICENSE.silero](vendor/LICENSE.silero) |
| Reference Whisper frontend and Rust translation | [Pipecat](https://github.com/pipecat-ai/pipecat) | BSD-2-Clause; [LICENSE.pipecat](vendor/LICENSE.pipecat) |
| Portions of the reference frontend | [Hugging Face Transformers](https://github.com/huggingface/transformers) | Apache-2.0; [LICENSE.transformers](vendor/LICENSE.transformers) |

The unmodified reference frontend is kept as
`vendor/whisper_features_reference.py`, including its copyright notices.
`vendor/generate_frontend_fixtures.py` creates deterministic synthetic numerical
test inputs. Those inputs are not recordings or conversational quality data.

Rust and optional Python dependencies retain their own licenses. Their
versions are recorded in `Cargo.lock` and `python/pyproject.toml`; the voice
agent examples pin theirs in `examples/pipecat/requirements.txt` and
`examples/livekit/requirements.txt`. Native ONNX Runtime is installed
separately; retain its distribution's notices when redistributing it.
