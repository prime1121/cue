# Development

Use Rust 1.85+ and Python 3.11+. The base Rust tests and model-downloader tests
are offline. The Python client tests need no Pipecat installation; adapter
tests use Pipecat 1.12.0.

## Complete verification

With Rust 1.85.1 on PATH and a Python environment active, run:

```console
python -m pip install -r requirements-dev.txt
python -m pip install -e "./python[pipecat]"
python scripts/prepare_models.py
python tools/verify.py
```

The verifier locates the native ONNX Runtime from the Python installation or
`ORT_DYLIB_PATH`, builds the current Rust source, enables native model/bridge
checks, and runs the policy examples. Empty or skipped Python suites fail the
check. Reports stay in the ignored `output/verification` directory. CI runs
this command on Linux with Python 3.11 and Rust 1.85.1.

## Rust checks

```console
cargo fmt --all -- --check
cargo test --release --locked
cargo run --release --locked -- replay examples/policy-demo.jsonl
cargo run --release --locked -- compare examples/paired-demo.jsonl
```

The included `assets/frontend` fixtures contain deterministic synthetic audio
and frozen feature matrices. Do not replace them with microphone recordings.
Their generator and upstream reference are retained under `vendor/`.

## Python checks

```console
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e "./python[pipecat]"
.venv/bin/python -m unittest discover -s python/tests -v
.venv/bin/python -m unittest discover -s scripts -p "test_*.py" -v
```

On Windows, substitute `.venv\Scripts\python.exe` for `.venv/bin/python`.
For the base client only, install `./python` and select `test_client.py`.
The opt-in native client test is skipped unless its environment variables are
set, as described in the [Python guide](../python/README.md).

## Native model parity

```console
python scripts/prepare_models.py
python scripts/prepare_models.py --check
```

Set `ORT_DYLIB_PATH` to the native ONNX Runtime library and run:

```console
cargo test --release --locked --test frontend_parity -- --include-ignored
```

This checks model probabilities and Silero state/reset behavior against the
frozen reference in addition to frontend numerical parity. See
[vendor/PROVENANCE.md](../vendor/PROVENANCE.md) for model versions, source
licenses, and the original reference measurements.
