# Contributing to Cue

Open an issue for a bug, integration problem, or proposed change. Include a
minimal reproduction, platform, Rust/Python versions, model hashes, and the
relevant configuration. Share audio only when you have permission; a synthetic
fixture is preferable for a small regression test.

## Development

Follow [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) to build and run the tests.
Keep a pull request focused on one change and explain the observed problem,
resulting behavior, and validation. Add a regression test for a behavior change.
Keep documentation and examples consistent with the code.

Use `cargo fmt` for Rust. Match existing Python formatting and type annotations.
Choose descriptive names and small functions. Comments should explain a
protocol constraint, numerical detail, or non-obvious decision rather than
repeat the code. Preserve upstream copyright headers and license notices.

The public compatibility surface includes the `sao-turn` CLI, `sao_turn`
library/Python imports, JSONL schema, and context/reset behavior. Call out
breaking changes explicitly. Do not silently change framing, default policy
settings, model revisions, or how timestamps are measured.

## Models and measurements

Model changes must update the pinned source, SHA-256, provenance, and relevant
tests. Do not commit weights, private recordings, provider credentials,
environment files, generated reports, or build output.

Separate policy regression tests from quality claims. For performance or
accuracy changes, provide the hardware, thread count, model versions, input
conditions, raw measurements, and the evaluation method. Follow
[docs/EVALUATION.md](docs/EVALUATION.md) when comparing endpoint behavior.

## Pull request checklist

- Run the checks relevant to the change and include their results.
- Update API or setup documentation when behavior changes.
- Keep fixtures small, deterministic, and redistributable.
- Retain third-party notices and document any new dependency or model.

By submitting a contribution, you agree to license it under this repository's
first-party [LICENSE](LICENSE). Existing third-party material remains under its
original terms.
