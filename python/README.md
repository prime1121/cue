# Cue Python bridge

An asyncio client and optional Pipecat adapter for the persistent Rust
`sao-turn stream` worker. The Python package does not bundle the executable,
models, or ONNX Runtime. Build and prepare them using the [product README](../README.md).

Install Python 3.11+, then from this `python` folder:

```console
python -m pip install -e .
```

For Pipecat support and the complete bridge test suite:

```console
python -m pip install -e ".[pipecat]"
python -m unittest discover -s tests -v
```

The Pipecat adapter is pinned/tested against Pipecat 1.12.0. Base client tests
can run without it using `python -m unittest discover -s tests -p test_client.py -v`.
Normal tests simulate native subprocess pipes; one native test is opt-in.

[examples/pipecat](../examples/pipecat) runs `SaoTurnProcessor` in a complete
voice bot, and [examples/livekit](../examples/livekit) drives `SaoTurnClient`
from a LiveKit agent. Both turn Cue's decisions into the framework's user turns.

`StartFrame` is forwarded after native readiness. Original audio passes through
unchanged and each native 32 ms observation produces `TurnObservationFrame`.
The adapter does not resample, manufacture silence, emit
`UserStoppedSpeakingFrame`, or directly control playback. Your application
consumes `end`, `wait`, `backchannel`, and `interrupt` decisions.

Each client owns one persistent subprocess with one request in flight,
deadlines, bounded stderr, and bounded JSON responses. Timeout, cancellation,
and protocol failure close/reap it to avoid response misalignment. Standalone
clients require an explicit `start()` after failure. A failed Pipecat processor
becomes unusable and reports an upstream error.

`TurnContextFrame(context=...)` replaces context. Supply `TurnContext` or a
mapping containing `agent_speaking`, `mode` (`conversation` or `digits`),
`expected_digits`, `text`, and `text_final`. Text is limited to 65536 UTF-8 bytes;
expected digits must be 1–128. Unknown fields and invalid types are rejected.
`TurnResetFrame()` clears native audio, VAD, clock and context. An `end` or
`backchannel` clears only transcript context, preserving application mode and
clock. Explicit configuration can be passed using `SaoTurnClient(config=path)`;
`TurnConfig.to_dict()` gives the native defaults.

Transcription frames are best-effort complete replacements, not accumulated
provider-specific segments. Interim frames set `text_final=False`, regular
frames set it to `True`, and bot-speaking frames update context in either
direction. For strict ordering, serialize context and audio at the application
boundary: Pipecat system/data queues have different priorities.

The adapter does not associate delayed STT results with an utterance generation.
A late final transcript can populate the next turn after `end`. Applications
using asynchronous STT must discard stale results before submitting context.
The native policy's text rules are experimental English heuristics; its
acoustic models are pretrained Smart Turn and Silero. There is no established
accuracy advantage or automatic multilingual semantic understanding.

To run the native integration test, set `SAO_TURN_RUN_NATIVE=1`,
`SAO_TURN_NATIVE_EXE`, `SAO_TURN_MODEL`, `SAO_TURN_VAD`, and optionally
`SAO_TURN_RUNTIME` to absolute paths, then run the tests above.

First-party source is under the repository's [MIT License](../LICENSE).
Upstream notices are retained under [vendor/PROVENANCE.md](../vendor/PROVENANCE.md).
