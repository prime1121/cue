# Streaming and policy API

The `sao-turn` executable and `sao_turn` library names are retained for
compatibility. Cue is the project name.

## Worker protocol

Start `sao-turn stream`, then send one JSON object per line on stdin. Each line
receives one JSON response on stdout. Use absolute executable and model paths
when the application's working directory is elsewhere.

```json
{"samples":[0,15,-20,0],"context":{"mode":"digits","expected_digits":10,"agent_speaking":false},"reset":false}
```

Samples are signed PCM16 integers, mono 16 kHz. Arbitrary chunks are buffered
into 512 samples / 32 ms. Small packets can produce no frames. A request is
limited to ten seconds of audio. Supply continuous input including silence;
the worker does not manufacture silence. Responses contain `frames` and
`buffered_samples`, including VAD/completion observations and optional decisions.

Context persists when omitted. Providing context replaces the entire context;
it is not a patch. Supply only transcript text available at that point in time.

```json
{"samples":[],"context":{"mode":"digits","expected_digits":10,"text":"nine eight seven","text_final":false,"agent_speaking":false}}
```

Context applies at the next completed audio frame. Reset with
`{"samples":[],"reset":true}` to clear audio, VAD, buffers, clock, and context.
Clear transcript text at turn boundaries. A backend failure requires a reset.
Each detector belongs to one stream; run synchronous inference on a worker
when using asynchronous transports.

## Decisions and defaults

Decisions are `end`, `wait`, `backchannel`, and `interrupt`. The application
decides how to act; the worker does not stop playback or start replies.

The default policy requires 200 ms of detected speech, checks completion after
200 ms silence with a 0.5 threshold, and forces an end at eight seconds silence.
Use `--config configs/live-conversation.json` for a three-second fallback.
Forced endpoints carry `forced:true`. These are tuning choices, not proven
optimal thresholds. Text rules and digit-word parsing are English heuristics;
they do not provide phone validation or multilingual semantic understanding.

`sao-turn score input.wav` scores completion over the trailing eight seconds.
It does not detect speech: silence can receive a high completion score. Use
`wav`, `stream`, or `StreamDetector` for VAD-gated decisions.

## Rust library

```rust,no_run
use sao_turn::{
    inference::{SileroVad, SmartTurnModel},
    policy::{Config, Context},
    stream::{NeuralBackend, StreamDetector},
};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let backend = NeuralBackend {
        turn: SmartTurnModel::load(Path::new("assets/models/smart-turn-v3.2-cpu.onnx"), None, 1)?,
        vad: SileroVad::load(Path::new("assets/models/silero-vad.onnx"), None, 1)?,
    };
    let mut detector = StreamDetector::new(backend, Config::default())?;
    for frame in detector.push(&[0.0_f32; 512], &Context::default())? {
        if let Some(event) = frame.decision {
            println!("{event:?}");
        }
    }
    Ok(())
}
```

## Benchmark manifest

`sao-turn benchmark manifest.jsonl --results paired.jsonl` processes real WAVs
through both policies. Each manifest line names a segment, its reference
endpoint, and optional timestamped context updates. Audio paths are relative
to the manifest.

```json
{"id":"call07-turn03","audio":"call07-turn03.wav","reference_end_ms":1200,"context":{"mode":"digits","expected_digits":10},"events":[{"at_ms":900,"context":{"mode":"digits","expected_digits":10,"text":"9876543210","text_final":true}}]}
```

`reference_end_ms:null` means the segment is known incomplete, not unannotated.
Initial transcripts are rejected to avoid future context. Both systems use
the same acoustic inference, so the comparison isolates policy changes.
Audio-clock endpoint delay excludes compute, STT, network, LLM, and TTS
latency. See [EVALUATION.md](EVALUATION.md) for measurement details.
