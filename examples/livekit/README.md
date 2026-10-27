# LiveKit example

A [LiveKit Agents](https://docs.livekit.io/agents/) 1.8.9 voice agent whose user
turns come from Cue instead of LiveKit's turn detector. The session runs with
manual turn detection, and Cue decides when the agent answers, stops, or keeps
talking. Deepgram transcribes, OpenAI answers, and Cartesia speaks.

## How it works

`CueAgent.stt_node` in [agent.py](agent.py) copies the audio the STT hears to
the native worker, resampled to 16 kHz. Transcripts and the agent's speaking
state become Cue context. Decisions map to session calls, using the thresholds
from `configs/live-conversation.json`:

| Cue observes | Session call |
| --- | --- |
| `end` | `interrupt()`, then `commit_user_turn()` so the LLM answers |
| `interrupt` | `interrupt()` |
| `backchannel` | `clear_user_turn()`; the agent keeps talking |
| `wait` | None |
| Speech longer than `max_backchannel_ms` (1.2 s) while the agent speaks | `interrupt()` |
| Speech for `min_voiced_ms` (200 ms) while the agent is thinking | `interrupt()` |

Transcripts that arrive while a commit waits for them belong to that turn, so
they are not sent to Cue as the next turn's text.

## Run

Build the worker and download the models, and obtain the native ONNX Runtime
1.24.4 library, as described in the [product README](../../README.md). Then,
from the repository root:

```console
python -m venv .venv-livekit
.venv-livekit/bin/python -m pip install -e ./python -r examples/livekit/requirements.txt
cp examples/livekit/.env.example examples/livekit/.env
```

Fill in `examples/livekit/.env` with your LiveKit project URL and keys, the
provider API keys, and `SAO_TURN_RUNTIME`. Start the agent, then join a room in
that project from any LiveKit client; LiveKit dispatches the agent to it:

```console
.venv-livekit/bin/python examples/livekit/agent.py dev
```

On Windows, use `.venv-livekit\Scripts\python.exe`. The worker and models
default to `target/release` and `assets/models`; override them with
`SAO_TURN_NATIVE_EXE`, `SAO_TURN_MODEL`, and `SAO_TURN_VAD`.

## Limits

`clear_user_turn()` restarts LiveKit's STT stream, so each backchannel costs an
STT reconnect. Use headphones or a client with echo cancellation: the worker
hears whatever the microphone picks up. Cue's text rules are English
heuristics; validate the thresholds on your own audio before relying on them.
