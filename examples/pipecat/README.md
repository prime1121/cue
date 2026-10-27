# Pipecat example

A Pipecat 1.12 voice bot whose user turns come from Cue instead of Pipecat's
own VAD and turn analyzer. Deepgram transcribes, OpenAI answers, Cartesia
speaks, and the browser connects through Pipecat's SmallWebRTC runner.

## How it works

`SaoTurnProcessor` sends the caller's 16 kHz audio, transcripts, and bot
speaking state to the native worker. `CueTurnTaking` in [bot.py](bot.py) turns
its observations into Pipecat user turns, using the thresholds from
`configs/live-conversation.json`:

| Cue observes | Bot is quiet | Bot is speaking |
| --- | --- | --- |
| Speech for `min_voiced_ms` (200 ms) | Open a user turn | Keep talking |
| Speech longer than `max_backchannel_ms` (1.2 s) | Turn already open | Interrupt and open a turn |
| `end` | Close the turn; the LLM answers | Interrupt, then answer |
| `interrupt` | Not emitted | Interrupt and open a turn |
| `backchannel` | Not emitted | Keep talking; drop the transcript |
| `wait` | Keep listening | Keep listening |

Transcripts heard outside a user turn, including everything said while the bot
speaks, are held until Cue rules on them, so a backchannel never reaches the
LLM. Cue's speech start and stop also travel upstream as Pipecat VAD frames, so
Deepgram finalizes its transcript as soon as the caller stops, as it would with
Pipecat's own VAD. An `end` waits up to 500 ms for that final transcript, so the
LLM gets the whole utterance. The user aggregator uses
`ExternalUserTurnStrategies` without interruptions, and `CueTurnTaking` sends
its `InterruptionFrame` downstream only: an interruption travelling upstream can
cancel `SaoTurnProcessor` in the middle of a native request.

## Run

Build the worker and download the models as described in the
[product README](../../README.md). Then, from the repository root:

```console
python -m venv .venv-pipecat
.venv-pipecat/bin/python -m pip install -e "./python[pipecat]" -r examples/pipecat/requirements.txt
cp examples/pipecat/.env.example .env
```

Add the provider API keys to `.env` and set `SAO_TURN_RUNTIME`
to the native ONNX Runtime library. Pipecat installs `onnxruntime==1.24.4` in the
same environment, so the library is in its `site-packages/onnxruntime/capi`
folder, for example `libonnxruntime.1.24.4.dylib` on macOS. Start the bot and
open the address it prints, http://localhost:7860:

```console
.venv-pipecat/bin/python examples/pipecat/bot.py
```

On Windows, use `.venv-pipecat\Scripts\python.exe`. The worker and models
default to `target/release` and `assets/models`; override them with
`SAO_TURN_NATIVE_EXE`, `SAO_TURN_MODEL`, and `SAO_TURN_VAD`.

## Noise reduction (optional)

[Hush](https://github.com/saolabsai/hush) removes background noise before Cue
and Deepgram hear the caller. Clone it next to this repository, build it as its
README describes, and install its Python package into the same environment:

```console
.venv-pipecat/bin/python -m pip install -e ../hush/python
```

Set `SAO_DENOISE_EXE` in `.env` to the absolute path of the built
`sao-denoise` binary. The bot then captures the browser audio at 48 kHz, Hush
cleans it, and `ResampleForCue` converts it to the 16 kHz that Cue requires.
Hush delays the audio by about 30 ms plus processing time, and it reduces noise
only: it does not cancel echo or remove other voices. Without `SAO_DENOISE_EXE`,
the bot uses the microphone audio as it arrives.

## Limits

Use headphones or a client with echo cancellation: the worker hears whatever
the microphone picks up, including the bot. Cue's text rules are English
heuristics, and a late STT result can still reach the next turn. While the bot
speaks, any pause restarts the 1.2 s barge-in count, so a barge-in with pauses
stops the bot only when Cue ends the turn. Validate the thresholds on your own
audio before relying on them.
