# Third-party sources and model provenance

The Rust application performs inference locally without a Python runtime. Python
is used only by the optional reference-fixture generator in this directory.

## Smart Turn v3.2 CPU

- Author: Pipecat / Daily.
- License: BSD-2-Clause, reproduced in `LICENSE.pipecat`.
- Original model: https://huggingface.co/pipecat-ai/smart-turn-v3
- Pinned redistribution: https://github.com/pipecat-ai/pipecat/blob/e6325f57fff7180e815e6bdb49ea4ede18e404ca/src/pipecat/audio/turn/smart_turn/data/smart-turn-v3.2-cpu.onnx
- SHA256: `2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f`.
- Input: float32 `[1,80,800]` named `input_features`.
- Output: completion probability, already passed through sigmoid.

## Silero VAD

- Author: Silero Team.
- License: MIT, reproduced in `LICENSE.silero`.
- Pinned model redistribution: https://github.com/pipecat-ai/pipecat/blob/e6325f57fff7180e815e6bdb49ea4ede18e404ca/src/pipecat/audio/vad/data/silero_vad.onnx
- SHA256: `597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004`.
- Wrapper reference: https://github.com/snakers4/silero-vad/blob/1e261b036686cd0017d500ee96acd1c4ba572a9d/src/silero_vad/utils_vad.py
- This integration uses 16 kHz only: 512 fresh samples plus 64 samples of context,
  recurrent state `[2,1,128]`, and scalar int64 sample rate `16000`.

## Audio frontend

`src/features.rs` is a Rust translation of the mathematical preprocessing in
Pipecat's `local_smart_turn_v3.py` and `_whisper_features.py`. It retains the last
eight seconds, left-pads to 128,000 samples, normalizes the full padded waveform,
and computes centered, reflect-padded 400-point periodic-Hann STFT with 160-sample
hop, 80 Slaney-normalized mel bands, log10 floor, 8-decade clipping and affine
scaling. Normalization is float32; spectral calculations are float64.

- Pinned reference file, retained verbatim as `whisper_features_reference.py`:
  https://github.com/pipecat-ai/pipecat/blob/e6325f57fff7180e815e6bdb49ea4ede18e404ca/src/pipecat/audio/turn/smart_turn/_whisper_features.py
- Pipecat copyright 2024-2026 Daily, BSD-2-Clause.
- Portions of that reference derive from Hugging Face Transformers,
  copyright 2022 The HuggingFace Inc. team, and copyright 2023 The HuggingFace Inc.
  team and the librosa & torchaudio authors, Apache-2.0. See `LICENSE.transformers`.

The Rust reduction order and FFT implementation can differ in floating-point
rounding from NumPy; parity is checked numerically, not claimed bit-for-bit.
`generate_frontend_fixtures.py` records complete reference feature matrices and
model outputs for deterministic synthetic audio. These fixtures test numerical
compatibility; they do not establish conversational accuracy or superiority to
Smart Turn.

## Verification on 2026-10-14

The full release test suite in `tests/frontend_parity.rs`, including its optional
native-model test, passed against ONNX Runtime 1.24.4 on Windows x86-64:

- Both 80-by-800 feature matrices matched the frozen Pipecat NumPy reference
  exactly after float32 conversion on this machine (maximum absolute error 0).
- Independent comparison against Transformers 4.57.6 `WhisperFeatureExtractor`
  produced maximum absolute errors of 0.00001207 (long input) and 0.00001068
  (short input), with mean errors below 0.000000121. Its Torch FFT follows a
  different floating-point path from the Pipecat NumPy frontend.
- Native Smart Turn probabilities matched Python ONNX Runtime exactly:
  long 0.889684319, short 0.877542257, silence 0.987036705.
- All 64 consecutive Silero frame probabilities and reset behavior matched the
  reference within 0.00002 absolute error.
- Observed single-thread release Smart Turn prediction times, including Rust
  preprocessing, were 77.9 ms (first long call), 70.4 ms (short), and 61.1 ms
  (silence). The 64 Silero frames took 12.4 ms total. These are small local
  engineering measurements, not a latency benchmark across hardware.

Silence receiving a high *completion* probability is expected from an endpoint
classifier and is why application turn detection must require detected speech
before considering completion scores.
