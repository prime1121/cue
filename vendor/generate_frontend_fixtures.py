"""Optional developer-only fixture generation; never used by the Rust runtime.

Run: python vendor/generate_frontend_fixtures.py
Requires numpy; onnxruntime additionally records reference inference outputs.
The frozen reference is Pipecat's vendored NumPy Whisper implementation.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "frontend"
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("reference", Path(__file__).with_name("whisper_features_reference.py"))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

# Deterministic LCG noise plus tones with an amplitude envelope and DC offset.
# Store input bytes so tests do not depend on cross-platform sine rounding.
count = 145731
noise = np.empty(count, dtype=np.float64)
state = 0xC0FFEE
for index in range(count):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    noise[index] = (state / 4294967296.0) * 2.0 - 1.0
t = np.arange(count, dtype=np.float64) / 16000.0
envelope = 0.2 + 0.8 * np.sin(2.0 * np.pi * 1.7 * t) ** 2
audio = (0.01 + envelope * (0.15 * np.sin(2.0 * np.pi * 173.0 * t)
         + 0.07 * np.sin(2.0 * np.pi * 911.0 * t) + 0.02 * noise)).astype("<f4")
audio.tofile(OUT / "probe.f32")

def frontend(samples):
    padded = samples[-128000:]
    padded = np.pad(padded, (128000 - len(padded), 0))
    return reference.compute_whisper_log_mel_features(padded, do_normalize=True)

short = audio[-20513:]
features = {"long": frontend(audio), "short": frontend(short), "silence": frontend(np.zeros(512, np.float32))}
for key in ("long", "short"):
    features[key].astype("<f4").tofile(OUT / f"{key}.mel.f32")

metadata = {
    "generator": "vendor/generate_frontend_fixtures.py",
    "reference": "pipecat-ai/pipecat@e6325f57fff7180e815e6bdb49ea4ede18e404ca",
    "numpy_version": np.__version__,
    "format": "little-endian float32; mel arrays are row-major [80,800]",
    "samples": count,
    "short_samples": 20513,
    "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in OUT.glob("*.f32")},
}
try:
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    model_dir = ROOT / "assets" / "models"
    smart = ort.InferenceSession(str(model_dir / "smart-turn-v3.2-cpu.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    vad = ort.InferenceSession(str(model_dir / "silero-vad.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    metadata["onnxruntime_version"] = ort.__version__
    metadata["smart_turn_probabilities"] = {
        key: float(smart.run(None, {"input_features": values[None]})[0].reshape(-1)[0])
        for key, values in features.items()
    }
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros((1, 64), np.float32)
    scores = []
    for frame in audio[:512*64].reshape(64, 512):
        samples = np.concatenate((context, frame[None]), axis=1)
        output, state = vad.run(None, {"input": samples, "state": state, "sr": np.array(16000, dtype=np.int64)})
        scores.append(float(output.reshape(-1)[0]))
        context = samples[:, -64:]
    metadata["silero_probabilities"] = scores
    print("Smart Turn inputs", [(x.name, x.shape, x.type) for x in smart.get_inputs()])
    print("Silero inputs", [(x.name, x.shape, x.type) for x in vad.get_inputs()])
except ImportError:
    print("onnxruntime unavailable; recorded only frontend reference fixtures")

(OUT / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in metadata.items() if k != "silero_probabilities"}, indent=2))
