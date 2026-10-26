"""Download Cue's pinned models and verify their SHA-256 checksums."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
import time
from urllib.request import Request, urlopen


REVISION = "e6325f57fff7180e815e6bdb49ea4ede18e404ca"
BASE_URL = f"https://raw.githubusercontent.com/pipecat-ai/pipecat/{REVISION}/src/pipecat/audio/"
MAX_BYTES = 16 * 1024 * 1024
SOCKET_TIMEOUT_SECONDS = 20
DOWNLOAD_TIMEOUT_SECONDS = 120
CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class Model:
    name: str
    source: str
    sha256: str


MODELS = (
    Model(
        "smart-turn-v3.2-cpu.onnx",
        "turn/smart_turn/data/smart-turn-v3.2-cpu.onnx",
        "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
    ),
    Model(
        "silero-vad.onnx",
        "vad/data/silero_vad.onnx",
        "597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004",
    ),
)


def checksum(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular model file: {path}")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"Model exceeds the {MAX_BYTES}-byte limit: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(model: Model, destination: Path, *, check_only: bool = False) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / model.name
    if target.exists() or target.is_symlink():
        actual = checksum(target)
        if actual != model.sha256:
            raise ValueError(
                f"Existing checksum mismatch for {target}; file left unchanged. "
                f"Expected {model.sha256}, got {actual}. Remove it explicitly before retrying."
            )
        return target
    if check_only:
        raise FileNotFoundError(f"Model is missing: {target}")

    temporary: Path | None = None
    started = time.monotonic()
    try:
        request = Request(BASE_URL + model.source, headers={"User-Agent": "Cue-model-preparation/0.1"})
        with urlopen(request, timeout=SOCKET_TIMEOUT_SECONDS) as response:
            if not response.geturl().startswith("https://"):
                raise ValueError("Model download redirected to an insecure URL")
            length = response.headers.get("Content-Length")
            if length is not None and (int(length) < 0 or int(length) > MAX_BYTES):
                raise ValueError(f"Invalid or oversized model Content-Length: {length}")
            descriptor, name = tempfile.mkstemp(prefix=f".{model.name}.", suffix=".partial", dir=destination)
            temporary = Path(name)
            digest = hashlib.sha256()
            total = 0
            with os.fdopen(descriptor, "wb") as output:
                while True:
                    if time.monotonic() - started > DOWNLOAD_TIMEOUT_SECONDS:
                        raise TimeoutError(f"Model download exceeded {DOWNLOAD_TIMEOUT_SECONDS} seconds")
                    # One socket read per iteration keeps the overall deadline
                    # observable even if the sender dribbles bytes slowly.
                    chunk = response.read1(CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise ValueError(f"Model download exceeds {MAX_BYTES} bytes")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if digest.hexdigest() != model.sha256:
                raise ValueError(f"Downloaded checksum mismatch for {model.name}; partial file discarded")
            if target.exists() or target.is_symlink():
                if checksum(target) != model.sha256:
                    raise ValueError(f"Destination changed during download; existing file left unchanged: {target}")
                return target
            os.replace(temporary, target)
            temporary = None
            return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "models")
    parser.add_argument("--check", action="store_true", help="Verify existing files without downloading")
    args = parser.parse_args()
    for model in MODELS:
        path = prepare(model, args.destination, check_only=args.check)
        print(f"{model.name}: SHA-256 verified ({path})")
    print("Model sources and licenses: vendor/PROVENANCE.md")


if __name__ == "__main__":
    main()
