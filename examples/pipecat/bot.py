"""Pipecat voice bot whose user turns follow Cue decisions instead of Pipecat's VAD."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import av
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, InputAudioRawFrame, InterimTranscriptionFrame,
    InterruptionFrame, LLMRunFrame, ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

from sao_turn import SaoTurnClient, TurnConfig
from sao_turn.pipecat import SaoTurnProcessor, TurnObservationFrame

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/live-conversation.json"
VOICE = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
# How long an end waits for the transcript the STT service finalizes when speech stops.
FINAL_TRANSCRIPT_WAIT_MS = 500
INSTRUCTIONS = (
    "You are a friendly voice assistant. Greet the user when the conversation starts. "
    "Answer in one or two short spoken sentences. If the user asks you to stop or wait, "
    "acknowledge it in a few words."
)

load_dotenv(override=True)


class ResampleForCue(FrameProcessor):
    """Converts microphone audio captured at 48 kHz for Hush to the 16 kHz that Cue requires.

    libswresample, which the WebRTC transport also uses, holds back about 1 ms of audio.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and frame.sample_rate != 16_000:
            audio = av.AudioFrame.from_ndarray(np.frombuffer(frame.audio, "<i2").reshape(1, -1), format="s16", layout="mono")
            audio.sample_rate = frame.sample_rate
            frame.audio = b"".join(resampled.to_ndarray().tobytes() for resampled in self._resampler.resample(audio))
            frame.sample_rate = 16_000
            if not frame.audio:
                return
        await self.push_frame(frame, direction)


class CueTurnTaking(FrameProcessor):
    """Proposes Pipecat user turns from Cue observations.

    Transcripts heard outside a user turn are held until Cue rules on them: a
    backchannel drops them, and an interrupt or end replays them into the turn.
    Cue's speech start and stop also travel upstream as VAD frames, so the STT
    service finalizes its transcript when the user stops, as it would with
    Pipecat's own VAD, and an end waits briefly for that transcript.
    """

    def __init__(self, config: TurnConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._bot_speaking = False
        self._user_turn = False
        self._speech_started_ms: int | None = None
        self._held: list[Frame] = []
        self._awaiting_final = False
        self._stop_at_ms: int | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TurnObservationFrame):
            await self._observe(frame.payload)
            return
        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            self._bot_speaking = isinstance(frame, BotStartedSpeakingFrame)
        elif isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            if isinstance(frame, TranscriptionFrame):
                self._awaiting_final = False
            if not self._user_turn:
                self._held.append(frame)
                return
            await self.push_frame(frame, direction)
            if self._stop_at_ms is not None and not self._awaiting_final:
                await self._stop_turn()
            return
        await self.push_frame(frame, direction)

    async def _observe(self, payload: dict[str, Any]) -> None:
        if self._speech_started_ms is None and payload["speech_probability"] >= self._config.speech_start_threshold:
            self._speech_started_ms = payload["at_ms"]
            # Speech resumed before a pending end closed the turn, so the turn continues.
            self._stop_at_ms = None
            await self.push_frame(VADUserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        elif self._speech_started_ms is not None and payload["speech_probability"] <= self._config.speech_end_threshold:
            self._speech_started_ms = None
            self._awaiting_final = True
            await self.push_frame(VADUserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        elif self._stop_at_ms is not None and payload["at_ms"] >= self._stop_at_ms:
            await self._stop_turn()
        decision = payload["decision"] or {}
        kind = decision.get("kind")
        if kind:
            logger.info(f"Cue {kind}: {decision['reason']}")
        if kind == "backchannel":
            self._held.clear()
            return
        if not self._user_turn and (kind in ("interrupt", "end") or self._speech_takes_floor(payload["at_ms"])):
            self._user_turn = True
            self._bot_speaking = False
            # Downstream only: an upstream interruption would cancel SaoTurnProcessor mid-request.
            await self.push_frame(InterruptionFrame())
            await self.push_frame(ProposedUserStartedSpeakingFrame())
            for held in self._held:
                await self.push_frame(held)
            self._held.clear()
        if kind == "end":
            if self._awaiting_final:
                self._stop_at_ms = payload["at_ms"] + FINAL_TRANSCRIPT_WAIT_MS
            else:
                await self._stop_turn()

    async def _stop_turn(self) -> None:
        self._user_turn = False
        self._stop_at_ms = None
        await self.push_frame(ProposedUserStoppedSpeakingFrame())

    def _speech_takes_floor(self, at_ms: int) -> bool:
        if self._speech_started_ms is None:
            return False
        speech_ms = at_ms - self._speech_started_ms
        # While the bot talks, only speech longer than any backchannel takes the floor.
        if self._bot_speaking:
            return speech_ms > self._config.max_backchannel_ms
        return speech_ms >= self._config.min_voiced_ms


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    executable = ROOT / "target/release" / ("sao-turn.exe" if os.name == "nt" else "sao-turn")
    cue = SaoTurnClient(
        executable=os.getenv("SAO_TURN_NATIVE_EXE") or executable,
        model=os.getenv("SAO_TURN_MODEL") or ROOT / "assets/models/smart-turn-v3.2-cpu.onnx",
        vad_model=os.getenv("SAO_TURN_VAD") or ROOT / "assets/models/silero-vad.onnx",
        runtime=os.getenv("SAO_TURN_RUNTIME") or None,
        config=CONFIG,
    )
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"], settings=OpenAILLMService.Settings(system_instruction=INSTRUCTIONS),
    )
    tts = CartesiaTTSService(api_key=os.environ["CARTESIA_API_KEY"], settings=CartesiaTTSService.Settings(voice=VOICE))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies(enable_interruptions=False)),
    )
    config = TurnConfig.from_dict(json.loads(CONFIG.read_text(encoding="utf-8")))
    task = PipelineTask(Pipeline([
        transport.input(), ResampleForCue(), stt, SaoTurnProcessor(cue), CueTurnTaking(config),
        user_aggregator, llm, tts, transport.output(), assistant_aggregator,
    ]))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport: BaseTransport, client: Any) -> None:
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport: BaseTransport, client: Any) -> None:
        await task.cancel()

    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)


def transport_params() -> TransportParams:
    """Adds Hush noise reduction when SAO_DENOISE_EXE names a built sao-denoise binary."""
    if not os.getenv("SAO_DENOISE_EXE"):
        return TransportParams(audio_in_enabled=True, audio_out_enabled=True)
    from sao_noise.pipecat import DeepFilterNetFilter

    # Hush's DeepFilterNet3 model works on 48 kHz audio; ResampleForCue converts its output for Cue and STT.
    return TransportParams(
        audio_in_enabled=True, audio_out_enabled=True, audio_in_sample_rate=48_000,
        audio_in_filter=DeepFilterNetFilter(os.environ["SAO_DENOISE_EXE"]),
    )


async def bot(runner_args: RunnerArguments) -> None:
    transport = await create_transport(runner_args, {"webrtc": transport_params})
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
