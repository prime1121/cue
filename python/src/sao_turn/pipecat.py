"""Pipecat 1.12 bridge for continuous 16 kHz mono input and turn observations."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame, BotStoppedSpeakingFrame, CancelFrame, DataFrame,
    EndFrame, Frame, InputAudioRawFrame, InterimTranscriptionFrame, StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .client import SaoTurnClient, TurnContext


@dataclass
class TurnContextFrame(DataFrame):
    """Replace current-turn context; does not reset audio or the native clock."""

    context: TurnContext | Mapping[str, Any]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.context = TurnContext.from_value(self.context)


@dataclass
class TurnResetFrame(DataFrame):
    """Explicitly clear native audio, VAD, policy, clock, and transcript state."""

    context: TurnContext | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.context = TurnContext.from_value(self.context)


@dataclass
class TurnObservationFrame(DataFrame):
    """One native FrameResult, including optional actionable policy decisions."""

    payload: dict[str, Any]


class SaoTurnProcessor(FrameProcessor):
    """Owns its client's lifecycle and preserves original input audio.

    End/Backchannel clears text through an empty context packet, never a full
    native reset. Interrupt remains a decision for the application to consume;
    this class does not synthesize UserStoppedSpeakingFrame or InterruptionFrame.
    """

    def __init__(
        self, client: SaoTurnClient, *, context: TurnContext | Mapping[str, Any] | None = None, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._client = client
        self._initial_context = TurnContext.from_value(context)
        self._context = self._initial_context
        self._context_lock = asyncio.Lock()
        self._started = False
        self._failed = False

    @property
    def context(self) -> TurnContext:
        return self._context

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (CancelFrame, EndFrame)):
            await self._client.close()
            self._started = False
            await self.push_frame(frame, direction)
            return
        try:
            if isinstance(frame, StartFrame):
                async with self._context_lock:
                    self._context = self._initial_context
                    await self._client.start()
                    await self._client.update_context(self._context)
                    self._started = True
                    self._failed = False
                await self.push_frame(frame, direction)
                return

            if self._failed:
                return

            if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
                async with self._context_lock:
                    updated = replace(self._context, agent_speaking=isinstance(frame, BotStartedSpeakingFrame))
                    await self._set_context(updated)
                await self.push_frame(frame, direction)
                return

            if direction != FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)
                return

            if isinstance(frame, TurnResetFrame):
                async with self._context_lock:
                    self._require_started()
                    context = TurnContext.from_value(frame.context)
                    await self._client.reset(context)
                    self._context = context
                await self.push_frame(frame, direction)
                return

            if isinstance(frame, TurnContextFrame):
                async with self._context_lock:
                    await self._set_context(TurnContext.from_value(frame.context))
                await self.push_frame(frame, direction)
                return

            if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
                async with self._context_lock:
                    context = replace(self._context, text=frame.text, text_final=isinstance(frame, TranscriptionFrame))
                    await self._set_context(context)
                await self.push_frame(frame, direction)
                return

            if isinstance(frame, InputAudioRawFrame):
                if frame.sample_rate != 16_000 or frame.num_channels != 1:
                    raise ValueError("SaoTurnProcessor requires mono 16000 Hz PCM16 input")
                if len(frame.audio) % 2:
                    raise ValueError("PCM16 input byte length must be even")
                await self.push_frame(frame, direction)
                async with self._context_lock:
                    self._require_started()
                    response = await self._client.process_audio(frame.audio, context=self._context)
                    for payload in response["frames"]:
                        observation = TurnObservationFrame(payload=payload)
                        observation.pts = frame.pts
                        observation.transport_source = frame.transport_source
                        await self.push_frame(observation, direction)
                    if any((item.get("decision") or {}).get("kind") in {"end", "backchannel"} for item in response["frames"]):
                        await self._set_context(replace(self._context, text="", text_final=False))
                return

            await self.push_frame(frame, direction)
        except asyncio.CancelledError:
            await asyncio.shield(self._client.close())
            self._started = False
            raise
        except Exception as error:
            self._failed = True
            self._started = False
            await self._client.close()
            await self.push_error(
                f"Sao Turn bridge failed: {error}", exception=error,
                force_treat_as_permanent=True,
            )

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("SaoTurnProcessor has not received a successful StartFrame")

    async def _set_context(self, context: TurnContext) -> None:
        self._require_started()
        await self._client.update_context(context)
        self._context = context

    async def cleanup(self) -> None:
        try:
            await self._client.close()
            self._started = False
        finally:
            await super().cleanup()
