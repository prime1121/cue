"""LiveKit voice agent whose user turns follow Cue decisions instead of LiveKit's turn detection."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent, AgentServer, AgentSession, AgentStateChangedEvent, JobContext, ModelSettings,
    UserInputTranscribedEvent, cli,
)
from livekit.plugins import cartesia, deepgram, openai

from sao_turn import SaoTurnClient, TurnConfig, TurnContext

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/live-conversation.json"
INSTRUCTIONS = (
    "You are a friendly voice assistant. Answer in one or two short spoken sentences. "
    "If the user asks you to stop or wait, acknowledge it in a few words."
)

logger = logging.getLogger("cue")

load_dotenv()


class CueAgent(Agent):
    """Feeds the audio its STT hears to Cue and turns Cue decisions into manual turn calls."""

    def __init__(self, cue: SaoTurnClient, config: TurnConfig) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self._cue = cue
        self._config = config
        self._audio: asyncio.Queue[rtc.AudioFrame] = asyncio.Queue()
        self._context = TurnContext()
        self._finals: list[str] = []
        self._speech_started_ms: int | None = None
        self._commit: asyncio.Future[str] | None = None
        self._feeder: asyncio.Task[None] | None = None

    async def on_enter(self) -> None:
        self.session.on("agent_state_changed", self._on_agent_state)
        self.session.on("user_input_transcribed", self._on_transcript)
        self._feeder = asyncio.create_task(self._feed())

    async def on_exit(self) -> None:
        self.session.off("agent_state_changed", self._on_agent_state)
        self.session.off("user_input_transcribed", self._on_transcript)
        if self._feeder is not None:
            self._feeder.cancel()

    async def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings) -> AsyncIterable[Any]:
        async def heard() -> AsyncIterable[rtc.AudioFrame]:
            async for frame in audio:
                self._audio.put_nowait(frame)
                yield frame

        async for event in Agent.default.stt_node(self, heard(), model_settings):
            yield event

    async def _feed(self) -> None:
        resampler: rtc.AudioResampler | None = None
        try:
            while True:
                frame = await self._audio.get()
                if frame.sample_rate == 16_000:
                    chunks = [frame]
                else:
                    resampler = resampler or rtc.AudioResampler(frame.sample_rate, 16_000)
                    chunks = resampler.push(frame)
                for chunk in chunks:
                    response = await self._cue.process_audio(chunk.data, context=self._context)
                    for result in response["frames"]:
                        self._observe(result)
        except Exception:
            logger.exception("Cue turn taking failed; closing the session")
            self.session.shutdown()

    def _observe(self, result: dict[str, Any]) -> None:
        if self._speech_started_ms is None and result["speech_probability"] >= self._config.speech_start_threshold:
            self._speech_started_ms = result["at_ms"]
        elif self._speech_started_ms is not None and result["speech_probability"] <= self._config.speech_end_threshold:
            self._speech_started_ms = None
        decision = result["decision"] or {}
        kind = decision.get("kind")
        if kind:
            logger.info("cue %s: %s", kind, decision["reason"])
        if kind == "backchannel":
            self.session.clear_user_turn()
        elif kind in ("interrupt", "end") or self._speech_takes_floor(result["at_ms"]):
            self.session.interrupt()
        if kind == "end":
            self._commit = self.session.commit_user_turn()
        if kind in ("end", "backchannel"):
            self._finals.clear()
            self._context = replace(self._context, text="", text_final=False)

    def _speech_takes_floor(self, at_ms: int) -> bool:
        state = self.session.agent_state
        if self._speech_started_ms is None or state not in ("thinking", "speaking"):
            return False
        speech_ms = at_ms - self._speech_started_ms
        # While the agent talks, only speech longer than any backchannel takes the floor.
        if state == "speaking":
            return speech_ms > self._config.max_backchannel_ms
        return speech_ms >= self._config.min_voiced_ms

    def _on_agent_state(self, event: AgentStateChangedEvent) -> None:
        self._context = replace(self._context, agent_speaking=event.new_state == "speaking")

    def _on_transcript(self, event: UserInputTranscribedEvent) -> None:
        # Results that arrive while a commit waits for them belong to the committed turn.
        if self._commit is not None and not self._commit.done():
            return
        if event.is_final:
            self._finals.append(event.transcript)
        text = " ".join([*self._finals, "" if event.is_final else event.transcript]).strip()
        self._context = replace(self._context, text=text, text_final=event.is_final)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    executable = ROOT / "target/release" / ("sao-turn.exe" if os.name == "nt" else "sao-turn")
    cue = SaoTurnClient(
        executable=os.getenv("SAO_TURN_NATIVE_EXE") or executable,
        model=os.getenv("SAO_TURN_MODEL") or ROOT / "assets/models/smart-turn-v3.2-cpu.onnx",
        vad_model=os.getenv("SAO_TURN_VAD") or ROOT / "assets/models/silero-vad.onnx",
        runtime=os.getenv("SAO_TURN_RUNTIME") or None,
        config=CONFIG,
    )
    await cue.start()
    ctx.add_shutdown_callback(cue.close)
    session = AgentSession(
        stt=deepgram.STT(), llm=openai.LLM(), tts=cartesia.TTS(), turn_handling={"turn_detection": "manual"},
    )
    config = TurnConfig.from_dict(json.loads(CONFIG.read_text(encoding="utf-8")))
    await session.start(agent=CueAgent(cue, config), room=ctx.room)
    await session.generate_reply(instructions="Greet the user in one short sentence.")


if __name__ == "__main__":
    cli.run_app(server)
