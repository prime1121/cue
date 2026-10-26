from __future__ import annotations

import unittest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    BotStartedSpeakingFrame, BotStoppedSpeakingFrame, CancelFrame, EndFrame,
    InputAudioRawFrame, InterimTranscriptionFrame, StartFrame, TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager

from sao_turn.client import TurnContext
from sao_turn.pipecat import SaoTurnProcessor, TurnContextFrame, TurnObservationFrame, TurnResetFrame


class FakeClient:
    def __init__(self):
        self.calls = []
        self.frames = []
        self.failure = None

    async def start(self):
        self.calls.append(("start",))
        if self.failure == "start":
            raise RuntimeError("test startup failure")

    async def update_context(self, context):
        self.calls.append(("context", context))
        return {"frames": [], "buffered_samples": 0}

    async def process_audio(self, audio, *, context):
        self.calls.append(("audio", audio, context))
        if self.failure == "audio":
            raise RuntimeError("test inference failure")
        return {"frames": self.frames, "buffered_samples": 0}

    async def reset(self, context):
        self.calls.append(("reset", context))

    async def close(self):
        self.calls.append(("close",))


class RecordingProcessor(SaoTurnProcessor):
    def __init__(self, client):
        super().__init__(client)
        self.output = []
        self.errors = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.output.append((frame, direction))

    async def push_error(self, message, **kwargs):
        self.errors.append((message, kwargs))


class ProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = FakeClient()
        self.processor = RecordingProcessor(self.client)
        self.manager = TaskManager()
        await self.processor.setup(FrameProcessorSetup(
            clock=SystemClock(), task_manager=self.manager, pipeline_worker=None,
        ))

    async def asyncTearDown(self):
        await self.processor.cleanup()
        self.assertFalse(self.manager.current_tasks())

    async def send(self, frame, direction=FrameDirection.DOWNSTREAM):
        await self.processor.process_frame(frame, direction)

    async def test_start_readiness_audio_identity_and_observations(self):
        start = StartFrame()
        await self.send(start)
        self.assertEqual(self.client.calls[0], ("start",))
        self.assertIs(self.processor.output[-1][0], start)
        self.client.frames = [{"at_ms": 32, "decision": None}, {"at_ms": 64, "decision": None}]
        audio = InputAudioRawFrame(audio=bytes(2_048), sample_rate=16_000, num_channels=1)
        audio.pts = 1234
        audio.metadata["test"] = 42
        await self.send(audio)
        self.assertIs(self.processor.output[-3][0], audio)
        events = [row[0] for row in self.processor.output if isinstance(row[0], TurnObservationFrame)]
        self.assertEqual([event.payload["at_ms"] for event in events], [32, 64])
        self.assertEqual(events[0].pts, 1234)
        self.assertEqual(audio.metadata["test"], 42)

    async def test_context_transcripts_and_bot_state_replace_safely(self):
        await self.send(StartFrame())
        await self.send(TurnContextFrame({"mode": "digits", "expected_digits": 4}))
        await self.send(InterimTranscriptionFrame(text="one", user_id="u", timestamp="t"))
        self.assertEqual(self.processor.context.text, "one")
        self.assertFalse(self.processor.context.text_final)
        await self.send(TranscriptionFrame(text="one two", user_id="u", timestamp="t"))
        self.assertEqual(self.processor.context.text, "one two")
        self.assertTrue(self.processor.context.text_final)
        self.assertEqual(self.processor.context.expected_digits, 4)
        await self.send(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        self.assertTrue(self.processor.context.agent_speaking)
        await self.send(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        self.assertFalse(self.processor.context.agent_speaking)
        with self.assertRaises(ValueError):
            TurnContextFrame({"unknown": 1})

    async def test_terminal_only_clears_text_and_reset_is_explicit(self):
        await self.send(StartFrame())
        await self.send(TurnContextFrame({"mode": "digits", "expected_digits": 4, "text": "1234", "text_final": True}))
        self.client.frames = [{"at_ms": 32, "decision": {"kind": "end"}}]
        await self.send(InputAudioRawFrame(audio=bytes(1_024), sample_rate=16_000, num_channels=1))
        self.assertEqual(self.processor.context.text, "")
        self.assertFalse(self.processor.context.text_final)
        self.assertEqual(self.processor.context.mode, "digits")
        self.assertFalse(any(call[0] == "reset" for call in self.client.calls))
        await self.send(TurnResetFrame())
        self.assertEqual(self.processor.context, TurnContext())
        self.assertEqual(self.client.calls[-1], ("reset", TurnContext()))

    async def test_interrupt_is_observation_only_and_backchannel_clears_text(self):
        await self.send(StartFrame())
        await self.send(TurnContextFrame({"text": "stop", "text_final": True, "agent_speaking": True}))
        self.client.frames = [{"at_ms": 32, "decision": {"kind": "interrupt"}}]
        await self.send(InputAudioRawFrame(audio=bytes(1_024), sample_rate=16_000, num_channels=1))
        self.assertEqual(self.processor.context.text, "stop")
        self.assertTrue(all(type(frame).__name__ not in {"UserStoppedSpeakingFrame", "InterruptionFrame"} for frame, _ in self.processor.output))
        self.client.frames = [{"at_ms": 64, "decision": {"kind": "backchannel"}}]
        await self.send(InputAudioRawFrame(audio=bytes(1_024), sample_rate=16_000, num_channels=1))
        self.assertEqual(self.processor.context.text, "")
        self.assertTrue(self.processor.context.agent_speaking)

    async def test_startup_and_audio_errors_close_and_mark_permanent(self):
        self.client.failure = "start"
        await self.send(StartFrame())
        self.assertFalse(any(isinstance(frame, StartFrame) for frame, _ in self.processor.output))
        self.assertTrue(self.processor.errors[-1][1]["force_treat_as_permanent"])
        self.assertEqual(self.client.calls[-1], ("close",))
        self.client.failure = None
        await self.send(StartFrame())
        self.client.failure = "audio"
        await self.send(InputAudioRawFrame(audio=bytes(1_024), sample_rate=16_000, num_channels=1))
        self.assertIn("inference failure", self.processor.errors[-1][0])
        self.assertEqual(self.client.calls[-1], ("close",))

    async def test_invalid_audio_and_end_cancel_cleanup(self):
        await self.send(StartFrame())
        await self.send(InputAudioRawFrame(audio=bytes(1_024), sample_rate=48_000, num_channels=1))
        self.assertIn("16000", self.processor.errors[-1][0])
        self.assertFalse(any(call[0] == "audio" for call in self.client.calls))
        end = EndFrame()
        await self.send(end)
        self.assertIs(self.processor.output[-1][0], end)
        cancel = CancelFrame()
        await self.send(cancel)
        self.assertIs(self.processor.output[-1][0], cancel)
        self.assertEqual(self.client.calls[-1], ("close",))


if __name__ == "__main__":
    unittest.main()
