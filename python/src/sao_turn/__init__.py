"""Async Rust bridge; install the ``pipecat`` extra for pipeline frames."""

from .client import (
    SaoTurnClient,
    TurnClientError,
    TurnClientTimeout,
    TurnConfig,
    TurnContext,
    TurnProtocolError,
)

__all__ = [
    "SaoTurnClient", "TurnClientError", "TurnClientTimeout", "TurnConfig",
    "TurnContext", "TurnProtocolError", "SaoTurnProcessor", "TurnContextFrame",
    "TurnObservationFrame", "TurnResetFrame",
]


def __getattr__(name: str):
    if name in {"SaoTurnProcessor", "TurnContextFrame", "TurnObservationFrame", "TurnResetFrame"}:
        from . import pipecat as bridge
        return getattr(bridge, name)
    raise AttributeError(name)
