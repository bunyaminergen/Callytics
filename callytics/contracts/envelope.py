"""Event envelope riding every message on the bus.

Deliberately the same shape as FinEcho's envelope so a message can cross the
service boundary between the two systems without a translation layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..util import utcnow, uuid7

ENVELOPE_SCHEMA_VERSION = 1


class EventEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid7)
    schema_version: int = ENVELOPE_SCHEMA_VERSION
    event_type: str
    occurred_at: datetime = Field(default_factory=utcnow)
    producer: str
    trace_id: UUID = Field(default_factory=uuid7)
    lead_id: UUID | None = None
    payload: dict[str, Any]

    def to_json(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> EventEnvelope:
        return cls.model_validate_json(raw)


def make_event(
    event_type: str,
    payload: BaseModel | dict[str, Any],
    *,
    producer: str,
    trace_id: UUID | None = None,
    lead_id: UUID | None = None,
) -> EventEnvelope:
    body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    return EventEnvelope(
        event_type=event_type,
        payload=body,
        producer=producer,
        trace_id=trace_id or uuid7(),
        lead_id=lead_id,
    )
