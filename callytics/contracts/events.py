"""Event payloads published on the bus. One class per topic."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from .intelligence import CallIntelligence


class LeadCaptured(BaseModel):
    lead_id: UUID
    source: str
    stage: str
    owner_id: UUID | None = None
    is_new: bool = True
    merged_into: UUID | None = None
    captured_at: datetime


class LeadStageChanged(BaseModel):
    lead_id: UUID
    from_stage: str | None
    to_stage: str
    changed_by: UUID | None
    reason: str
    occurred_at: datetime


class LeadOwnerChanged(BaseModel):
    lead_id: UUID
    from_owner: UUID | None
    to_owner: UUID | None
    reason: str
    occurred_at: datetime


class ActivityRecorded(BaseModel):
    activity_id: UUID
    lead_id: UUID
    type: str
    channel: str
    occurred_at: datetime


class CallIntelligenceReceived(BaseModel):
    lead_id: UUID
    intelligence: CallIntelligence


class ProposalRaised(BaseModel):
    proposal_id: UUID
    lead_id: UUID
    kind: str
    confidence: float
    source: str


class ProposalDecided(BaseModel):
    proposal_id: UUID
    lead_id: UUID
    status: str
    decided_by: UUID | None


class ScoreRecomputed(BaseModel):
    lead_id: UUID
    score: int
    band: str
    previous_score: int | None = None


class SlaBreached(BaseModel):
    lead_id: UUID
    rule: str
    due_at: datetime
    breached_at: datetime
    owner_id: UUID | None
    escalate_to: UUID | None = None


class AudienceSyncRequested(BaseModel):
    segment_id: UUID
    destination: str
    lead_count: int
    requested_by: UUID | None = None
    dry_run: bool = True


class NotificationRequested(BaseModel):
    recipient_id: UUID
    kind: str
    lead_id: UUID | None = None
    title: str
    body: str
    payload: dict[str, Any] = Field(default_factory=dict)
