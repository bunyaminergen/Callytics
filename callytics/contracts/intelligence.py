"""Intelligence contracts: scores, proposals, insights, next-best-action.

The design rule for everything in this module: **machines propose, humans
dispose, and every output carries its evidence.** The competing product demoed
to us auto-changes lead stages from call content, and their own sales engineer
conceded managers "will never have that trust" in it. The fix is not to drop
the automation — it is to make it accountable. A ``Proposal`` names what it
wants to change, how sure it is, and exactly which timeline facts justify it.
A counsellor accepts or rejects in one click, and the decision is recorded,
which also produces the labelled dataset needed to improve the model.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .vocabulary import Channel


class Evidence(BaseModel):
    """A pointer to a fact already on the timeline.

    Never free text invented by a model: an evidence item must reference a real
    activity so a manager can click through and read the source.
    """

    model_config = ConfigDict(frozen=True)

    activity_id: UUID | None = None
    kind: str
    detail: str
    occurred_at: datetime | None = None


class ScoreContribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    points: float
    detail: str


class LeadScore(BaseModel):
    """Explainable 0-100 lead score.

    Deterministic and additive on purpose. A counsellor asking "why is this a
    78?" gets the arithmetic, not a shrug — which is the difference between a
    score people sort by and a score people ignore.
    """

    model_config = ConfigDict(frozen=True)

    lead_id: UUID
    score: int
    band: Literal["cold", "warm", "hot"]
    contributions: tuple[ScoreContribution, ...] = ()
    computed_at: datetime

    @field_validator("score")
    @classmethod
    def _bounded(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError("score must be within [0, 100]")
        return v


class ContactWindow(BaseModel):
    """A recommended time-of-day window to reach this lead."""

    model_config = ConfigDict(frozen=True)

    weekday: int | None  # 0=Mon .. 6=Sun; None = any day
    start: time
    end: time
    channel: Channel
    confidence: float
    observations: int
    rationale: str


class EngagementProfile(BaseModel):
    """When and how a lead actually responds.

    This is the honest version of the competitor's "AI insights". It is built
    only from observed responses — inbound messages, answered calls, email
    opens — and it reports how many observations back it. With too little
    history it says so instead of inventing a window.
    """

    model_config = ConfigDict(frozen=True)

    lead_id: UUID
    windows: tuple[ContactWindow, ...] = ()
    preferred_channel: Channel | None = None
    responsiveness: float = 0.0  # answered / attempted, 0..1
    total_observations: int = 0
    sufficient_data: bool = False
    computed_at: datetime


ProposalKind = Literal["stage_change", "owner_change", "field_update", "task", "consent_change"]
ProposalStatus = Literal["pending", "accepted", "rejected", "expired", "auto_applied"]


class Proposal(BaseModel):
    """A machine-suggested change awaiting a human decision."""

    id: UUID
    lead_id: UUID
    kind: ProposalKind
    proposed: dict[str, Any]
    current: dict[str, Any] = Field(default_factory=dict)
    confidence: float
    rationale: str
    evidence: tuple[Evidence, ...] = ()
    source: str  # e.g. "finecho:call_summary", "rules:sla"
    status: ProposalStatus = "pending"
    created_at: datetime
    decided_at: datetime | None = None
    decided_by: UUID | None = None
    decision_note: str | None = None

    @field_validator("confidence")
    @classmethod
    def _bounded(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        return v


class NextBestAction(BaseModel):
    """What this counsellor should do next on this lead, and why."""

    model_config = ConfigDict(frozen=True)

    lead_id: UUID
    action: str
    channel: Channel
    priority: int  # 0 = most urgent
    reason: str
    suggested_at: datetime | None = None
    evidence: tuple[Evidence, ...] = ()


class CallIntelligence(BaseModel):
    """FinEcho's output, mapped onto CRM vocabulary.

    Callytics does not transcribe or score calls itself — FinEcho owns that and
    stays a separate service. This is the contract at that boundary.
    """

    call_id: UUID
    lead_phone: str
    agent_key: str | None = None
    occurred_at: datetime
    duration_seconds: float | None = None
    language: str | None = None
    summary: str | None = None
    points_to_remember: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    hooks: list[str] = Field(default_factory=list)
    promised_actions: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)
    detected_program: str | None = None
    call_type: str | None = None
    overall_score: float | None = None
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    compliance_flags: list[str] = Field(default_factory=list)
    recording_uri: str | None = None
    model_tag: str | None = None


class AudienceSegment(BaseModel):
    """A cohort exported to Meta/Google for re-targeting.

    Exports are hashed identifiers only, and consent is filtered before the
    hash — a lead with ``do_not_*`` set never reaches an ad platform.
    """

    id: UUID
    name: str
    filter_json: dict[str, Any]
    destination: Literal["meta", "google", "internal"]
    lead_count: int = 0
    last_synced_at: datetime | None = None
    external_audience_id: str | None = None
