"""Pydantic contracts for every boundary: API, event bus, integrations."""

from __future__ import annotations

from .activity import ActivityCreate, ActivityView, CallLog, StageChange, TaskCreate, TaskView
from .envelope import EventEnvelope, make_event
from .intelligence import (
    AudienceSegment,
    CallIntelligence,
    ContactWindow,
    EngagementProfile,
    Evidence,
    LeadScore,
    NextBestAction,
    Proposal,
    ScoreContribution,
)
from .lead import Consent, LeadCreate, LeadIdentity, LeadView, MergeDecision
from .vocabulary import (
    LEAD_SOURCES,
    LEAD_STAGES,
    TEAMS,
    ActivityType,
    CallDisposition,
    Channel,
    Direction,
)

__all__ = [
    "LEAD_SOURCES",
    "LEAD_STAGES",
    "TEAMS",
    "ActivityCreate",
    "ActivityType",
    "ActivityView",
    "AudienceSegment",
    "CallDisposition",
    "CallIntelligence",
    "CallLog",
    "Channel",
    "Consent",
    "ContactWindow",
    "Direction",
    "EngagementProfile",
    "EventEnvelope",
    "Evidence",
    "LeadCreate",
    "LeadIdentity",
    "LeadScore",
    "LeadView",
    "MergeDecision",
    "NextBestAction",
    "Proposal",
    "ScoreContribution",
    "StageChange",
    "TaskCreate",
    "TaskView",
    "make_event",
]
