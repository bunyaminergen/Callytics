"""Lead, identity and consent contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..util import normalize_email, normalize_phone


class Consent(BaseModel):
    """Contactability flags.

    These are hard gates, not preferences: every outbound channel checks them
    before sending, and the check lives in one place (``domain.consent``) so a
    new channel cannot forget it. DPDP Act consent is auditable, so each flag
    change lands on the timeline.
    """

    model_config = ConfigDict(frozen=True)

    do_not_call: bool = False
    do_not_sms: bool = False
    do_not_email: bool = False
    do_not_whatsapp: bool = False
    whatsapp_opt_in_at: datetime | None = None
    consent_source: str | None = None
    consent_timestamp: datetime | None = None


class LeadIdentity(BaseModel):
    """The subset of fields used to decide whether two records are one person."""

    model_config = ConfigDict(frozen=True)

    phone: str | None = None
    alternate_phone: str | None = None
    email: str | None = None
    full_name: str | None = None

    @field_validator("phone", "alternate_phone", mode="before")
    @classmethod
    def _norm_phone(cls, v: str | None) -> str | None:
        return normalize_phone(v) if v else None

    @field_validator("email", mode="before")
    @classmethod
    def _norm_email(cls, v: str | None) -> str | None:
        return normalize_email(v) if v else None

    @property
    def phones(self) -> list[str]:
        return [p for p in (self.phone, self.alternate_phone) if p]


class LeadCreate(BaseModel):
    """Inbound payload for lead capture (web form, ad webhook, import, walk-in)."""

    identity: LeadIdentity
    source: str
    source_campaign: str | None = None
    source_medium: str | None = None
    source_referrer: str | None = None
    stage: str | None = None
    owner_id: UUID | None = None
    team: str | None = None
    consent: Consent = Field(default_factory=Consent)
    fields: dict[str, Any] = Field(default_factory=dict)
    external_ids: dict[str, str] = Field(default_factory=dict)
    captured_at: datetime | None = None

    @field_validator("fields")
    @classmethod
    def _reject_empty_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        if any(not k or not k.strip() for k in v):
            raise ValueError("custom field keys must be non-empty")
        return v


class LeadView(BaseModel):
    """What the counsellor screen and the API return for a single lead."""

    id: UUID
    identity: LeadIdentity
    stage: str
    source: str
    owner_id: UUID | None
    owner_name: str | None
    team: str | None
    consent: Consent
    fields: dict[str, Any]
    score: int
    score_band: str
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime | None
    first_touch_at: datetime | None
    touched: bool
    external_ids: dict[str, str] = Field(default_factory=dict)


class MergeDecision(BaseModel):
    """Result of identity resolution when a capture matches existing leads."""

    model_config = ConfigDict(frozen=True)

    matched_lead_id: UUID | None
    confidence: float
    reason: str
    action: str  # attach|merge|create|review

    @field_validator("confidence")
    @classmethod
    def _bounded(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        return v
