"""Explainable lead scoring.

Design constraint: a counsellor must be able to ask "why is this lead a 78?"
and get an answer they can argue with. So the score is a bounded sum of named
contributions, each with a human-readable detail string, rather than a model
output. That buys three things a black-box score does not:

* it is testable — every branch below has a unit test;
* it is tunable by the sales head without a retraining cycle;
* it produces the labelled features a learned model would later need.

Recency uses exponential decay with a configurable half-life, because a lead
who engaged twice last week is worth more than one who engaged ten times last
year, and a linear window makes that flip abruptly at the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from ..contracts.intelligence import ScoreContribution
from ..contracts.vocabulary import BULK_SOURCES, HIGH_INTENT_SOURCES, PAID_SOURCES
from .stages import stage_position

SCORE_MODEL_VERSION = "rules/v1"

BAND_WARM_MIN = 40
BAND_HOT_MIN = 70


@dataclass
class ScoringFacts:
    """Everything the scorer needs, already aggregated by the repository.

    Passing a flat fact object rather than an ORM row keeps this module pure and
    makes the unit tests readable.
    """

    stage: str
    source: str
    created_at: datetime
    now: datetime
    connected_calls: int = 0
    call_attempts: int = 0
    inbound_messages: int = 0
    email_opens: int = 0
    last_engagement_at: datetime | None = None
    appointments_kept: int = 0
    appointments_missed: int = 0
    filled_field_count: int = 0
    total_field_count: int = 0
    latest_call_score: float | None = None
    has_objection: bool = False
    is_suppressed: bool = False
    contributions: list[ScoreContribution] = field(default_factory=list)


def _decay(days: float, half_life_days: float) -> float:
    """1.0 at zero days, 0.5 at one half-life, asymptotic to 0."""
    if days <= 0:
        return 1.0
    return math.pow(0.5, days / max(half_life_days, 0.5))


def compute(facts: ScoringFacts, *, half_life_days: float = 14.0) -> tuple[int, str, list[ScoreContribution]]:
    """Return ``(score, band, contributions)``."""
    contributions: list[ScoreContribution] = []

    def add(feature: str, points: float, detail: str) -> None:
        if points:
            contributions.append(ScoreContribution(feature=feature, points=round(points, 2), detail=detail))

    # --- source quality (max 15) -----------------------------------------
    if facts.source in HIGH_INTENT_SOURCES:
        add("source", 15, f"{facts.source} is a high-intent source")
    elif facts.source in BULK_SOURCES:
        add("source", -5, f"{facts.source} is bulk-acquired data")
    elif facts.source in PAID_SOURCES:
        add("source", 5, f"{facts.source} is a paid campaign")

    # --- funnel position (max 25) ----------------------------------------
    # Stages are ordered by the funnel, so position is a usable proxy for
    # progress. Capped so that stage alone never dominates behaviour.
    position = stage_position(facts.stage)
    if position <= 8:  # Open .. Already joined, i.e. the live part of the funnel
        add("stage", min(25.0, position * 3.0), f"stage {facts.stage!r} is at funnel position {position}")

    # --- two-way engagement (max 30) -------------------------------------
    if facts.connected_calls:
        add("connected_calls", min(15.0, facts.connected_calls * 5.0), f"{facts.connected_calls} connected call(s)")
    if facts.inbound_messages:
        add(
            "inbound_messages",
            min(10.0, facts.inbound_messages * 2.5),
            f"{facts.inbound_messages} inbound message(s) from the lead",
        )
    if facts.email_opens:
        add("email_opens", min(5.0, facts.email_opens * 1.0), f"{facts.email_opens} email open(s)")

    # --- recency multiplier on engagement --------------------------------
    if facts.last_engagement_at is not None:
        days = (facts.now - facts.last_engagement_at).total_seconds() / 86400.0
        factor = _decay(days, half_life_days)
        engaged_points = sum(
            c.points for c in contributions if c.feature in {"connected_calls", "inbound_messages", "email_opens"}
        )
        penalty = -engaged_points * (1.0 - factor)
        add("recency", penalty, f"last engagement {days:.0f}d ago (decay factor {factor:.2f})")
    elif facts.call_attempts:
        add("recency", -5, "no successful engagement despite outbound attempts")

    # --- commitment signals (max 15) -------------------------------------
    if facts.appointments_kept:
        add("appointments", min(15.0, facts.appointments_kept * 7.5), f"{facts.appointments_kept} appointment(s) kept")
    if facts.appointments_missed:
        add("appointments_missed", -facts.appointments_missed * 5.0, f"{facts.appointments_missed} no-show(s)")

    # --- profile completeness (max 5) ------------------------------------
    if facts.total_field_count:
        ratio = facts.filled_field_count / facts.total_field_count
        add("profile", ratio * 5.0, f"{facts.filled_field_count}/{facts.total_field_count} qualifying fields filled")

    # --- conversation quality from FinEcho (max 10) ----------------------
    if facts.latest_call_score is not None:
        # FinEcho scores 0-10 against the sales rubric. A strong call is a
        # genuine buying signal; a weak one is a coaching signal, not a lead
        # signal, so the downside is deliberately smaller than the upside.
        delta = (facts.latest_call_score - 5.0) / 5.0
        add(
            "call_quality",
            10.0 * delta if delta > 0 else 4.0 * delta,
            f"last call scored {facts.latest_call_score}/10",
        )

    # --- hard negatives ---------------------------------------------------
    if facts.has_objection:
        add("objection", -8, "an unresolved objection was raised on the last call")
    if facts.is_suppressed:
        add("suppressed", -25, "lead has suppressed one or more contact channels")
    if facts.call_attempts >= 5 and facts.connected_calls == 0:
        add("unreachable", -10, f"{facts.call_attempts} attempts with no connect")

    total = sum(c.points for c in contributions)
    score = max(0, min(100, round(total)))
    band = "hot" if score >= BAND_HOT_MIN else "warm" if score >= BAND_WARM_MIN else "cold"
    return score, band, contributions


def band_for(score: int) -> str:
    return "hot" if score >= BAND_HOT_MIN else "warm" if score >= BAND_WARM_MIN else "cold"
