"""What should this counsellor do next, and why?

The failure mode this replaces is a counsellor opening a list of 400 leads
sorted by date and working it top-down. Sorting by score alone is not much
better: it buries the lead who asked a question two hours ago beneath a
high-scoring lead who is unreachable until evening.

So actions are ranked by *urgency of the obligation*, not by lead value:

0. a promise we made on a call and have not kept,
1. an inbound message that has gone unanswered,
2. an SLA clock about to breach,
3. a scheduled task or appointment that is due,
4. a warm lead sitting inside its known contact window,
5. routine follow-up.

Every action carries the evidence behind it, so the counsellor can see why it
surfaced rather than trusting a ranking they cannot inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID

from ..contracts.intelligence import EngagementProfile, Evidence, NextBestAction
from ..contracts.lead import Consent
from ..contracts.vocabulary import Channel
from ..domain.consent import allowed_channels
from .engagement_timing import next_window_start


@dataclass
class ActionFacts:
    """Aggregated state for one lead at recommendation time."""

    lead_id: UUID
    stage: str
    score: int
    consent: Consent
    now: datetime
    unanswered_inbound_at: datetime | None = None
    open_promises: list[str] = field(default_factory=list)
    promise_made_at: datetime | None = None
    sla_due_at: datetime | None = None
    first_touch_at: datetime | None = None
    next_task_due_at: datetime | None = None
    next_task_title: str | None = None
    appointment_at: datetime | None = None
    last_contact_at: datetime | None = None
    profile: EngagementProfile | None = None
    is_terminal: bool = False


#: How long an unanswered inbound message may sit before it is the top priority.
INBOUND_GRACE = timedelta(minutes=30)
#: A lead untouched for this long is due a routine follow-up.
STALE_AFTER = timedelta(days=3)


def recommend(facts: ActionFacts, *, limit: int = 3) -> list[NextBestAction]:
    """Return the ranked actions for one lead, most urgent first."""
    if facts.is_terminal:
        return []

    channels = allowed_channels(facts.consent)
    if not channels:
        return []

    preferred = _preferred(facts, channels)
    out: list[NextBestAction] = []

    # 0 — an unkept promise. We said we would do something; that outranks
    #     everything else, including a hotter lead.
    if facts.open_promises:
        out.append(
            NextBestAction(
                lead_id=facts.lead_id,
                action=f"Deliver on promise: {facts.open_promises[0]}",
                channel=preferred,
                priority=0,
                reason="a commitment made on a previous call has not been closed out",
                suggested_at=facts.now,
                evidence=tuple(
                    Evidence(kind="promised action", detail=p, occurred_at=facts.promise_made_at)
                    for p in facts.open_promises[:3]
                ),
            )
        )

    # 1 — the lead spoke to us and got no reply.
    if facts.unanswered_inbound_at and facts.now - facts.unanswered_inbound_at > INBOUND_GRACE:
        waited = facts.now - facts.unanswered_inbound_at
        out.append(
            NextBestAction(
                lead_id=facts.lead_id,
                action="Reply to the lead's message",
                channel=Channel.WHATSAPP if Channel.WHATSAPP in channels else preferred,
                priority=1,
                reason=f"lead messaged {_humanize(waited)} ago and has had no reply",
                suggested_at=facts.now,
                evidence=(
                    Evidence(
                        kind="inbound message",
                        detail="awaiting reply",
                        occurred_at=facts.unanswered_inbound_at,
                    ),
                ),
            )
        )

    # 2 — first-touch SLA about to breach (or already breached).
    if facts.sla_due_at and facts.first_touch_at is None:
        remaining = facts.sla_due_at - facts.now
        if remaining < timedelta(hours=1):
            breached = remaining < timedelta(0)
            out.append(
                NextBestAction(
                    lead_id=facts.lead_id,
                    action="Make first contact",
                    channel=Channel.CALL if Channel.CALL in channels else preferred,
                    priority=2,
                    reason=(
                        f"first-touch SLA {'breached' if breached else 'due'} "
                        f"{_humanize(abs(remaining))} {'ago' if breached else 'from now'}"
                    ),
                    suggested_at=facts.now,
                    evidence=(Evidence(kind="sla", detail=f"due {facts.sla_due_at.isoformat()}"),),
                )
            )

    # 3 — something is on the calendar.
    if facts.appointment_at and facts.now < facts.appointment_at < facts.now + timedelta(days=1):
        out.append(
            NextBestAction(
                lead_id=facts.lead_id,
                action="Confirm tomorrow's appointment",
                channel=preferred,
                priority=3,
                reason="an appointment is scheduled within 24 hours and unconfirmed",
                suggested_at=facts.now,
                evidence=(Evidence(kind="appointment", detail=facts.appointment_at.isoformat()),),
            )
        )
    if facts.next_task_due_at and facts.next_task_due_at <= facts.now:
        out.append(
            NextBestAction(
                lead_id=facts.lead_id,
                action=facts.next_task_title or "Complete the scheduled task",
                channel=preferred,
                priority=3,
                reason="a scheduled task is due",
                suggested_at=facts.now,
                evidence=(Evidence(kind="task", detail=facts.next_task_due_at.isoformat()),),
            )
        )

    # 4 — a warm lead is inside the window where they historically respond.
    if facts.score >= 40 and facts.profile is not None and facts.profile.sufficient_data:
        window_start = next_window_start(facts.profile, facts.now)
        if window_start is not None and window_start <= facts.now + timedelta(minutes=5):
            best = facts.profile.windows[0]
            out.append(
                NextBestAction(
                    lead_id=facts.lead_id,
                    action="Call now — lead is in their responsive window",
                    channel=best.channel if best.channel in channels else preferred,
                    priority=4,
                    reason=best.rationale,
                    suggested_at=window_start,
                    evidence=(Evidence(kind="engagement window", detail=best.rationale),),
                )
            )

    # 5 — routine follow-up on a lead going cold.
    if facts.last_contact_at and facts.now - facts.last_contact_at > STALE_AFTER and not out:
        out.append(
            NextBestAction(
                lead_id=facts.lead_id,
                action="Follow up — no contact recently",
                channel=preferred,
                priority=5,
                reason=f"no contact for {_humanize(facts.now - facts.last_contact_at)}",
                suggested_at=facts.now,
                evidence=(Evidence(kind="last contact", detail=facts.last_contact_at.isoformat()),),
            )
        )

    out.sort(key=lambda a: a.priority)
    return out[:limit]


def _preferred(facts: ActionFacts, channels: list[Channel]) -> Channel:
    if facts.profile and facts.profile.preferred_channel in channels:
        return facts.profile.preferred_channel
    return channels[0]


def _humanize(delta: timedelta) -> str:
    seconds = int(abs(delta).total_seconds())
    if seconds < 3600:
        return f"{max(1, seconds // 60)}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"
