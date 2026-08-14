"""When is this lead actually reachable?

The competing demo pitched this as "AI insights: at what time they are active
on WhatsApp, at what time email is active". The idea is genuinely good and it
is the fix for a real, measurable problem — call connection rate. The execution
detail that matters is honesty about sample size: with three observations you
cannot claim someone "is active at 7pm", and a confident-sounding wrong window
is worse than no window, because counsellors will follow it.

So this module reports ``sufficient_data`` and an observation count alongside
every window, and refuses to emit windows below a floor. Only *responses* count
as observations — an outbound call nobody answered says nothing about when the
lead is free.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from uuid import UUID

from ..contracts.intelligence import ContactWindow, EngagementProfile
from ..contracts.vocabulary import Channel
from ..util import utcnow

#: Below this, we say so rather than guessing.
MIN_OBSERVATIONS = 5
#: An hour must hold at least this share of responses to become a window.
MIN_SHARE = 0.20
MAX_WINDOWS = 3


@dataclass(frozen=True)
class Response:
    """One observed response *from* the lead, in the account's local timezone."""

    occurred_at: datetime
    channel: Channel


@dataclass(frozen=True)
class Attempt:
    """One outbound attempt, used only for the responsiveness ratio."""

    occurred_at: datetime
    channel: Channel
    connected: bool


def _merge_adjacent(hours: list[int]) -> list[tuple[int, int]]:
    """Collapse [10, 11, 15] into [(10, 12), (15, 16)] as half-open ranges."""
    if not hours:
        return []
    ordered = sorted(set(hours))
    spans: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for hour in ordered[1:]:
        if hour == prev + 1:
            prev = hour
            continue
        spans.append((start, prev + 1))
        start = prev = hour
    spans.append((start, prev + 1))
    return spans


def build_profile(
    lead_id: UUID,
    responses: list[Response],
    attempts: list[Attempt] | None = None,
    *,
    now: datetime | None = None,
) -> EngagementProfile:
    """Derive contactable windows from observed lead responses."""
    now = now or utcnow()
    attempts = attempts or []

    total = len(responses)
    connected = sum(1 for a in attempts if a.connected)
    responsiveness = (connected / len(attempts)) if attempts else 0.0

    if total < MIN_OBSERVATIONS:
        return EngagementProfile(
            lead_id=lead_id,
            windows=(),
            preferred_channel=_preferred_channel(responses),
            responsiveness=round(responsiveness, 3),
            total_observations=total,
            sufficient_data=False,
            computed_at=now,
        )

    hour_counts = Counter(r.occurred_at.hour for r in responses)
    threshold = max(1, int(total * MIN_SHARE))
    busy_hours = [hour for hour, count in hour_counts.items() if count >= threshold]

    channel = _preferred_channel(responses)
    windows: list[ContactWindow] = []
    for start_hour, end_hour in _merge_adjacent(busy_hours):
        observations = sum(hour_counts[h] for h in range(start_hour, end_hour))
        share = observations / total
        windows.append(
            ContactWindow(
                weekday=None,
                start=time(hour=start_hour),
                end=time(hour=end_hour % 24) if end_hour < 24 else time(hour=23, minute=59),
                channel=channel or Channel.CALL,
                confidence=round(min(0.95, share * (1 - 1 / (total**0.5))), 3),
                observations=observations,
                rationale=(
                    f"{observations} of {total} responses arrived between "
                    f"{start_hour:02d}:00 and {end_hour:02d}:00"
                ),
            )
        )

    windows.sort(key=lambda w: w.observations, reverse=True)
    return EngagementProfile(
        lead_id=lead_id,
        windows=tuple(windows[:MAX_WINDOWS]),
        preferred_channel=channel,
        responsiveness=round(responsiveness, 3),
        total_observations=total,
        sufficient_data=True,
        computed_at=now,
    )


def _preferred_channel(responses: list[Response]) -> Channel | None:
    if not responses:
        return None
    counts = Counter(r.channel for r in responses)
    return counts.most_common(1)[0][0]


def next_window_start(profile: EngagementProfile, after: datetime) -> datetime | None:
    """The next datetime falling inside the lead's best window, if known.

    Returns ``after`` itself when it is already inside the window, so a
    counsellor acting on the suggestion right now is not told to wait a day.
    """
    if not profile.sufficient_data or not profile.windows:
        return None

    best = profile.windows[0]
    end_hour = 24 if best.end.hour == 23 and best.end.minute == 59 else best.end.hour
    if best.start.hour <= after.hour < end_hour:
        return after

    candidate = after.replace(hour=best.start.hour, minute=0, second=0, microsecond=0)
    if candidate <= after:
        candidate += timedelta(days=1)
    return candidate
