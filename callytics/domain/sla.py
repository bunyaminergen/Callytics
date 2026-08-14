"""First-touch and follow-up SLAs.

A paid Facebook lead that sits untouched for a day is money already spent and
wasted, so paid sources get the tightest clock. Bulk-purchased college data
gets no first-touch SLA at all — putting a 15-minute clock on 10,000 imported
rows would only train counsellors to ignore the alert entirely.

Clocks run in business hours. A lead arriving at 22:40 is not breached at 22:55;
it is due shortly after the desk opens.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..contracts.vocabulary import BULK_SOURCES, HIGH_INTENT_SOURCES, PAID_SOURCES


@dataclass(frozen=True)
class SlaRule:
    name: str
    first_touch: timedelta | None
    follow_up: timedelta


PAID_SLA = SlaRule("paid", first_touch=timedelta(minutes=15), follow_up=timedelta(days=2))
HIGH_INTENT_SLA = SlaRule("high_intent", first_touch=timedelta(minutes=30), follow_up=timedelta(days=2))
STANDARD_SLA = SlaRule("standard", first_touch=timedelta(hours=4), follow_up=timedelta(days=3))
BULK_SLA = SlaRule("bulk", first_touch=None, follow_up=timedelta(days=7))


def rule_for(source: str) -> SlaRule:
    if source in BULK_SOURCES:
        return BULK_SLA
    if source in PAID_SOURCES:
        return PAID_SLA
    if source in HIGH_INTENT_SOURCES:
        return HIGH_INTENT_SLA
    return STANDARD_SLA


def add_business_time(
    start: datetime,
    delta: timedelta,
    *,
    day_start_hour: int = 9,
    day_end_hour: int = 20,
) -> datetime:
    """Advance ``start`` by ``delta``, counting only business hours.

    Handles the common cases the naive version gets wrong: arrival before the
    desk opens, arrival after it closes, and a delta spanning several days.
    """
    if day_end_hour <= day_start_hour:
        raise ValueError("day_end_hour must be after day_start_hour")

    cursor = start
    remaining = delta
    day_length = timedelta(hours=day_end_hour - day_start_hour)

    for _ in range(365):  # bounded; a delta beyond a year is a config error
        open_at = cursor.replace(hour=day_start_hour, minute=0, second=0, microsecond=0)
        close_at = cursor.replace(hour=day_end_hour, minute=0, second=0, microsecond=0)

        if cursor < open_at:
            cursor = open_at
        elif cursor >= close_at:
            cursor = (cursor + timedelta(days=1)).replace(
                hour=day_start_hour, minute=0, second=0, microsecond=0
            )
            continue

        available = close_at - cursor
        if remaining <= available:
            return cursor + remaining
        remaining -= available
        cursor = (cursor + timedelta(days=1)).replace(hour=day_start_hour, minute=0, second=0, microsecond=0)
        if remaining > day_length * 400:  # pragma: no cover - defensive
            break

    raise ValueError("SLA delta too large to resolve within a year of business time")


def first_touch_due(
    source: str,
    captured_at: datetime,
    *,
    day_start_hour: int = 9,
    day_end_hour: int = 20,
) -> datetime | None:
    """When first contact must have happened, or None if this source has no SLA."""
    rule = rule_for(source)
    if rule.first_touch is None:
        return None
    return add_business_time(
        captured_at,
        rule.first_touch,
        day_start_hour=day_start_hour,
        day_end_hour=day_end_hour,
    )


def is_breached(due_at: datetime | None, first_touch_at: datetime | None, now: datetime) -> bool:
    if due_at is None:
        return False
    if first_touch_at is not None:
        return first_touch_at > due_at
    return now > due_at
