"""Lead assignment.

The live account routes by team (Calicut, TVM, Bangalore, HO CC FA/DM, online),
and a lead's language and location decide which team can actually serve it —
assigning a Malayalam-only walk-in enquiry to the Bangalore desk wastes both
sides' time. Routing therefore filters for eligibility first and only then
balances load.

Load balancing is least-open-leads rather than round-robin: round-robin keeps
handing work to whoever is already drowning, which is how leads rot untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """A counsellor available to receive work."""

    user_id: object
    name: str
    teams: frozenset[str]
    languages: frozenset[str] = frozenset()
    open_leads: int = 0
    capacity: int = 150
    is_available: bool = True

    @property
    def headroom(self) -> int:
        return self.capacity - self.open_leads


@dataclass(frozen=True)
class RoutingRequest:
    source: str
    language: str | None = None
    district: str | None = None
    course_category: str | None = None
    preferred_team: str | None = None
    excluded_user_ids: frozenset[object] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RoutingDecision:
    user_id: object | None
    reason: str
    considered: int


def choose(request: RoutingRequest, candidates: list[Candidate]) -> RoutingDecision:
    """Pick the counsellor who should own this lead."""
    pool = [c for c in candidates if c.is_available and c.user_id not in request.excluded_user_ids]
    considered = len(pool)
    if not pool:
        return RoutingDecision(user_id=None, reason="no available counsellor", considered=0)

    if request.preferred_team:
        team_pool = [c for c in pool if request.preferred_team in c.teams]
        if team_pool:
            pool = team_pool

    if request.language:
        language_pool = [c for c in pool if not c.languages or request.language in c.languages]
        if language_pool:
            pool = language_pool
        else:
            return RoutingDecision(
                user_id=None,
                reason=f"no counsellor speaks {request.language}",
                considered=considered,
            )

    with_headroom = [c for c in pool if c.headroom > 0]
    if not with_headroom:
        # Everyone is at capacity. Returning None is correct: the lead goes to
        # an unassigned queue a manager can see, rather than being buried in
        # someone's overflowing list.
        return RoutingDecision(user_id=None, reason="all eligible counsellors are at capacity", considered=considered)

    # Deterministic: fewest open leads, ties broken by name so tests are stable.
    winner = min(with_headroom, key=lambda c: (c.open_leads, c.name))
    return RoutingDecision(
        user_id=winner.user_id,
        reason=f"{winner.name} has the fewest open leads ({winner.open_leads}) among {len(with_headroom)} eligible",
        considered=considered,
    )
