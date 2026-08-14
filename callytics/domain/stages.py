"""Stage machine.

LeadSquared lets any stage move to any other stage, which is why the live
account has leads sitting in "JOINED" that later became "Not interested". We
keep the same stage names for continuity but add the guard rails that were
missing: terminal stages need an explicit reopen, and a revenue stage cannot be
entered by an automated actor.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.vocabulary import (
    ARTEFACT_STAGES,
    CLOSED_LOST_STAGES,
    CLOSED_WON_STAGES,
    LEAD_STAGES,
    REVENUE_STAGES,
)


class InvalidStage(ValueError):
    pass


@dataclass(frozen=True)
class TransitionResult:
    allowed: bool
    reason: str
    requires_reason: bool = False


#: Entering these stages materially affects reported revenue, so a human must
#: do it. An automated proposal may suggest them; it may never apply them.
HUMAN_ONLY_STAGES: frozenset[str] = REVENUE_STAGES | {"Confirmed - Waiting for payment"}

#: Leaving a terminal stage is legitimate (a lead does come back) but it is an
#: exception that must be justified, not a silent overwrite.
TERMINAL_STAGES: frozenset[str] = CLOSED_LOST_STAGES | CLOSED_WON_STAGES


def validate(stage: str) -> str:
    if stage not in LEAD_STAGES:
        raise InvalidStage(f"unknown stage {stage!r}")
    return stage


def can_transition(from_stage: str | None, to_stage: str, *, actor_is_human: bool) -> TransitionResult:
    """Decide whether a stage change is permitted.

    ``from_stage`` is None for a lead being created.
    """
    validate(to_stage)
    if from_stage is not None:
        validate(from_stage)

    if from_stage == to_stage:
        return TransitionResult(allowed=False, reason="lead is already in this stage")

    if to_stage in ARTEFACT_STAGES:
        return TransitionResult(allowed=False, reason=f"{to_stage!r} is an import artefact, not a real stage")

    if to_stage in HUMAN_ONLY_STAGES and not actor_is_human:
        return TransitionResult(
            allowed=False,
            reason=f"{to_stage!r} affects reported revenue and must be set by a person",
        )

    if from_stage in TERMINAL_STAGES:
        return TransitionResult(
            allowed=True,
            reason=f"reopening a lead from terminal stage {from_stage!r}",
            requires_reason=True,
        )

    return TransitionResult(allowed=True, reason="permitted")


def is_terminal(stage: str) -> bool:
    return stage in TERMINAL_STAGES


def stage_position(stage: str) -> int:
    """Funnel position, used by scoring and by the pipeline chart ordering."""
    try:
        return LEAD_STAGES.index(stage)
    except ValueError as exc:  # pragma: no cover - guarded by validate()
        raise InvalidStage(f"unknown stage {stage!r}") from exc
