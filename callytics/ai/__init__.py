"""Intelligence layer.

Everything here follows one rule: **propose with evidence, never silently
mutate.** Each module turns observed facts into a suggestion a human can
inspect, accept or reject — and the decision is recorded so the suggestions
get better.
"""

from __future__ import annotations

from . import engagement_timing, next_best_action, stage_inference

__all__ = ["engagement_timing", "next_best_action", "stage_inference"]
