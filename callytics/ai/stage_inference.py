"""Infer stage changes from call intelligence — as proposals, never as writes.

Context for why this module is shaped the way it is. In the competitor demo,
the prospect asked the obvious question: after I finish a call, do I change the
stage by hand, or does the AI read my call and change it for me? The vendor's
answer was that they had built the automatic version, but recommended against
it — *"doing that, we will never have that trust"*. That is an accurate read of
the risk and a bad resolution of it. Silently rewriting a manager's pipeline
from a model's reading of a phone call destroys trust the first time it is
wrong, and "make the human do it all manually" throws away the value.

Callytics takes the third option. Every inference becomes a :class:`Proposal`
carrying:

* the exact change requested (from-stage → to-stage),
* a confidence derived from how unambiguous the signal was,
* **evidence** — the specific FinEcho fields that triggered it, quoted, so a
  counsellor can verify in two seconds without opening the recording,
* a one-click accept/reject that is recorded.

Nothing is applied automatically unless an administrator explicitly enables
auto-apply *and* the confidence clears a configured floor, and even then
revenue-affecting stages stay human-only (see ``domain.stages``). The accepted
and rejected decisions accumulate into exactly the labelled dataset needed to
make later versions better — the manual-only approach never produces that.

Inference runs over FinEcho's *structured* output (objections, promised actions,
call type, disposition), not over raw transcript text. Structured input keeps
this module deterministic and unit-testable, and keeps the LLM's judgement
inside the service that owns it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from ..contracts.intelligence import CallIntelligence, Evidence
from ..contracts.vocabulary import CONNECTED_DISPOSITIONS, CallDisposition
from ..domain.stages import HUMAN_ONLY_STAGES, can_transition
from ..util import utcnow, uuid7

SOURCE = "finecho:call_intelligence"


@dataclass(frozen=True)
class Signal:
    """One matched pattern and what it implies."""

    to_stage: str
    confidence: float
    rationale: str
    evidence: tuple[Evidence, ...]


# Patterns are matched against FinEcho's structured lists. Each entry is
# (compiled pattern, target stage, base confidence, human-readable label).
_COMMITMENT = re.compile(
    r"\b(pay(ment|ing)?|fee(s)? transfer|enrol|enroll|admission|register(ed)?|book(ed)? (the )?seat|"
    r"join(ing)? (on|from|next)|advance)\b",
    re.I,
)
_CALLBACK = re.compile(
    r"\b(call (me )?(back|later|tomorrow|after)|ring (me )?(back|later)|revert|get back to (me|you)|"
    r"discuss with (my )?(parent|father|mother|family|husband|wife)|check and (revert|confirm))\b",
    re.I,
)
_REJECTION = re.compile(
    r"\b(not interested|no longer interested|don'?t want|already (joined|enrolled|admitted)|"
    r"joined (another|other|elsewhere)|chose another|going with (another|different))\b",
    re.I,
)
_DEFERRAL = re.compile(
    r"\b(next (year|intake|batch|season)|after (my )?(exam|result|graduation)|later this year|"
    r"maybe (next|later)|not (right )?now)\b",
    re.I,
)
_COMPETITOR = re.compile(r"\b(competitor|another institute|other institute|cheaper elsewhere)\b", re.I)


def _scan(text: str, kind: str) -> list[tuple[str, float, str, Evidence]]:
    """Return candidate (stage, confidence, rationale, evidence) for one string."""
    out: list[tuple[str, float, str, Evidence]] = []
    checks = (
        (_REJECTION, "Not interested", 0.82, "explicit rejection"),
        (_COMPETITOR, "Lost to competitor", 0.78, "competitor mentioned"),
        (_COMMITMENT, "Confirmed - Waiting for payment", 0.74, "commitment to pay or enrol"),
        (_DEFERRAL, "Maybe in future", 0.70, "deferred to a future intake"),
        (_CALLBACK, "Call back later", 0.68, "callback requested"),
    )
    for pattern, stage, confidence, label in checks:
        match = pattern.search(text)
        if match:
            out.append(
                (
                    stage,
                    confidence,
                    f"{label} in {kind}",
                    Evidence(kind=kind, detail=text.strip()[:300]),
                )
            )
    return out


def infer(
    intel: CallIntelligence,
    current_stage: str,
    *,
    disposition: CallDisposition | None = None,
    lead_id: UUID | None = None,
) -> Signal | None:
    """Derive the single best-supported stage change, or None.

    Returns at most one signal: raising three competing proposals against one
    call is noise, and the counsellor would have to adjudicate them anyway.
    """
    # An unanswered call is a disposition fact, not a content inference. It is
    # handled here because it is the most common outcome by far.
    if disposition is not None and disposition not in CONNECTED_DISPOSITIONS:
        mapped = {
            CallDisposition.NO_ANSWER: "Did Not Pick",
            CallDisposition.BUSY: "Did Not Pick",
            CallDisposition.SWITCHED_OFF: "Not able to contact",
            CallDisposition.INVALID_NUMBER: "Junk leads",
            CallDisposition.WRONG_NUMBER: "Junk leads",
        }.get(disposition)
        if mapped and mapped != current_stage:
            return Signal(
                to_stage=mapped,
                confidence=0.90,
                rationale=f"call disposition reported as {disposition.value}",
                evidence=(
                    Evidence(kind="disposition", detail=disposition.value, occurred_at=intel.occurred_at),
                ),
            )
        return None

    candidates: list[tuple[str, float, str, Evidence]] = []
    for objection in intel.objections:
        candidates += _scan(objection, "objection")
    for promised in intel.promised_actions:
        candidates += _scan(promised, "promised action")
    for step in intel.suggested_next_steps:
        candidates += _scan(step, "suggested next step")
    for point in intel.points_to_remember:
        candidates += _scan(point, "point to remember")
    if intel.summary:
        candidates += _scan(intel.summary, "call summary")

    if not candidates:
        return None

    # Group by target stage; corroboration across independent fields raises
    # confidence, but never to certainty.
    best_stage = max(
        {c[0] for c in candidates},
        key=lambda s: (
            max(c[1] for c in candidates if c[0] == s),
            sum(1 for c in candidates if c[0] == s),
        ),
    )
    matching = [c for c in candidates if c[0] == best_stage]
    base = max(c[1] for c in matching)
    corroboration = min(0.10, 0.03 * (len(matching) - 1))
    confidence = round(min(0.95, base + corroboration), 3)

    if best_stage == current_stage:
        return None

    transition = can_transition(current_stage, best_stage, actor_is_human=True)
    if not transition.allowed:
        return None

    rationale = "; ".join(dict.fromkeys(c[2] for c in matching))
    evidence = tuple(dict.fromkeys(c[3] for c in matching))[:4]
    return Signal(to_stage=best_stage, confidence=confidence, rationale=rationale, evidence=evidence)


def to_proposal(
    signal: Signal,
    lead_id: UUID,
    current_stage: str,
    *,
    call_id: UUID | None = None,
) -> dict:
    """Build the persistable proposal row for a signal.

    ``dedupe_key`` binds the proposal to the call that produced it, so a
    redelivered FinEcho webhook cannot raise the same suggestion twice.
    """
    return {
        "id": uuid7(),
        "lead_id": lead_id,
        "kind": "stage_change",
        "proposed": {"stage": signal.to_stage},
        "current": {"stage": current_stage},
        "confidence": signal.confidence,
        "rationale": signal.rationale,
        "evidence": [e.model_dump(mode="json") for e in signal.evidence],
        "source": SOURCE,
        "status": "pending",
        "dedupe_key": f"{SOURCE}:{call_id}:{signal.to_stage}" if call_id else None,
        "created_at": utcnow(),
    }


def may_auto_apply(signal: Signal, *, enabled: bool, min_confidence: float) -> bool:
    """Auto-apply is opt-in, floor-gated, and never touches revenue stages."""
    if not enabled:
        return False
    if signal.to_stage in HUMAN_ONLY_STAGES:
        return False
    return signal.confidence >= min_confidence
