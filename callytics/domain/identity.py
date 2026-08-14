"""Identity resolution: decide whether an inbound capture is a person we know.

Duplicate leads are the single most expensive data defect in this business. Two
counsellors working the same student is wasted spend and an embarrassing call;
worse, a duplicate splits the timeline so neither counsellor sees the full
history. The competing product's answer was a nightly duplicate report. Ours is
to resolve identity at write time, before the row exists.

The policy is deliberately conservative: automatic merges happen only on an
exact contactable identifier (phone or email). Weaker signals (same name, same
campaign, same day) raise a review rather than silently joining two people —
merging two real students is far more damaging than carrying a duplicate for a
day, because a merge is not cleanly reversible.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.lead import LeadIdentity, MergeDecision
from ..util import normalize_name


@dataclass(frozen=True)
class ExistingLead:
    """The projection of an existing lead needed to compare identity."""

    lead_id: object  # UUID, kept loose so this module stays import-light
    phones: tuple[str, ...]
    emails: tuple[str, ...]
    full_name: str | None
    is_merged: bool = False


# Confidence levels. Named so tests and docs reference the same constants.
CONF_EXACT_PHONE = 0.99
CONF_EXACT_EMAIL = 0.95
CONF_NAME_ONLY = 0.55
AUTO_ATTACH_THRESHOLD = 0.90


def resolve(incoming: LeadIdentity, candidates: list[ExistingLead]) -> MergeDecision:
    """Choose between attaching to an existing lead, creating, or review.

    ``candidates`` should already be narrowed by the repository to leads sharing
    at least one identifier or a normalised name — this function does not query.
    """
    live = [c for c in candidates if not c.is_merged]
    if not live:
        return MergeDecision(
            matched_lead_id=None,
            confidence=1.0,
            reason="no candidate shares an identifier",
            action="create",
        )

    incoming_phones = set(incoming.phones)
    incoming_email = incoming.email

    # 1. Exact phone match — the strongest signal we have.
    for candidate in live:
        overlap = incoming_phones.intersection(candidate.phones)
        if overlap:
            return MergeDecision(
                matched_lead_id=candidate.lead_id,  # type: ignore[arg-type]
                confidence=CONF_EXACT_PHONE,
                reason=f"phone {sorted(overlap)[0]} already belongs to this lead",
                action="attach",
            )

    # 2. Exact email match.
    if incoming_email:
        for candidate in live:
            if incoming_email in candidate.emails:
                return MergeDecision(
                    matched_lead_id=candidate.lead_id,  # type: ignore[arg-type]
                    confidence=CONF_EXACT_EMAIL,
                    reason=f"email {incoming_email} already belongs to this lead",
                    action="attach",
                )

    # 3. Name-only overlap. Never automatic: "Amal" is not an identifier in a
    #    dataset where hundreds of leads share a first name.
    incoming_name = normalize_name(incoming.full_name)
    if incoming_name:
        for candidate in live:
            if normalize_name(candidate.full_name) == incoming_name:
                return MergeDecision(
                    matched_lead_id=candidate.lead_id,  # type: ignore[arg-type]
                    confidence=CONF_NAME_ONLY,
                    reason=f"name matches an existing lead ({incoming.full_name}) but no shared identifier",
                    action="review",
                )

    return MergeDecision(
        matched_lead_id=None,
        confidence=1.0,
        reason="candidates share no identifier with the incoming record",
        action="create",
    )


def should_auto_attach(decision: MergeDecision) -> bool:
    return decision.action == "attach" and decision.confidence >= AUTO_ATTACH_THRESHOLD
