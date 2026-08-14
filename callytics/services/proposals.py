"""Accepting and rejecting machine proposals.

The decision is the product. Accepting applies the change through the same
guarded path a manual edit uses — a proposal is not a backdoor around the stage
machine. Rejecting is equally valuable: it is a labelled negative example, and
it is why the rejection note is captured rather than discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from ..contracts.activity import StageChange
from ..contracts.vocabulary import ActivityType, Channel, Direction
from ..db import repo
from ..db.models import Proposal
from ..services import leads as lead_service
from ..util import utcnow


class ProposalError(RuntimeError):
    pass


@dataclass
class DecisionResult:
    proposal_id: UUID
    status: str
    applied: bool
    detail: str


def decide(
    session: Session,
    proposal_id: UUID,
    *,
    accept: bool,
    decided_by: UUID | None,
    note: str | None = None,
) -> DecisionResult:
    proposal = session.get(Proposal, proposal_id)
    if proposal is None:
        raise ProposalError(f"proposal {proposal_id} not found")
    if proposal.status != "pending":
        raise ProposalError(f"proposal is already {proposal.status}")

    proposal.decided_at = utcnow()
    proposal.decided_by = decided_by
    proposal.decision_note = note

    if not accept:
        proposal.status = "rejected"
        _log(session, proposal, "rejected", decided_by, note)
        return DecisionResult(proposal_id, "rejected", False, note or "rejected by user")

    lead = repo.get_lead(session, proposal.lead_id)
    if lead is None:
        raise ProposalError(f"lead {proposal.lead_id} not found")

    applied_detail = ""
    if proposal.kind == "stage_change":
        to_stage = proposal.proposed.get("stage")
        if not to_stage:
            raise ProposalError("stage_change proposal has no target stage")
        # actor_is_human=True: a person is accepting it, which is exactly the
        # accountability the guard rail is asking for.
        lead_service.change_stage(
            session,
            lead,
            StageChange(
                from_stage=lead.stage,
                to_stage=to_stage,
                changed_by=decided_by,
                reason=f"accepted proposal: {proposal.rationale}"[:200],
                proposal_id=proposal.id,
            ),
            actor_is_human=True,
        )
        applied_detail = f"stage set to {to_stage}"
    elif proposal.kind == "field_update":
        updates = {k: v for k, v in proposal.proposed.items() if v is not None}
        lead.fields = {**(lead.fields or {}), **updates}
        applied_detail = f"updated {', '.join(sorted(updates))}"
    elif proposal.kind == "owner_change":
        lead.owner_id = UUID(str(proposal.proposed["owner_id"]))
        applied_detail = "owner reassigned"
    else:
        raise ProposalError(f"cannot apply proposal kind {proposal.kind!r}")

    proposal.status = "accepted"
    _log(session, proposal, "accepted", decided_by, note)
    return DecisionResult(proposal_id, "accepted", True, applied_detail)


def _log(session: Session, proposal: Proposal, status: str, actor: UUID | None, note: str | None) -> None:
    repo.record_activity(
        session,
        lead_id=proposal.lead_id,
        type=ActivityType.PROPOSAL_DECIDED.value,
        channel=Channel.SYSTEM.value,
        direction=Direction.INTERNAL.value,
        occurred_at=utcnow(),
        actor_id=actor,
        subject=f"Proposal {status}: {proposal.kind}",
        body=note,
        payload={
            "proposal_id": str(proposal.id),
            "kind": proposal.kind,
            "proposed": proposal.proposed,
            "confidence": float(proposal.confidence),
            "source": proposal.source,
            "status": status,
        },
    )
    repo.audit(
        session,
        actor_id=actor,
        actor_label=str(actor) if actor else "system",
        action=f"proposal.{status}",
        target_type="proposal",
        target_id=str(proposal.id),
        detail={"lead_id": str(proposal.lead_id), "kind": proposal.kind},
    )
