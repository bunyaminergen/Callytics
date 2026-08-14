"""Proposal review — the human decision surface for machine suggestions."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...db import repo
from ...services import proposals as proposal_service
from ..deps import Caller, get_db, require_caller, require_human

router = APIRouter(prefix="/api/proposals", tags=["proposals"])


class DecisionRequest(BaseModel):
    accept: bool
    note: str | None = None


@router.get("")
def list_pending(
    lead_id: UUID | None = None,
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db),
    _: Caller = Depends(require_caller),
) -> list[dict]:
    """Pending suggestions, newest first — the manager's review queue."""
    rows = repo.pending_proposals(db, lead_id, limit=limit)
    return [
        {
            "id": str(p.id),
            "lead_id": str(p.lead_id),
            "kind": p.kind,
            "proposed": p.proposed,
            "current": p.current,
            "confidence": float(p.confidence),
            "rationale": p.rationale,
            "evidence": p.evidence,
            "source": p.source,
            "created_at": p.created_at,
        }
        for p in rows
    ]


@router.post("/{proposal_id}/decision")
def decide(
    proposal_id: UUID,
    payload: DecisionRequest,
    db: Session = Depends(get_db),
    caller: Caller = Depends(require_human),
) -> dict:
    """Accept or reject a proposal.

    Rejections are kept, not discarded: they are the negative labels that make
    the next version of the inference better.
    """
    try:
        result = proposal_service.decide(
            db,
            proposal_id,
            accept=payload.accept,
            decided_by=caller.user_id,
            note=payload.note,
        )
    except proposal_service.ProposalError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return {
        "proposal_id": str(result.proposal_id),
        "status": result.status,
        "applied": result.applied,
        "detail": result.detail,
    }
