"""Inbound webhooks. Every one of these is authenticated and idempotent."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from ...integrations import finecho
from ...services import intelligence as intel_service
from ...settings import get_settings
from ..deps import get_db

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/finecho/call", status_code=status.HTTP_202_ACCEPTED)
async def finecho_call(
    request: Request,
    x_finecho_signature: str | None = Header(default=None, alias="X-FinEcho-Signature"),
    db: Session = Depends(get_db),
) -> dict:
    """Receive a completed call summary/evaluation from FinEcho.

    Verified by HMAC over the raw body — this endpoint writes to a lead's
    timeline, so an unauthenticated version would let anyone forge call history.
    """
    settings = get_settings()
    raw = await request.body()

    if not settings.finecho_webhook_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CALLYTICS_FINECHO_WEBHOOK_SECRET is not configured",
        )
    if not finecho.verify_signature(raw, x_finecho_signature or "", settings.finecho_webhook_secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad signature")

    try:
        payload = await request.json()
        intel = finecho.parse_webhook(payload)
    except finecho.FinEchoError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"malformed payload: {exc}") from exc

    result = intel_service.ingest(db, intel)

    # An unmatched call is reported, not swallowed: it is usually a personal
    # call or a lead that should exist. 202 either way — FinEcho must not retry
    # a call we have correctly decided not to store.
    return {
        "matched": result.matched,
        "lead_id": str(result.lead_id) if result.lead_id else None,
        "proposal_id": str(result.proposal_id) if result.proposal_id else None,
        "auto_applied": result.auto_applied,
        "detail": result.reason,
    }
