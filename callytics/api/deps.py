"""Shared FastAPI dependencies: DB session and caller authentication."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import User
from ..db.session import get_sessionmaker
from ..settings import get_settings


def get_db() -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@dataclass(frozen=True)
class Caller:
    user_id: UUID | None
    label: str
    is_service: bool


def require_caller(
    authorization: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> Caller:
    """Authenticate a service-to-service caller and resolve the acting user.

    This is a shared-secret gate suitable for calls originating inside the
    cluster. Interactive users authenticate at the portal, which then calls
    this API with the service token plus ``X-User-Id`` for attribution — every
    write is still recorded against a real person.
    """
    settings = get_settings()
    if not settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CALLYTICS_API_TOKEN is not configured; refusing to serve unauthenticated requests",
        )

    provided = (authorization or "").removeprefix("Bearer ").strip()
    if provided != settings.api_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing token")

    if not x_user_id:
        return Caller(user_id=None, label="service", is_service=True)

    try:
        user_uuid = UUID(x_user_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="X-User-Id is not a UUID") from exc

    user = db.execute(select(User).where(User.id == user_uuid)).scalars().first()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="unknown or inactive user")
    return Caller(user_id=user.id, label=user.name, is_service=False)


def require_human(caller: Caller = Depends(require_caller)) -> Caller:
    """For actions that must be attributable to a person, not a service token."""
    if caller.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="this action requires an acting user (send X-User-Id)",
        )
    return caller
