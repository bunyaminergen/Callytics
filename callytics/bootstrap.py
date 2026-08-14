"""Seed the configurable vocabulary and org structure.

Idempotent by design — running it against a live database only fills gaps, so
it is safe in a deploy step and safe to re-run after adding a stage or source.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .contracts.vocabulary import TEAMS
from .db.models import Base, FieldDefinition, LeadSource, LeadStage, Team, User
from .db.session import get_engine
from .integrations.leadsquared import field_definition_rows, source_rows, stage_rows
from .util import slugify, uuid7


def create_all() -> None:
    """Create tables directly. Alembic owns production; this is for dev/tests."""
    Base.metadata.create_all(get_engine())


def _upsert(session: Session, model: type, key_field: str, rows: list[dict[str, Any]]) -> int:
    """Insert rows whose key is absent. Existing rows are left untouched so
    admin edits (renaming a source, deactivating a stage) survive a re-seed."""
    existing = set(session.execute(select(getattr(model, key_field))).scalars().all())
    added = 0
    for row in rows:
        if row[key_field] in existing:
            continue
        session.add(model(**row))
        added += 1
    return added


def seed_vocabulary(session: Session) -> dict[str, int]:
    return {
        "stages": _upsert(session, LeadStage, "name", stage_rows()),
        "sources": _upsert(session, LeadSource, "name", source_rows()),
        "fields": _upsert(session, FieldDefinition, "key", field_definition_rows()),
    }


def seed_teams(session: Session, names: tuple[str, ...] = TEAMS) -> int:
    existing = set(session.execute(select(Team.name)).scalars().all())
    added = 0
    for name in names:
        if name in existing:
            continue
        session.add(Team(id=uuid7(), name=name))
        added += 1
    return added


def ensure_user(
    session: Session,
    name: str,
    *,
    role: str = "counsellor",
    email: str | None = None,
    key: str | None = None,
) -> User:
    """Get-or-create a user.

    ``key`` is the folder-safe counsellor key shared with FinEcho — it is how a
    recording under ``CallQA/<counsellor_key>/`` is attributed to a CRM user.
    """
    user_key = key or slugify(name)
    found = session.execute(select(User).where(User.key == user_key)).scalars().first()
    if found is not None:
        return found
    user = User(id=uuid7(), key=user_key, name=name, email=email, role=role)
    session.add(user)
    session.flush()
    return user


def seed_all(session: Session) -> dict[str, int]:
    counts = seed_vocabulary(session)
    counts["teams"] = seed_teams(session)
    return counts
