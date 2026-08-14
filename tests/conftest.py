"""Test fixtures.

The whole suite runs against in-memory SQLite with no Kafka, no MySQL and no
FinEcho process. That is deliberate: the contract between layers is what these
tests protect, and requiring infrastructure to run them means they stop being
run.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("CALLYTICS_MYSQL_DSN", "sqlite://")
os.environ.setdefault("CALLYTICS_API_TOKEN", "test-token")
os.environ.setdefault("CALLYTICS_FINECHO_WEBHOOK_SECRET", "test-secret")
os.environ.setdefault("CALLYTICS_BROKER", "memory")

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from callytics import bootstrap  # noqa: E402
from callytics.db import models  # noqa: E402
from callytics.db import session as db_session


@pytest.fixture
def engine():
    """One in-memory database per test, shared across connections."""
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    models.Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def db(engine, monkeypatch) -> Iterator[Session]:
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    monkeypatch.setattr(db_session, "get_engine", lambda: engine)
    monkeypatch.setattr(db_session, "get_sessionmaker", lambda: factory)

    session = factory()
    bootstrap.seed_all(session)
    session.commit()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def counsellor(db):
    user = bootstrap.ensure_user(db, "Anju", role="counsellor", key="anju")
    db.commit()
    return user


@pytest.fixture
def manager(db):
    user = bootstrap.ensure_user(db, "Riswan", role="manager", key="riswan")
    db.commit()
    return user


@pytest.fixture
def client(db, engine, monkeypatch):
    """TestClient wired to the same in-memory database as ``db``."""
    from fastapi.testclient import TestClient

    from callytics.api import deps
    from callytics.api.app import create_app

    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    def _get_db() -> Iterator[Session]:
        s = factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    app = create_app()
    app.dependency_overrides[deps.get_db] = _get_db
    return TestClient(app)


@pytest.fixture
def auth():
    return {"Authorization": "Bearer test-token"}
