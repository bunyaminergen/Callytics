"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..settings import get_settings
from .routers import leads, proposals, webhooks


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Callytics",
        version="0.1.0",
        description=(
            "AI-native CRM for education sales. Conversation intelligence is provided "
            "by FinEcho across a service boundary; this API owns leads, timeline, "
            "pipeline and the proposal review surface."
        ),
    )

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(leads.router)
    app.include_router(proposals.router)
    app.include_router(webhooks.router)

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok", "env": settings.env}

    return app


app = create_app()
