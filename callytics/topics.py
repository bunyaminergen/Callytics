"""Topic names. Import these instead of writing string literals."""

from __future__ import annotations

from .settings import get_settings


def _t(name: str) -> str:
    return f"{get_settings().topic_prefix}.{name}"


def leads_captured() -> str:
    return _t("leads.captured.v1")


def leads_stage_changed() -> str:
    return _t("leads.stage_changed.v1")


def leads_owner_changed() -> str:
    return _t("leads.owner_changed.v1")


def activities_recorded() -> str:
    return _t("activities.recorded.v1")


def call_intelligence() -> str:
    return _t("calls.intelligence.v1")


def proposals_raised() -> str:
    return _t("proposals.raised.v1")


def proposals_decided() -> str:
    return _t("proposals.decided.v1")


def scores_recomputed() -> str:
    return _t("scores.recomputed.v1")


def sla_breached() -> str:
    return _t("sla.breached.v1")


def audience_sync() -> str:
    return _t("audience.sync.v1")


def notifications() -> str:
    return _t("notifications.v1")


def all_topics() -> list[str]:
    return [
        leads_captured(),
        leads_stage_changed(),
        leads_owner_changed(),
        activities_recorded(),
        call_intelligence(),
        proposals_raised(),
        proposals_decided(),
        scores_recomputed(),
        sla_breached(),
        audience_sync(),
        notifications(),
    ]
