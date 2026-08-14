"""FinEcho client and webhook mapping — the conversation-intelligence boundary.

FinEcho is a git submodule of this repo but **not** a Python import. It runs as
its own service with its own model stack (faster-whisper, IndicTrans2, a
fine-tuned Llama), and pulling that dependency tree into the CRM's process
would make the CRM impossible to deploy on ordinary hosting. The submodule
pins the version we integrate against; this module is the only thing that
talks to it.

Direction of travel:

* **FinEcho → Callytics** (primary): FinEcho posts a completed call summary or
  evaluation to ``/webhooks/finecho/call``. That is the hot path and it is
  push, not poll, so a summary appears on the timeline seconds after it is
  produced.
* **Callytics → FinEcho** (secondary): fetching a recording URL, or requesting
  a re-evaluation of a call a manager disputes.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime
from typing import Any
from uuid import UUID

import requests

from ..contracts.intelligence import CallIntelligence
from ..settings import get_settings


class FinEchoError(RuntimeError):
    pass


def verify_signature(body: bytes, signature: str, secret: str) -> bool:
    """Constant-time HMAC-SHA256 check on the webhook body.

    The webhook carries call content, so an unauthenticated endpoint would let
    anyone write arbitrary summaries onto a lead's timeline.
    """
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    provided = signature.removeprefix("sha256=").strip()
    return hmac.compare_digest(expected, provided)


def parse_webhook(payload: dict[str, Any]) -> CallIntelligence:
    """Map FinEcho's summary/evaluation payload onto the CRM contract.

    FinEcho splits its output across a summary document and an evaluation
    document; a webhook may carry either or both, so every evaluation-side
    field is optional here.
    """
    summary = payload.get("summary") or {}
    evaluation = payload.get("evaluation") or {}
    call = payload.get("call") or payload

    call_id = call.get("call_id") or call.get("id")
    if not call_id:
        raise FinEchoError("webhook payload has no call id")

    phone = call.get("lead_phone") or call.get("phone")
    if not phone:
        raise FinEchoError("webhook payload has no lead phone")

    occurred_raw = call.get("recorded_at") or call.get("occurred_at")
    if not occurred_raw:
        raise FinEchoError("webhook payload has no recording timestamp")

    return CallIntelligence(
        call_id=UUID(str(call_id)),
        lead_phone=str(phone),
        agent_key=call.get("employee_id") or call.get("counsellor_key"),
        occurred_at=_parse_dt(occurred_raw),
        duration_seconds=call.get("duration_seconds"),
        language=call.get("language_primary") or call.get("language"),
        summary=summary.get("summary"),
        points_to_remember=list(summary.get("points_to_remember") or []),
        objections=[_text(o) for o in (summary.get("objections") or [])],
        hooks=[_text(h) for h in (summary.get("hooks") or [])],
        promised_actions=[_text(a) for a in (summary.get("promised_actions") or [])],
        suggested_next_steps=[_text(s) for s in (summary.get("suggested_next_steps") or [])],
        detected_program=evaluation.get("detected_program"),
        call_type=evaluation.get("call_type"),
        overall_score=evaluation.get("overall_score"),
        criteria=list(evaluation.get("criteria") or []),
        compliance_flags=list(evaluation.get("compliance_flags") or []),
        recording_uri=call.get("audio_uri") or call.get("recording_uri"),
        model_tag=summary.get("model_tag") or evaluation.get("model_tag"),
    )


def _text(value: Any) -> str:
    """FinEcho emits some list entries as objects; keep the human-readable part."""
    if isinstance(value, dict):
        return str(value.get("text") or value.get("detail") or value.get("value") or value)
    return str(value)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class FinEchoClient:
    """Outbound calls to FinEcho's internal API."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout_s: int | None = None,
    ) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.finecho_url).rstrip("/")
        self._token = token or settings.finecho_token
        self._timeout_s = timeout_s or settings.finecho_timeout_s

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self._base_url}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            response = requests.request(method, url, headers=headers, timeout=self._timeout_s, **kwargs)
        except requests.RequestException as exc:
            raise FinEchoError(f"FinEcho request failed: {exc}") from exc
        if response.status_code >= 400:
            raise FinEchoError(f"FinEcho {response.status_code}: {response.text[:300]}")
        return response.json()

    def get_call(self, call_id: UUID) -> CallIntelligence:
        return parse_webhook(self._request("GET", f"/api/calls/{call_id}"))

    def request_reevaluation(self, call_id: UUID, *, requested_by: str, reason: str) -> dict[str, Any]:
        """Ask FinEcho to re-score a call a manager disputes.

        FinEcho runs this on its Lane B queue, so it never competes with the
        live transcription pipeline.
        """
        return self._request(
            "POST",
            "/api/evaluation-jobs",
            json={"call_ids": [str(call_id)], "requested_by": requested_by, "reason": reason},
        )

    def health(self) -> bool:
        try:
            self._request("GET", "/healthz")
        except FinEchoError:
            return False
        return True
