"""LeadSquared migration and dual-run reconciliation.

Replacing a CRM fails at cutover, not at build time. The risk is not that the
new system lacks features — it is that 100k leads arrive with mangled phone
numbers, a stage nobody recognises, and no owner, and the sales floor loses a
week. This module exists to make cutover boring:

* **Field mapping.** The live account has 117 lead fields. The ~15 that drive
  queries become real columns; the rest become ``field_definitions`` rows
  carrying their original ``mx_`` schema name, so nothing is lost and any
  report can be traced back to its LeadSquared origin.
* **Value fidelity.** Stage and source strings are imported verbatim. A
  migrated lead reads identically in both systems, which is what makes a
  dual-run reconciliation meaningful.
* **Dual run.** ``reconcile`` compares a Callytics lead against its
  LeadSquared twin and reports drift, so the two can run side by side until the
  numbers agree.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests

from ..contracts.lead import Consent, LeadCreate, LeadIdentity
from ..contracts.vocabulary import (
    ARTEFACT_STAGES,
    BULK_SOURCES,
    HIGH_INTENT_SOURCES,
    LEAD_SOURCES,
    LEAD_STAGES,
    NURTURE_STAGES,
    PAID_SOURCES,
    REVENUE_STAGES,
)
from ..util import normalize_phone

# --- field mapping --------------------------------------------------------

#: LeadSquared schema name → Callytics promoted column. These are the fields
#: that appear in list views, filters and joins, so they earn a real column.
PROMOTED_FIELDS: dict[str, str] = {
    "ProspectID": "external_ids.leadsquared",
    "FirstName": "full_name",
    "EmailAddress": "email",
    "Phone": "phone",
    "Mobile": "alternate_phone",
    "ProspectStage": "stage",
    "Source": "source",
    "SourceCampaign": "source_campaign",
    "SourceMedium": "source_medium",
    "SourceReferrer": "source_referrer",
    "OwnerId": "owner_id",
    "CreatedOn": "captured_at",
    "DoNotCall": "do_not_call",
    "DoNotEmail": "do_not_email",
    "DoNotSMS": "do_not_sms",
    "mx_Consent_Timestamp": "consent_timestamp",
}

#: Everything else worth keeping becomes a custom field. Key is the Callytics
#: key; value is (label, data_type, legacy schema name).
CUSTOM_FIELD_MAP: dict[str, tuple[str, str, str]] = {
    "preferred_call_time": ("Preferred Call Time", "select", "mx_Preferred_Call_Time"),
    "whatsapp_consent": ("WhatsApp Consent", "select", "mx_WhatsApp_Consent"),
    "enquiry_detail": ("Enquiry Detail", "text", "mx_Enquiry_Detail"),
    "enquiry_type": ("Enquiry Type", "select", "mx_Enquiry_Type"),
    "detected_course": ("Detected Course", "select", "mx_Detected_Course"),
    "district": ("Location of Lead", "select", "mx_District"),
    "vertical": ("Vertical", "select", "mx_Vertical"),
    "call_counter": ("Call Counter", "number", "mx_Call_Counter"),
    "has_laptop": ("Do you have laptop", "text", "mx_Do_you_have_laptop"),
    "is_working_professional": ("Are you a working professional", "text", "mx_Are_you_a_working_professional"),
    "year_of_graduation": ("Year of Graduation", "number", "mx_Year_of_Graduation"),
    "highest_qualification": ("Highest Qualification", "text", "mx_Highest_Qualification"),
    "lead_note": ("Lead Note", "text", "mx_Lead_Note"),
    "course_location": ("Course Location", "select", "mx_Course_Location"),
    "course_purpose": ("Course Purpose", "text", "mx_Course_Purpose"),
    "course_duration": ("Course Duration", "select", "mx_Course_Duration"),
    "course_category": ("Course Category", "select", "mx_Course_Category"),
    "language": ("Language", "select", "mx_Language"),
    "learning_location": ("Learning Location", "select", "mx_Learning_Location"),
    "course_mode": ("Course Mode", "select", "mx_Course_Mode"),
    "bcom_subjects": ("BCom Subjects", "multiselect", "mx_BCom_Subjects"),
    "year_of_passout": ("Year of Passout", "select", "mx_Semester"),
    "university": ("University", "select", "mx_University"),
    "franchisee_code": ("Franchisee Code", "select", "mx_Franchisee_Code"),
    "occupation": ("Occupation", "select", "mx_Occupation"),
    "courses": ("Courses", "multiselect", "mx_Courses"),
    "city": ("City", "text", "mx_City"),
    "state": ("State", "text", "mx_State"),
    "country": ("Country", "select", "mx_Country"),
    "join_timeline": ("When will you join", "text", "mx_when_will_you_be_able_to_join_for_the_course_"),
    "learning_motivation": ("Why this course", "text", "mx_why_do_you_want_to_learn_this_course"),
}

#: Fields deliberately dropped: LeadSquared internals with no meaning outside
#: it, or web-analytics columns the account never populated.
DROPPED_FIELDS: frozenset[str] = frozenset(
    {
        "ProspectActivityId_Max",
        "ProspectActivityId_Min",
        "DeletionStatusCode",
        "StatusCode",
        "StatusReason",
        "TotalVisits",
        "PageViewsPerVisit",
        "AvgTimePerVisit",
        "RelatedLandingPageId",
        "FirstLandingPageSubmissionId",
        "TwitterId",
        "FacebookId",
        "LinkedInId",
        "SkypeId",
        "GTalkId",
        "GooglePlusId",
        "PhotoUrl",
        "Latitude",
        "Longitude",
        "SourceIPAddress",
    }
)


def field_definition_rows() -> list[dict[str, Any]]:
    """Seed rows for ``field_definitions``, preserving legacy schema names."""
    return [
        {
            "key": key,
            "label": label,
            "data_type": data_type,
            "legacy_schema_name": legacy,
            "position": index,
            "is_active": True,
            "is_pii": key in {"city", "state", "lead_note"},
        }
        for index, (key, (label, data_type, legacy)) in enumerate(CUSTOM_FIELD_MAP.items())
    ]


def stage_rows() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "position": index,
            "is_open": name not in ARTEFACT_STAGES,
            "is_revenue": name in REVENUE_STAGES,
            "is_nurture": name in NURTURE_STAGES,
            "is_artefact": name in ARTEFACT_STAGES,
            "is_active": True,
        }
        for index, name in enumerate(LEAD_STAGES)
    ]


def source_rows() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "is_paid": name in PAID_SOURCES,
            "is_high_intent": name in HIGH_INTENT_SOURCES,
            "is_bulk": name in BULK_SOURCES,
            "is_active": True,
        }
        for name in LEAD_SOURCES
    ]


# --- record translation ---------------------------------------------------


@dataclass
class ImportIssue:
    prospect_id: str
    field: str
    problem: str


@dataclass
class ImportStats:
    seen: int = 0
    converted: int = 0
    skipped: int = 0
    issues: list[ImportIssue] = field(default_factory=list)


_TRUE = {"true", "1", "yes", "y"}


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in _TRUE if value is not None else False


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def to_lead_create(record: dict[str, Any], stats: ImportStats | None = None) -> LeadCreate | None:
    """Translate one LeadSquared lead record into a capture payload.

    Returns None when the record has no usable contact identifier — importing a
    lead nobody can contact only inflates the database and the funnel.
    """
    stats = stats or ImportStats()
    stats.seen += 1
    prospect_id = str(record.get("ProspectID") or "")

    phone = normalize_phone(record.get("Phone"))
    alternate = normalize_phone(record.get("Mobile"))
    email = (record.get("EmailAddress") or "").strip() or None

    if not phone and not alternate and not email:
        stats.skipped += 1
        stats.issues.append(ImportIssue(prospect_id, "identity", "no phone or email"))
        return None

    if record.get("Phone") and not phone:
        stats.issues.append(ImportIssue(prospect_id, "Phone", f"unparseable: {record.get('Phone')!r}"))

    stage = (record.get("ProspectStage") or "Open").strip()
    if stage not in LEAD_STAGES:
        stats.issues.append(ImportIssue(prospect_id, "ProspectStage", f"unknown stage {stage!r}, using Open"))
        stage = "Open"

    source = (record.get("Source") or "Select Source").strip()
    if source not in LEAD_SOURCES:
        stats.issues.append(ImportIssue(prospect_id, "Source", f"unknown source {source!r}, using Select Source"))
        source = "Select Source"

    fields: dict[str, Any] = {}
    for key, (_label, _type, legacy) in CUSTOM_FIELD_MAP.items():
        value = record.get(legacy)
        if value not in (None, "", []):
            fields[key] = value

    consent = Consent(
        do_not_call=_boolish(record.get("DoNotCall")),
        do_not_sms=_boolish(record.get("DoNotSMS")),
        do_not_email=_boolish(record.get("DoNotEmail")),
        # LeadSquared has no explicit WhatsApp suppression column; the account
        # tracks it in mx_WhatsApp_Consent, so absence means not opted in.
        do_not_whatsapp=str(record.get("mx_WhatsApp_Consent") or "").strip().lower() not in {"yes", "true", "opted in"},
        consent_timestamp=_dt(record.get("mx_Consent_Timestamp")),
        consent_source="leadsquared_import",
    )

    stats.converted += 1
    return LeadCreate(
        identity=LeadIdentity(
            phone=phone or alternate,
            alternate_phone=alternate if phone else None,
            email=email,
            full_name=(record.get("FirstName") or "").strip() or None,
        ),
        source=source,
        source_campaign=record.get("SourceCampaign"),
        source_medium=record.get("SourceMedium"),
        source_referrer=record.get("SourceReferrer"),
        stage=stage,
        consent=consent,
        fields=fields,
        external_ids={"leadsquared": prospect_id} if prospect_id else {},
        captured_at=_dt(record.get("CreatedOn")),
    )


# --- reconciliation -------------------------------------------------------


@dataclass
class Drift:
    prospect_id: str
    field: str
    callytics_value: Any
    leadsquared_value: Any


def reconcile(callytics_lead: dict[str, Any], lsq_record: dict[str, Any]) -> list[Drift]:
    """Compare a migrated lead against its LeadSquared twin during dual run."""
    prospect_id = str(lsq_record.get("ProspectID") or "")
    drift: list[Drift] = []

    comparisons = (
        ("stage", callytics_lead.get("stage"), (lsq_record.get("ProspectStage") or "").strip()),
        ("source", callytics_lead.get("source"), (lsq_record.get("Source") or "").strip()),
        ("phone", callytics_lead.get("phone"), normalize_phone(lsq_record.get("Phone"))),
        (
            "email",
            (callytics_lead.get("email") or "").lower() or None,
            (lsq_record.get("EmailAddress") or "").strip().lower() or None,
        ),
    )
    for name, ours, theirs in comparisons:
        if ours != theirs:
            drift.append(Drift(prospect_id, name, ours, theirs))
    return drift


# --- read-only source client ---------------------------------------------


class LeadSquaredReader:
    """Minimal read-only client for migration and reconciliation.

    Deliberately read-only: during dual run LeadSquared remains the system of
    record, and a migration tool that can write to the old system is a way to
    corrupt the thing you are still depending on.
    """

    def __init__(
        self,
        base_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        rate_per_min: int | None = None,
    ) -> None:
        from ..settings import get_settings

        settings = get_settings()
        self._base_url = (base_url or settings.lsq_base_url).rstrip("/")
        self._auth = {
            "accessKey": access_key or settings.lsq_access_key,
            "secretKey": secret_key or settings.lsq_secret_key,
        }
        self._min_interval = 60.0 / max(1, rate_per_min or settings.lsq_rate_per_min)
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self._throttle()
        response = requests.get(
            f"{self._base_url}/{path.lstrip('/')}",
            params={**self._auth, **(params or {})},
            timeout=60,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"LeadSquared {response.status_code}: {response.text[:300]}")
        return response.json()

    def iter_leads(self, *, page_size: int = 200, max_pages: int = 10_000) -> Any:
        """Yield lead records page by page."""
        for page in range(max_pages):
            body = self._get(
                "LeadManagement.svc/Leads.Get",
                {"Parameters.Index": page + 1, "Parameters.PageSize": page_size},
            )
            rows = body if isinstance(body, list) else body.get("Leads", [])
            if not rows:
                return
            yield from rows
