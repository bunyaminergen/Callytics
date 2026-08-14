"""LeadSquared migration: field mapping, value fidelity, reconciliation."""

from __future__ import annotations

from callytics.contracts.vocabulary import LEAD_SOURCES, LEAD_STAGES
from callytics.integrations import leadsquared as lsq


def _record(**kw):
    base = {
        "ProspectID": "abc-123",
        "FirstName": "Amal Raj",
        "EmailAddress": "Amal@Example.com",
        "Phone": "+91 98765 43210",
        "ProspectStage": "Interested",
        "Source": "FB Lead Ads",
        "CreatedOn": "2026-03-01T10:00:00Z",
        "mx_Course_Category": "ACCA",
        "mx_District": "Kozhikode",
    }
    base.update(kw)
    return base


def test_seed_rows_cover_the_live_account():
    """Parity check against the vocabulary read from the production tenant."""
    assert len(lsq.stage_rows()) == len(LEAD_STAGES) == 18
    assert len(lsq.source_rows()) == len(LEAD_SOURCES) == 51
    assert {r["name"] for r in lsq.stage_rows()} == set(LEAD_STAGES)


def test_revenue_stages_are_flagged():
    by_name = {r["name"]: r for r in lsq.stage_rows()}
    assert by_name["JOINED"]["is_revenue"]
    assert by_name["Already joined"]["is_revenue"]
    # A commitment is not revenue — matches LeadSquared reporting defaults.
    assert not by_name["Confirmed - Waiting for payment"]["is_revenue"]
    assert by_name["Import"]["is_artefact"]


def test_paid_and_bulk_sources_are_flagged():
    by_name = {r["name"]: r for r in lsq.source_rows()}
    assert by_name["FB Lead Ads"]["is_paid"]
    assert by_name["College Data purchase"]["is_bulk"]
    assert by_name["Direct Walk-in"]["is_high_intent"]


def test_field_definitions_keep_legacy_schema_names():
    rows = {r["key"]: r for r in lsq.field_definition_rows()}
    assert rows["course_category"]["legacy_schema_name"] == "mx_Course_Category"
    assert rows["year_of_passout"]["legacy_schema_name"] == "mx_Semester"
    assert all(r["label"] for r in rows.values())


def test_record_converts_with_normalised_identity():
    stats = lsq.ImportStats()
    payload = lsq.to_lead_create(_record(), stats)
    assert payload is not None
    assert payload.identity.phone == "919876543210"
    assert payload.identity.email == "amal@example.com"
    assert payload.stage == "Interested"
    assert payload.fields["course_category"] == "ACCA"
    assert payload.external_ids["leadsquared"] == "abc-123"
    assert stats.converted == 1


def test_uncontactable_record_is_skipped_not_imported():
    stats = lsq.ImportStats()
    assert lsq.to_lead_create(_record(Phone="", Mobile="", EmailAddress=""), stats) is None
    assert stats.skipped == 1
    assert stats.issues[0].problem == "no phone or email"


def test_unparseable_phone_is_reported():
    stats = lsq.ImportStats()
    lsq.to_lead_create(_record(Phone="n/a", Mobile="", EmailAddress="a@b.com"), stats)
    assert any(i.field == "Phone" for i in stats.issues)


def test_unknown_stage_falls_back_and_is_reported():
    stats = lsq.ImportStats()
    payload = lsq.to_lead_create(_record(ProspectStage="Something Odd"), stats)
    assert payload.stage == "Open"
    assert any("unknown stage" in i.problem for i in stats.issues)


def test_consent_defaults_to_no_whatsapp_without_explicit_opt_in():
    payload = lsq.to_lead_create(_record())
    assert payload.consent.do_not_whatsapp is True

    opted_in = lsq.to_lead_create(_record(mx_WhatsApp_Consent="Yes"))
    assert opted_in.consent.do_not_whatsapp is False


def test_do_not_call_is_carried_across():
    payload = lsq.to_lead_create(_record(DoNotCall="true"))
    assert payload.consent.do_not_call is True


def test_reconcile_reports_drift():
    ours = {"stage": "Interested", "source": "FB Lead Ads", "phone": "919876543210", "email": "a@b.com"}
    theirs = _record(ProspectStage="JOINED", EmailAddress="a@b.com")
    drift = lsq.reconcile(ours, theirs)
    assert len(drift) == 1
    assert drift[0].field == "stage"
    assert drift[0].callytics_value == "Interested"
    assert drift[0].leadsquared_value == "JOINED"


def test_reconcile_is_quiet_when_aligned():
    theirs = _record()
    ours = {
        "stage": "Interested",
        "source": "FB Lead Ads",
        "phone": "919876543210",
        "email": "amal@example.com",
    }
    assert lsq.reconcile(ours, theirs) == []


def test_dropped_fields_are_explicit():
    """Dropping a field must be a decision, not an accident."""
    assert "ProspectActivityId_Max" in lsq.DROPPED_FIELDS
    overlap = lsq.DROPPED_FIELDS.intersection(
        {legacy for _, _, legacy in lsq.CUSTOM_FIELD_MAP.values()}
    )
    assert not overlap, "a field cannot be both mapped and dropped"
