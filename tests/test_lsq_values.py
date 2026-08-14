"""Data-quality handling, driven by what the live account actually contains.

Every case here was observed in the production tenant, not invented. The
counts in the docstrings are from a 6-week sample of converted leads.
"""

from __future__ import annotations

from callytics.contracts.vocabulary import HIGH_INTENT_SOURCES, PAID_SOURCES
from callytics.integrations import leadsquared as lsq
from callytics.integrations import lsq_values as v

# --- placeholders ---------------------------------------------------------


def test_dropdown_defaults_are_treated_as_missing():
    """'Select Area' was the most common district value in the account (188)."""
    assert v.is_placeholder("Select Area")
    assert v.is_placeholder("Select Source")
    assert v.is_placeholder("NA")
    assert v.is_placeholder("(not set)")
    assert v.is_placeholder("other_courses")
    assert v.is_placeholder("   ")
    assert v.is_placeholder(None)


def test_real_values_survive():
    assert not v.is_placeholder("KOZHIKODE")
    assert not v.is_placeholder("Accounting & Finance")
    assert v.clean("ERNAKULAM") == "ERNAKULAM"
    assert v.clean("Select Area") is None


# --- course canonicalisation ---------------------------------------------


def test_same_course_spelled_differently_resolves_to_one():
    """'SAP S4/HANA FI' (9 leads) and 'sap_s/4hana_fi' (3) are one course."""
    assert v.canonical_course("SAP S4/HANA FI") == "SAP S4/HANA FI"
    assert v.canonical_course("sap_s/4hana_fi") == "SAP S4/HANA FI"


def test_acronym_and_full_name_resolve_to_one():
    assert v.canonical_course("IBAP") == "IBAP"
    assert v.canonical_course("international_business_accounting_professional") == "IBAP"


def test_slug_leaking_from_web_form_is_normalised():
    """'accounting_&_finance' is a URL slug that reached a display field."""
    assert v.canonical_course("accounting_&_finance") == "Accounting & Finance"


def test_configured_duplicates_collapse():
    """The configuration itself lists the same programme twice."""
    assert v.canonical_course("Gulf Accounting Analyst Program") == "Gulf Accounting Analyst"
    assert v.canonical_course("Gulf Accounting Analyst") == "Gulf Accounting Analyst"


def test_non_courses_in_the_catalogue_are_dropped():
    assert v.canonical_course("NA") is None


def test_unknown_course_is_kept_not_discarded():
    """A real answer we have not seen before must not be silently deleted."""
    assert v.canonical_course("Quantum Bookkeeping") == "Quantum Bookkeeping"


# --- multi-value parsing --------------------------------------------------


def test_semicolon_separator():
    assert v.parse_courses("PGDIFA;APBFA") == ["PGDIFA", "APBFA"]


def test_plus_separator():
    """'CAS + SAP' and 'IBAP+SAP FI' both appear in production."""
    assert v.parse_courses("CAS + SAP") == ["CAS", "SAP"]
    assert v.parse_courses("IBAP+SAP FI") == ["IBAP", "SAP FI"]


def test_ambiguous_fragments_are_preserved_not_guessed():
    """Splitting 'CAS + SAP' leaves a bare 'SAP', which is not a course.

    There are three SAP courses in the catalogue (FICO, MM, S4/HANA FI) and
    nothing in the record says which. Guessing would silently attribute a lead
    to the wrong programme, so the fragment is kept verbatim and surfaces in
    ImportStats.issues as outside the catalogue — a human resolves it.
    """
    assert "SAP" not in v.COURSES
    assert v.parse_courses("CAS + SAP")[1] == "SAP"


def test_mixed_and_deduplicated():
    assert v.parse_courses("CBAT;CBAT") == ["CBAT"]
    assert v.parse_courses("NA") == []
    assert v.parse_courses("(not set)") == []


def test_list_input_is_accepted():
    assert v.parse_courses(["CBAT", "Tally"]) == ["CBAT", "Tally"]


# --- source classification, corrected from measured conversions -----------


def test_inbound_whatsapp_counts_as_high_intent():
    """Second-largest source of enrolments (54 of 258) — not a cold ad click."""
    assert "Whatsapp Enquiry - KL" in HIGH_INTENT_SOURCES
    assert "Whatsapp Enquiry KA" in HIGH_INTENT_SOURCES


def test_paid_whatsapp_is_both_paid_and_high_intent():
    """The click cost money; the lead then messaged us. Both are true."""
    for source in ("Meta Whatsapp KL", "Google Whatsapp KL"):
        assert source in PAID_SOURCES
        assert source in HIGH_INTENT_SOURCES


def test_top_converting_sources_all_score_positively():
    from callytics.contracts.vocabulary import BULK_SOURCES

    top = ["Inbound Phone call", "Whatsapp Enquiry - KL", "Referral - New admissions", "Direct Walk-in"]
    for source in top:
        assert source in HIGH_INTENT_SOURCES
        assert source not in BULK_SOURCES


# --- field definitions ----------------------------------------------------


def test_select_fields_are_seeded_with_real_options():
    """Options came from production data, not the configuration screen."""
    rows = {r["key"]: r for r in lsq.field_definition_rows()}
    assert "KOZHIKODE" in rows["district"]["options"]
    assert "Accounting & Finance" in rows["course_category"]["options"]
    assert rows["courses"]["options"], "the course field must offer the catalogue"


def test_dead_fields_are_seeded_inactive():
    """mx_Vertical is populated on 3% of converted leads — it is not usable."""
    rows = {r["key"]: r for r in lsq.field_definition_rows()}
    assert rows["vertical"]["is_active"] is False
    assert rows["district"]["is_active"] is True


def test_qualifying_fields_are_a_short_curated_list():
    assert len(v.QUALIFYING_FIELDS) < 10
    assert "vertical" not in v.QUALIFYING_FIELDS, "a 97%-empty field cannot inform a score"


# --- import path ----------------------------------------------------------


def _record(**kw):
    base = {
        "ProspectID": "abc-123",
        "FirstName": "Amal Raj",
        "Phone": "+91 98765 43210",
        "ProspectStage": "Interested",
        "Source": "Whatsapp Enquiry - KL",
        "CreatedOn": "2026-03-01T10:00:00Z",
    }
    base.update(kw)
    return base


def test_import_drops_placeholder_district():
    payload = lsq.to_lead_create(_record(mx_District="Select Area"))
    assert "district" not in payload.fields


def test_import_keeps_real_district():
    payload = lsq.to_lead_create(_record(mx_District="KOZHIKODE"))
    assert payload.fields["district"] == "KOZHIKODE"


def test_import_canonicalises_and_splits_courses():
    payload = lsq.to_lead_create(_record(mx_Courses="sap_s/4hana_fi;CBAT"))
    assert payload.fields["courses"] == ["SAP S4/HANA FI", "CBAT"]


def test_import_reports_courses_outside_the_catalogue():
    stats = lsq.ImportStats()
    lsq.to_lead_create(_record(mx_Courses="Quantum Bookkeeping"), stats)
    assert any("not in catalogue" in i.problem for i in stats.issues)
