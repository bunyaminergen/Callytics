"""Observed values from the live LeadSquared account, and how to clean them.

Everything here was read from the production tenant, not from its configuration
screen — which matters, because the two disagree badly. Three problems showed up
that pure schema inspection cannot see, and each one would have silently
corrupted the migration:

1. **Placeholder values that look like data.** ``Select Source``, ``Select
   Area``, ``NA``, ``other_courses`` are dropdown defaults nobody changed. In a
   6-week sample, ``Select Area`` was the single most common "district" (188
   leads) and ``Select Source`` appears as a lead source. Imported verbatim
   they would become real-looking segments that mean nothing.

2. **The same course spelled several ways.** ``SAP S4/HANA FI`` (9 leads) and
   ``sap_s/4hana_fi`` (3) are one course. ``IBAP`` (2) and
   ``international_business_accounting_professional`` (1) are one course.
   ``accounting_&_finance`` is a URL slug that leaked out of a web form into a
   display field. LeadSquared's own duplicate finder misses all of these
   because it only compares case, spacing and punctuation — these differ
   semantically, so they need a real synonym map.

3. **Two multi-value separators.** ``PGDIFA;APBFA`` uses a semicolon,
   ``CAS + SAP`` and ``IBAP+SAP FI`` use a plus. Splitting on one loses the
   other.

Field population is also far worse than the 117-field schema suggests. Measured
over converted leads in a 6-week window: ``mx_Vertical`` 97 % empty,
``mx_Courses`` 68 % empty, ``mx_Course_Category`` 70 % empty. That is why
scoring uses a curated qualifying-field list rather than counting all 31 mapped
fields — most of them are simply never filled, and rewarding their absence-or-
presence would be scoring noise.
"""

from __future__ import annotations

import re

# --- placeholders ---------------------------------------------------------

#: Dropdown defaults and junk markers. Treated as NULL on import: they are the
#: absence of an answer wearing the costume of one.
PLACEHOLDER_VALUES: frozenset[str] = frozenset(
    {
        "select source",
        "select area",
        "select",
        "(not set)",
        "not set",
        "na",
        "n/a",
        "none",
        "null",
        "-",
        "--",
        "other_courses",
        "others",
        "other",
        "test",
    }
)


def is_placeholder(value: object) -> bool:
    """True when a value carries no information despite being non-empty."""
    if value is None:
        return True
    text = str(value).strip().lower()
    return text == "" or text in PLACEHOLDER_VALUES


def clean(value: object) -> object | None:
    """Return the value, or None if it is empty or a placeholder."""
    return None if is_placeholder(value) else value


# --- districts ------------------------------------------------------------

#: Observed in production, uppercase as stored. Kerala-centric with a few
#: Bangalore/overseas outliers, matching where the centres actually are.
DISTRICTS: tuple[str, ...] = (
    "ERNAKULAM",
    "KOLLAM",
    "KOZHIKODE",
    "TRIVANDRUM",
    "MALAPPURAM",
    "THRISSUR",
    "ALAPPUZHA",
    "PATHANAMTHITTA",
    "PALAKKAD",
    "KANNUR",
    "WAYANAD",
    "KOTTAYAM",
    "IDUKKI",
    "KASARGOD",
    "BENGALURU URBAN",
    "BENGALURU SOUTH",
    "TAMIL NADU",
    "DUBAI",
)

COURSE_CATEGORIES: tuple[str, ...] = ("Accounting & Finance", "Digital Marketing", "Data Analytics")

VERTICALS: tuple[str, ...] = ("Business and Finance", "Digital Skills")


# --- courses --------------------------------------------------------------

#: The account's configured course list (70 entries). Note that the
#: configuration itself carries duplicates and junk — ``Gulf Accounting
#: Analyst`` vs ``Gulf Accounting Analyst Program``, plus ``NA`` and
#: ``Job Enquiry`` — which CANONICAL_COURSES resolves.
COURSES: tuple[str, ...] = (
    "USDC", "CBAT", "SAP S4/HANA FI", "Tally", "COA", "PGDM", "Income Tax", "GSTA", "FFE", "BOS",
    "MS Excel", "Logistics", "PGBAT", "CA Foundation", "CA Inter", "CA Final", "CSEET",
    "CMA Foundation-Indian", "GST Elearning", "Income Tax Elearning", "PAT E-learning",
    "B.Com Tuition", "Foreign Accounting", "DIA", "PGDIFA", "QB Elearning", "QBO",
    "GST Practitioner", "CHRPP", "Tally E-learning", "E-Commerce", "Gulf VAT", "CMA USA",
    "Gulf VAT E-learning", "MS Excel E-learning", "Gulf Accounting Analyst",
    "Gulf VAT E-learning - Eng", "GST Elearning - Malayalam", "Bank Coaching", "Digital Marketing",
    "ACCA", "ACCA LEVEL 1", "ACCA LEVEL 2", "ACCA LEVEL 3", "ACCA ALL LEVELS", "SAP MM",
    "Retail Banker", "IBAP", "BASP", "Zoho Books", "CAS", "Data Analytics", "Study Abroad",
    "Sage 50", "Office Administration", "Creators Cut", "APBFA", "DA", "Business Analyst",
    "Financial Analyst", "MIS Analyst", "CBFA", "Power Bi", "Tally Prime 7.0",
    "Cloud & AI Accounting", "SAP FICO",
)

#: Synonym → canonical. Keys are compared after :func:`_fold`, so case,
#: underscores, spaces and punctuation are already normalised; only genuine
#: aliases need an entry here.
COURSE_ALIASES: dict[str, str] = {
    "saps4hanafi": "SAP S4/HANA FI",
    "saps4hanafi ": "SAP S4/HANA FI",
    "sapshanafi": "SAP S4/HANA FI",
    "internationalbusinessaccountingprofessional": "IBAP",
    "gulfaccountinganalystprogram": "Gulf Accounting Analyst",
    "accountingfinance": "Accounting & Finance",
    "digitalmarketing": "Digital Marketing",
    "dataanalytics": "Data Analytics",
    "powerbi": "Power Bi",
    "sapfico": "SAP FICO",
    "sapmm": "SAP MM",
    "zohobooks": "Zoho Books",
    "msexcel": "MS Excel",
    "tallyprime70": "Tally Prime 7.0",
    "cloudaiaccounting": "Cloud & AI Accounting",
}

#: Configured entries that are not really courses.
NON_COURSES: frozenset[str] = frozenset({"NA", "Job Enquiry"})

#: Both separators appear in production data.
_MULTI_SEPARATOR = re.compile(r"[;+]")


def _fold(value: str) -> str:
    """Aggressive fold for alias lookup: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def canonical_course(value: object) -> str | None:
    """Resolve one course string to its canonical spelling, or None."""
    if is_placeholder(value):
        return None
    raw = str(value).strip()
    folded = _fold(raw)
    if not folded:
        return None

    if folded in COURSE_ALIASES:
        return COURSE_ALIASES[folded]

    for course in COURSES:
        if _fold(course) == folded:
            return None if course in NON_COURSES else course

    # Unknown but non-empty: keep the original rather than discard a real
    # answer we simply have not seen before. The importer reports it.
    return raw


def parse_courses(value: object) -> list[str]:
    """Split a multi-value course field and canonicalise each entry.

    Handles both separators seen in production (``;`` and ``+``) and drops
    placeholders, so ``"PGDIFA;APBFA"`` and ``"CAS + SAP"`` both work.
    """
    if is_placeholder(value):
        return []
    if isinstance(value, list | tuple):  # noqa: SIM108 - clearer than a long ternary
        parts = [str(v) for v in value]
    else:
        parts = _MULTI_SEPARATOR.split(str(value))

    out: list[str] = []
    for part in parts:
        course = canonical_course(part)
        if course and course not in out:
            out.append(course)
    return out


# --- scoring inputs -------------------------------------------------------

#: Fields that indicate a genuinely qualified enquiry AND are actually filled
#: often enough to carry signal. Deliberately short: measured population rates
#: showed most mapped fields are empty on the majority of even *converted*
#: leads, so counting all 31 would score noise.
QUALIFYING_FIELDS: tuple[str, ...] = (
    "courses",
    "course_category",
    "district",
    "highest_qualification",
    "occupation",
    "course_mode",
)


#: Measured share of *converted* leads with the field populated, 2026-06-01 to
#: 2026-08-14. Recorded so the next person does not have to rediscover that
#: these fields are mostly empty, and so the seeder can flag them.
OBSERVED_POPULATION: dict[str, float] = {
    "vertical": 0.03,
    "courses": 0.32,
    "course_category": 0.30,
    "district": 0.52,  # excluding the "Select Area" placeholder
}

#: Below this, a field is marked unreliable in ``field_definitions`` and is
#: kept out of scoring.
UNRELIABLE_BELOW = 0.10
