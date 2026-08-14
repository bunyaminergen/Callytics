# 04 — Migration from LeadSquared

CRM replacements do not fail on features. They fail on cutover weekend, when
100k leads arrive with mangled phone numbers, a stage nobody recognises and no
owner, and the sales floor loses a week.

## Principles

1. **Import values verbatim.** Stage and source strings come across unchanged.
   A migrated lead reads identically in both systems, which is the only thing
   that makes a side-by-side reconciliation meaningful.
2. **Lose no field silently.** Mapped or explicitly dropped — and the dropped
   list is a tested constant, with a test asserting no field is both.
3. **Never import an uncontactable lead.** A row with no phone and no email
   inflates the database and the funnel. It is skipped and reported.
4. **Read-only against LeadSquared.** During dual run it remains the system of
   record. A migration tool that can write to the system you still depend on is
   a way to corrupt it.

## Field mapping

| Category | Count | Destination |
|---|---|---|
| Promoted | ~15 | indexed columns on `leads` |
| Custom | ~31 | `leads.fields` + a `field_definitions` row keeping the `mx_` name |
| Dropped | ~20 | explicit constant, LeadSquared internals and unused web analytics |

Every custom field retains `legacy_schema_name`, so any historical report can
be traced back to its LeadSquared origin.

## What the data actually looks like

Measured against the live tenant, not its configuration screen. The two
disagree badly, and every item below would have corrupted the migration if the
mapping had been built from schema alone.

### Placeholder values that look like data

Dropdown defaults nobody changed. Over a 6-week sample of converted leads,
`Select Area` was the **most common district value in the account** (188
leads) — more than Ernakulam, Kollam and Kozhikode combined. `Select Source`
is a configured lead source. `NA` and `other_courses` appear in the course
field.

Imported verbatim these become real-looking segments that mean nothing, and
they will be charted. `lsq_values.PLACEHOLDER_VALUES` nulls them on import.

### Field population is far lower than the schema implies

Share of **converted** leads with the field filled (2026-06-01 → 2026-08-14):

| Field | Populated |
|---|---|
| `mx_Vertical` | 3 % |
| `mx_Courses` | 32 % |
| `mx_Course_Category` | 30 % |
| `mx_District` | 52 % (excluding `Select Area`) |

Two consequences. `mx_Vertical` is seeded **inactive** — carrying it keeps the
mapping traceable, but showing a 97 %-empty field to counsellors is noise.
And scoring counts a curated `QUALIFYING_FIELDS` list rather than all 31
mapped fields; completeness across mostly-empty fields measures which web form
the lead came through, not how qualified they are.

### The same course spelled several ways

| Variant | Leads | Canonical |
|---|---|---|
| `SAP S4/HANA FI` | 9 | `SAP S4/HANA FI` |
| `sap_s/4hana_fi` | 3 | ↳ same course |
| `IBAP` | 2 | `IBAP` |
| `international_business_accounting_professional` | 1 | ↳ same course |
| `accounting_&_finance` | 4 | `Accounting & Finance` (a URL slug that leaked out of a web form) |
| `Gulf Accounting Analyst` / `...Program` | 1 / 1 | one programme, listed twice **in the configuration** |

LeadSquared's own `find_duplicate_values` reports **zero** duplicate groups
here, because it only compares case, spacing and punctuation. These differ
semantically, so `COURSE_ALIASES` is a real synonym map rather than a
case-fold.

### Two multi-value separators

`PGDIFA;APBFA` uses a semicolon; `CAS + SAP` and `IBAP+SAP FI` use a plus.
`parse_courses` handles both.

Splitting on `+` leaves an ambiguous fragment — bare `SAP`, when the catalogue
has three SAP courses. It is **kept verbatim and reported**, never guessed:
attributing a lead to the wrong programme is worse than flagging it for a
human.

## Known translation decisions

- **WhatsApp consent.** LeadSquared has no suppression column for it; the
  account tracks `mx_WhatsApp_Consent`. Absence is treated as *not opted in* —
  the conservative reading, and the one the DPDP Act expects.
- **Unknown stage or source** falls back to `Open` / `Select Source` and is
  reported in `ImportStats.issues` rather than failing the row.
- **Phone normalisation** runs on import. An unparseable number is reported;
  if the row has another identifier it still imports.

## Cutover sequence

1. **Seed** stages, sources, field definitions, teams, users.
2. **Bulk import** with `ImportStats` collected. Review the issue report
   *before* going further — it is the last cheap moment to find a systematic
   mapping error.
3. **Dual run.** Both systems live. FinEcho pushes to both. `reconcile()` diffs
   each lead nightly and reports drift on stage, source, phone and email.
4. **Cut** when drift is zero for a full week and managers agree the numbers
   match.
5. **Freeze** LeadSquared to read-only. Keep it readable for a quarter.

## Reconciliation

`reconcile(callytics_lead, lsq_record)` returns a list of `Drift` — field, our
value, their value. Zero drift over a sustained period is the objective
criterion for cutover; "it looks fine" is not.
