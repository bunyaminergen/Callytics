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
