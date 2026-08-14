# 02 — Data model

## The append-only timeline

`activities` is never updated in place. Every row is a fact that happened:
a stage change, a call, a message, a proposal decision. A lead's current
state is a projection over that history.

This is the direct answer to the scenario raised in the Meritto demo — a
counsellor inherits a lead that has been worked for six months and needs the
background. In LeadSquared today the answer is "read the notes". Here it is one
query, ordered, typed, and complete: `GET /api/leads/{id}/timeline`.

It also means a disputed stage change can always be reconstructed: who moved
it, when, on what reason, and whether a proposal was involved.

`leads` still carries denormalised counters (`score`, `last_activity_at`,
`connected_calls`) because list views cannot aggregate the timeline per row at
read time. Those are caches, rebuildable from activities — never a source of
truth.

## Promoted columns vs. the JSON tail

The live LeadSquared account has **117 lead fields**. Modelling each as a
column produces a table nobody can migrate; modelling all of them as JSON
produces a table nobody can query.

So: roughly fifteen fields that appear in filters, sorts and joins get real
indexed columns. The rest live in `leads.fields` (JSON), described by a
`field_definitions` row carrying label, data type, options and the original
`mx_` schema name.

Promoting a field later is a migration, not a redesign — and the reverse
mapping to LeadSquared survives, so a report can always be traced to its
origin.

## Identity

`lead_identifiers` holds one row per contactable identifier with a
`UNIQUE(kind, value)` constraint. That constraint — not a nightly cleanup job —
is what prevents duplicate leads, because it fails at write time.

A person legitimately has several numbers, so identifiers are a separate table
rather than columns. `leads.phone` remains as the display primary.

Resolution policy (`domain/identity.py`): automatic attach on an exact phone or
email match; **name matches never auto-attach.** In a dataset where hundreds of
leads share a first name, merging two real students is far more damaging than
carrying a duplicate for a day — a merge is not cleanly reversible. Name-only
matches raise a review.

## Stages and sources as data

`lead_stages` and `lead_sources` are tables, seeded from
`contracts/vocabulary.py` with the account's real values. Admins rename and
reorder; the code does not enumerate them at runtime.

They live in code as well so that tests, the seed script and the migration
importer share one definition. Flags on each row (`is_revenue`, `is_nurture`,
`is_artefact`, `is_paid`, `is_bulk`, `is_high_intent`) drive behaviour
elsewhere: SLA tightness, scoring, funnel exclusion, ad-audience eligibility.

`Import` is flagged `is_artefact` and excluded from the funnel — it is a data
artefact in the source account, not a pipeline position.

## Time

Neither MySQL nor SQLite stores a UTC offset, so a naive datetime comes back
out and any comparison against an aware `utcnow()` raises `TypeError` — in
production, on whichever code path first compares them.

Rather than defensive conversions scattered through the domain layer, the
`UtcDateTime` column type normalises at the boundary: aware on the way in,
tagged UTC on the way out. Every timestamp column uses it.

## Consent

Four suppression flags plus opt-in provenance, on the lead. `domain/consent.py`
is the only place that reads them, and every outbound channel routes through
it, so adding a channel cannot skip the check.

Ad-audience export uses a stricter rule (`marketable()`): *any* suppression
flag disqualifies the lead entirely, because we cannot control which channel an
ad platform will use.

## Audit

`audit_log` records every write a human could later dispute — who, what, when,
against which target. Proposal decisions land here as well as on the timeline:
the timeline is the lead's story, the audit log is the operator record.
