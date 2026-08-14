# 05 — Roadmap

## Built and tested

| Area | State |
|---|---|
| Domain rules (identity, consent, stages, scoring, routing, SLA) | complete, pure, unit-tested |
| Append-only timeline | complete |
| Lead capture with write-time dedupe | complete |
| Explainable lead scoring | complete |
| Engagement timing | complete |
| Stage inference → proposals | complete |
| Proposal decision surface | complete |
| Next-best-action | complete |
| FinEcho webhook (HMAC, idempotent) | complete |
| LeadSquared migration mapping + reconciliation | complete |
| Schema + Alembic baseline (20 tables) | complete |
| FastAPI surface | complete for the above |

82 tests, no infrastructure required.

## Next, in order

**1. Counsellor and manager portal.** The API is the whole product surface
today, which means no counsellor can use it. This is the gating item for a
pilot. Needs: lead list with saved/shared views, lead detail with timeline,
the proposal review queue, a today view driven by next-best-action.

**2. Worker deployment.** Event consumers exist as contracts and topics but run
in-process. Split into Kafka consumer groups for score recomputation,
SLA-breach detection and notification fan-out.

**3. WhatsApp inbox with supervisor intervention.** The demo's "Echo" module —
a manager sees every counsellor's conversation and can take one over, with the
handover recorded. Contract exists (`WHATSAPP_INTERVENTION`); the integration
does not.

**4. Telephony.** Click-to-call, CDR ingest, barge and whisper. `CallLog`
exists; no vendor is wired. Sequence after the portal, since the value depends
on a screen to click from.

**5. Ad-audience sync.** Push nurture-stage cohorts to Meta and Google as
hashed audiences. `AudienceSegment` and the strict consent gate are in place;
the platform clients are not. This is the demo's "re-push" feature, with
consent filtering applied *before* hashing.

**6. Learned scoring and inference.** Only once the proposal decision log has a
quarter of real accept/reject labels. Replace the rules with a model, and use
the existing rules as the baseline it must beat.

## Deliberately not doing

- **Auto-applying stage changes by default.** See `03-intelligence.md`. This is
  a product position, not a missing feature.
- **A general workflow builder.** The demo leaned heavily on "99% can be
  automated". Configurable automation is where CRMs become unmaintainable. Ship
  specific, tested automations; add more when a real need names itself.
- **Replacing FinEcho's models.** It owns conversation intelligence. Callytics
  owns the lead.

## Open questions

- **Team routing** currently balances by open-lead count. The live account
  routes by centre (Calicut, TVM, Bangalore, Vyttila, HO). Confirm whether
  language and district should hard-filter or only weight.
- **Score half-life** defaults to 14 days, unvalidated. Tune against historical
  conversion once enough history is migrated.
- **Appointment double-booking** is enforced on exact `(host, starts_at)`.
  Real calendars need overlap detection, which needs a decision on slot length.
