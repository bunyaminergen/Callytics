# Callytics

AI-native CRM for education sales. Built to replace LeadSquared at Finprov,
with call intelligence supplied by [FinEcho](https://github.com/jesu-devs/FinEcho)
across a service boundary.

> **Repository history.** This repo previously held an unrelated open-source
> project (`bunyaminergen/Callytics`, GPL-3.0). All of that source has been
> removed; its licence is preserved at `.legacy/LICENSE.upstream-callytics`
> for the historical commits. Everything under `callytics/` is new,
> independent work.

---

## Why this exists

Finprov currently runs on LeadSquared and evaluated Meritto as a replacement.
Both are competent lead databases. Neither closes the loop between *what was
said on a call* and *what the pipeline says* — and that gap is where the
revenue leaks.

The design brief came out of two places: the live LeadSquared account (read
directly — 117 lead fields, 18 stages, 51 sources, 43 users across 9 teams) and
a recorded Meritto sales demo. The demo is worth quoting, because it named the
core problem out loud. Asked whether the AI would update a lead's stage after
reading a call, the vendor's engineer said they had built it but advised
against turning it on: *"doing that, we will never have that trust."*

That is an honest read of the risk and a poor resolution of it. Silently
rewriting a manager's pipeline from a model's reading of a phone call destroys
trust the first time it is wrong. Making counsellors do everything by hand
throws the value away.

**Callytics takes the third option: the machine proposes, a human disposes, and
every suggestion shows its evidence.**

---

## The core loop

```
Call happens
   ↓
FinEcho          transcribes (Malayalam/Kannada/English), translates,
                 summarises, scores against the sales rubric
   ↓ webhook (HMAC-signed)
Callytics        attaches the summary to the lead's timeline
   ↓
Stage inference  reads FinEcho's *structured* output — objections, promised
                 actions, disposition — and raises a Proposal
   ↓
Counsellor       sees: proposed change · confidence · quoted evidence
                 accepts or rejects in one click
   ↓
Pipeline moves   and the decision becomes a labelled training example
```

Nothing moves the pipeline on its own. Auto-apply is opt-in, confidence-gated,
and **can never set a revenue stage** — `JOINED` and `Already joined` are
human-only by construction, enforced in `domain/stages.py` and covered by test.

---

## What is different from LeadSquared and Meritto

| Capability | LeadSquared | Meritto (as demoed) | Callytics |
|---|---|---|---|
| Stage change from call content | manual | auto, "no trust" per vendor | **proposal + evidence + one-click decision** |
| Lead score | opaque number | opaque number | **additive, every point explained** |
| Best time to contact | — | "AI insights" | **windows with observation counts; refuses to guess below 5 samples** |
| Duplicate leads | nightly report | nightly report | **resolved at write time; unique identifier constraint** |
| Consent | per-channel flags | per-channel flags | **one gate every channel routes through; stricter rule for ad exports** |
| Call recordings | attachment | attachment | **structured intelligence: objections, promises, next steps, rubric score** |
| Why a lead surfaced | — | — | **ranked next-best-action with evidence** |

---

## Layout

```
callytics/contracts/     Pydantic models for every boundary (+ the account's real vocabulary)
callytics/domain/        Pure rules: identity, consent, stages, scoring, routing, SLA
callytics/ai/            Engagement timing, stage inference, next-best-action
callytics/services/      Orchestration: capture, call-intelligence ingest, proposal decisions
callytics/db/            SQLAlchemy models, repository, Alembic migrations
callytics/api/           FastAPI app, routers, auth dependencies
callytics/integrations/  FinEcho client + webhook, LeadSquared migration
vendor/finecho/          FinEcho, as a pinned git submodule (service boundary, not an import)
```

**`domain/` and `ai/` never touch the database or the network.** Every rule in
them is a pure function with a unit test, which is why the suite runs in under
two seconds with no infrastructure.

---

## Development

```bash
uv venv --python 3.12 .venv
uv pip install -e '.[dev,api]'

make verify          # ruff + pytest
make run             # uvicorn on :8080
```

Tests run against in-memory SQLite with stubbed FinEcho — no Kafka, no MySQL,
no models. Production runs MySQL; the `UtcDateTime` column type normalises
timezone handling across both.

```bash
git submodule update --init --recursive   # pull FinEcho
alembic upgrade head                      # apply schema
python -c "from callytics.bootstrap import *; ..."   # seed vocabulary
```

---

## Migration from LeadSquared

Cutover is where CRM replacements fail, so `integrations/leadsquared.py` is
built around it:

- **Verbatim values.** Stages and sources import as-is, so a migrated lead
  reads identically in both systems and existing reports still mean something.
- **No field loss.** ~15 hot fields become indexed columns; the rest become
  `field_definitions` rows that retain their original `mx_` schema name.
  Dropped fields are an explicit, tested list — not an accident.
- **Uncontactable rows are skipped, not imported**, and reported in
  `ImportStats.issues` with the reason.
- **Dual run.** `reconcile()` diffs a Callytics lead against its LeadSquared
  twin so both can run side by side until the numbers agree.

---

## Configuration

All configuration flows through `callytics/settings.py` (prefix
`CALLYTICS_`). There is no `os.getenv` anywhere else in the tree. See
`.env.example`.

Two settings govern the trust model:

| Setting | Default | Effect |
|---|---|---|
| `CALLYTICS_STAGE_PROPOSAL_AUTO_APPLY` | `false` | When false, *every* inferred stage change waits for a human |
| `CALLYTICS_STAGE_PROPOSAL_MIN_CONFIDENCE` | `0.80` | Floor for auto-apply, when enabled |

---

## Status

Working end to end and tested: lead capture with identity resolution, the
append-only timeline, explainable scoring, the FinEcho webhook, stage
inference with proposals, the decision surface, and the LeadSquared migration
mapping.

Designed and specified but not yet built: the counsellor/manager web UI, the
WhatsApp inbox with supervisor intervention, telephony (click-to-call, barge,
whisper), ad-audience sync to Meta/Google, and the Kafka worker deployment.
See `docs/design/` and the roadmap for the sequencing.
