# 03 — Intelligence

This is the part that has to be better than the alternatives, so it is worth
being precise about what "better" means. Not more models. **More decisions a
human can check.**

## The propose/dispose model

### The problem, stated by the competitor

In the Meritto demo, the prospect asked directly: after a counsellor finishes a
call, do they change the stage manually, or does the AI read the call and
change it? The vendor's engineer answered that they had built the automatic
version but recommended against enabling it —

> "doing that, we will never have that trust. Because speaking with them, in
> what way to change, we can't say, right? ... what builds a bit more trust is
> manually changing the stages."

He is right about the risk. A model that silently rewrites the pipeline is
wrong in public, on a manager's revenue report, and the trust never comes back.
But "so do it all by hand" discards the value *and* produces no data with which
to ever improve.

### The resolution

Both options are bad because they collapse a two-step process into one. Split
it:

| | Inference | Decision |
|---|---|---|
| Who | machine | named human |
| Output | a `Proposal` | an applied change |
| Recorded | always | always, with actor |
| Reversible | n/a — nothing changed | yes, via the stage machine |

A `Proposal` carries four things, and the fourth is the one that matters:

1. the exact change (`Open` → `Confirmed - Waiting for payment`),
2. a confidence,
3. a rationale,
4. **evidence** — the specific FinEcho fields that triggered it, quoted.

A counsellor sees *"promised action: 'Student will pay the admission fee on
Monday'"* and decides in two seconds without opening the recording. That is a
faster interaction than editing a dropdown, and it is auditable.

### What auto-apply is allowed to do

Off by default. When an administrator enables it, two guards remain:

- a configurable confidence floor (`0.80` by default), and
- **revenue stages are never eligible.** `JOINED` and `Already joined` are
  human-only in `domain/stages.py`, enforced regardless of confidence and
  covered by `test_auto_apply_never_touches_revenue_stages`.

A model may observe that a student said they enrolled. It may not book the
revenue.

### The compounding benefit

Every accept and every reject is stored with its evidence. After a quarter that
is a labelled dataset of real counsellor judgements on real calls — exactly
what is needed to replace the current rule-based inference with a learned one,
and to measure whether that would be an improvement. The manual-only approach
generates nothing.

## Why inference reads structured output, not transcripts

`ai/stage_inference.py` matches against FinEcho's *structured* fields —
`objections`, `promised_actions`, `suggested_next_steps`, `call_type`,
disposition — never raw transcript text.

- It keeps this module deterministic and unit-testable.
- It keeps LLM judgement inside the service that owns the model, its prompts
  and its grounding checks.
- It means a change to FinEcho's model does not silently change CRM behaviour;
  it changes the inputs, which the contract validates.

Corroboration across independent fields raises confidence, capped at 0.95.
Nothing is ever certain.

## Lead scoring

Additive and explainable. `LeadScore` returns a list of `ScoreContribution`,
each with a feature name, points and a human-readable detail. The published
score is exactly the clamped sum — tested, so the explanation can never drift
into decoration.

Feature groups: source quality, funnel position, two-way engagement, recency
decay, commitment signals (appointments kept/missed), profile completeness,
conversation quality from FinEcho's rubric score, and hard negatives
(suppression, repeated no-connects, unresolved objections).

Two deliberate choices:

- **Recency is exponential decay, not a window.** A 30-day window makes a lead
  worth full value on day 29 and nothing on day 31. Decay with a configurable
  half-life is smooth and tunable.
- **A weak call hurts less than a strong call helps.** A low rubric score is
  usually a *coaching* signal about the counsellor, not a signal about the
  lead's intent. Treating it symmetrically would penalise leads for being
  handled badly.

Rules rather than a learned model, for now, because the score must be arguable
by the sales head without a retraining cycle — and because this produces the
features a learned model would need later.

## Engagement timing

The competitor pitched "AI insights: at what time they are active on WhatsApp,
at what time email is active". The underlying idea is good and addresses a
real, measurable metric — call connection rate.

The execution detail that decides whether it is useful: **honesty about sample
size.** A confident-sounding wrong window is worse than no window, because
counsellors will follow it and lose calls.

So `EngagementProfile` reports `total_observations` and `sufficient_data`, and
emits no windows below five observations. Only genuine *responses* count —
inbound messages, answered calls, email opens. An outbound call nobody picked
up says nothing about when the lead is free.

## Next-best-action

Ranked by **urgency of obligation**, not lead value:

| Priority | Trigger |
|---|---|
| 0 | a promise made on a call and not kept |
| 1 | an inbound message with no reply (past a 30-minute grace) |
| 2 | first-touch SLA about to breach or breached |
| 3 | a due task or an appointment within 24h |
| 4 | a warm lead currently inside its known contact window |
| 5 | routine follow-up on a going-cold lead |

Sorting by score alone buries the lead who asked a question two hours ago
underneath a high-scoring lead who is unreachable until evening. A promise the
company made outranks both.

Every action names its channel — filtered through `domain.consent`, so a
Do-Not-Call lead is never recommended for a call — and carries its evidence.
