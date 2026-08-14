# Callytics design specification

Binding design documents. Where code and these documents disagree, that is a
bug in one of them — say which, and fix it.

| Doc | Subject |
|---|---|
| [01-architecture.md](01-architecture.md) | System shape, service boundaries, event flow |
| [02-data-model.md](02-data-model.md) | Schema decisions, the append-only timeline, custom fields |
| [03-intelligence.md](03-intelligence.md) | The propose/dispose model, scoring, timing, next-best-action |
| [04-migration.md](04-migration.md) | LeadSquared cutover, dual run, reconciliation |
| [05-roadmap.md](05-roadmap.md) | What is built, what is next, in what order |

## The one-paragraph version

Callytics is a lead system of record with a conversation-intelligence feed.
FinEcho (a separate service, pinned here as a submodule) turns call recordings
into structured facts. Callytics attaches those facts to a lead's append-only
timeline, derives explainable signals from them, and — where a fact implies a
pipeline change — raises a **proposal** that a named human accepts or rejects.
The system never silently rewrites a manager's pipeline, and it never produces
a number it cannot explain.

## Design rules

These are non-negotiable and each is enforced by a test:

1. **Machines propose; humans dispose.** No inference mutates lead state
   without either a recorded human decision or an explicitly enabled,
   confidence-gated auto-apply — which can never set a revenue stage.
2. **Every output carries evidence.** A score lists its contributions. A
   proposal quotes the fields that triggered it. A recommendation states its
   reason. If it cannot be explained, it does not ship.
3. **The timeline is append-only.** Activities are facts. Current state is a
   projection.
4. **Identity is resolved at write time**, on exact identifiers only.
5. **Consent is one gate.** Every outbound channel routes through
   `domain.consent`; ad exports use a stricter rule.
6. **`domain/` and `ai/` are pure.** No DB, no network, no framework.
7. **Configuration has one source.** `settings.py`, nowhere else.
