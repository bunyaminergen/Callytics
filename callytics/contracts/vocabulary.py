"""The account's real vocabulary, transcribed from the live LeadSquared tenant.

Every value below was read from the production account (18 stages, 51 sources,
9 teams) rather than invented, so a migrated lead keeps the exact stage and
source string it had in LeadSquared. Managers keep their existing reports and
nobody has to relearn the pipeline on cutover day.

Adding a stage/source is a migration, not a code edit: these lists seed the
``lead_stages`` / ``lead_sources`` tables, and the DB is authoritative at
runtime. They live in code so that tests, the migration importer and the seed
script all agree on one definition.
"""

from __future__ import annotations

from enum import StrEnum

# --- pipeline stages ------------------------------------------------------
# Order is the funnel order managers already think in, not alphabetical.

STAGE_OPEN = "Open"
STAGE_JOINED = "JOINED"
STAGE_ALREADY_JOINED = "Already joined"
STAGE_CONFIRMED_WAITING = "Confirmed - Waiting for payment"

LEAD_STAGES: tuple[str, ...] = (
    "Open",
    "Did Not Pick",
    "Call back later",
    "Will check and revert",
    "Interested",
    "Qualified",
    "Confirmed - Waiting for payment",
    "JOINED",
    "Already joined",
    "Maybe in future",
    "Not able to contact",
    "Not interested",
    "Lost to competitor",
    "Language Barrier",
    "Job Inquiry",
    "Marketing Calls",
    "Junk leads",
    "Import",
)

#: Stages that count as revenue. Mirrors the LeadSquared reporting default so
#: migrated dashboards produce identical numbers.
REVENUE_STAGES: frozenset[str] = frozenset({"JOINED", "Already joined"})

#: A commitment, not an enrolment — deliberately excluded from revenue.
COMMITMENT_STAGES: frozenset[str] = frozenset({"Confirmed - Waiting for payment"})

#: Terminal stages: no SLA, no nurture, excluded from working queues.
CLOSED_LOST_STAGES: frozenset[str] = frozenset(
    {
        "Not interested",
        "Lost to competitor",
        "Junk leads",
        "Already joined",
        "Marketing Calls",
        "Job Inquiry",
    }
)

CLOSED_WON_STAGES: frozenset[str] = frozenset({"JOINED"})

#: Data artefacts rather than real pipeline positions — hidden from the funnel.
ARTEFACT_STAGES: frozenset[str] = frozenset({"Import"})

#: Stages that are dormant but legitimately re-marketable later. These are the
#: pool the ad-audience sync draws from.
NURTURE_STAGES: frozenset[str] = frozenset(
    {"Maybe in future", "Not interested", "Did Not Pick", "Not able to contact", "Lost to competitor"}
)


def is_open(stage: str) -> bool:
    return stage not in CLOSED_LOST_STAGES and stage not in CLOSED_WON_STAGES and stage not in ARTEFACT_STAGES


def is_revenue(stage: str) -> bool:
    return stage in REVENUE_STAGES


# --- lead sources ---------------------------------------------------------

LEAD_SOURCES: tuple[str, ...] = (
    "Organic Search",
    "Referral Sites",
    "Direct Traffic",
    "Social Media",
    "Inbound Email",
    "Inbound Phone call",
    "Outbound Phone call",
    "Pay per Click Ads",
    "Seminar",
    "Channel Partner Referral",
    "Employee Referral",
    "Mail Campaign",
    "SMS Campaign",
    "Newspaper Ads/Outdoor",
    "Whatsapp Enquiry - KL",
    "Website Enquiry/Chat",
    "FB Lead Ads",
    "Youtube",
    "Google Ads",
    "Assessment Test",
    "Campaign comments",
    "FB Lead Ads Manjeri",
    "GMB",
    "FB Lead Ads Bangalore",
    "eLearning Leads",
    "eLearning Leads KA",
    "TV Ad",
    "Influencers",
    "Voice Campaign",
    "FB Lead North Kerala - BLR",
    "FB Lead Ad - ACCA",
    "Webinar Leads",
    "FB Ads Tvm",
    "Referral - New admissions",
    "ACCA E-Book",
    "Classifieds",
    "Sulekha",
    "Fb Lead Ad Calicut",
    "College Data purchase",
    "JOB FAIR",
    "Seminar BLR",
    "Direct Walk-in",
    "Poetic Ads - Calicut DM",
    "Referral - Existing Students & Alumni",
    "Whatsapp Enquiry KA",
    "Google Whatsapp KL",
    "Meta Whatsapp KL",
    "Google Whatsapp KA",
    "Meta Whatsapp KA",
    "Landing Page - Course",
    "Select Source",
)

#: Paid sources cost money per lead, so they get the tightest first-touch SLA.
PAID_SOURCES: frozenset[str] = frozenset(
    {
        "Pay per Click Ads",
        "FB Lead Ads",
        "Google Ads",
        "FB Lead Ads Manjeri",
        "FB Lead Ads Bangalore",
        "FB Lead North Kerala - BLR",
        "FB Lead Ad - ACCA",
        "FB Ads Tvm",
        "Fb Lead Ad Calicut",
        "Poetic Ads - Calicut DM",
        "TV Ad",
        "Newspaper Ads/Outdoor",
        "Sulekha",
        "Classifieds",
        "Influencers",
        "College Data purchase",
        # Ad spend that lands in WhatsApp. These are deliberately in both
        # PAID_SOURCES and HIGH_INTENT_SOURCES: the click cost money (so they
        # earn the tightest SLA) and the lead then messaged us (so they score
        # as high intent). The two sets answer different questions.
        "Google Whatsapp KL",
        "Google Whatsapp KA",
        "Meta Whatsapp KL",
        "Meta Whatsapp KA",
    }
)

#: Sources that convert best, ranked from measured enrolments rather than
#: assumption. Over 2026-07-01 to 2026-08-14 (258 enrolments) the mix was:
#: Inbound Phone call 65, Whatsapp Enquiry - KL 54, Referral - New admissions
#: 43, Referral - Existing Students & Alumni 24, FB Lead Ads 22, Direct Walk-in
#: 16, Google Ads 14.
#:
#: The correction worth noting: inbound WhatsApp enquiries are the *second
#: largest* source of enrolments. They were initially filed with paid social
#: because of the channel name, which would have under-scored the account's
#: best-converting inbound route. An inbound "Hi" on WhatsApp is a person
#: raising their hand, not an impression served — it belongs here.
HIGH_INTENT_SOURCES: frozenset[str] = frozenset(
    {
        "Direct Walk-in",
        "Referral - Existing Students & Alumni",
        "Referral - New admissions",
        "Employee Referral",
        "Channel Partner Referral",
        "Inbound Phone call",
        "Website Enquiry/Chat",
        "Whatsapp Enquiry - KL",
        "Whatsapp Enquiry KA",
        "Google Whatsapp KL",
        "Google Whatsapp KA",
        "Meta Whatsapp KL",
        "Meta Whatsapp KA",
    }
)

#: Bulk-purchased or scraped lists — low intent, must never get a tight SLA.
BULK_SOURCES: frozenset[str] = frozenset({"College Data purchase", "Import", "Select Source"})


# --- teams ----------------------------------------------------------------

TEAMS: tuple[str, ...] = (
    "Center CC's",
    "HO CC FA",
    "HO CC DM",
    "Vyttila Center CC's",
    "Bangalore",
    "TVM",
    "Calicut",
    "L1",
    "online",
)


# --- channels & activity types -------------------------------------------


class Channel(StrEnum):
    CALL = "call"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    SMS = "sms"
    WALK_IN = "walk_in"
    MEETING = "meeting"
    WEB = "web"
    SYSTEM = "system"


class Direction(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"


class ActivityType(StrEnum):
    """Every row that can land on a lead's timeline.

    The timeline is append-only: an activity is a fact that happened, never a
    mutable record. Stage changes are activities too, which is what makes the
    six-month lead history in the transcript actually reconstructable.
    """

    LEAD_CREATED = "lead_created"
    LEAD_MERGED = "lead_merged"
    STAGE_CHANGED = "stage_changed"
    OWNER_CHANGED = "owner_changed"
    FIELD_CHANGED = "field_changed"

    CALL_LOGGED = "call_logged"
    CALL_RECORDING = "call_recording"
    CALL_SUMMARY = "call_summary"
    CALL_EVALUATION = "call_evaluation"

    WHATSAPP_MESSAGE = "whatsapp_message"
    WHATSAPP_INTERVENTION = "whatsapp_intervention"
    EMAIL_SENT = "email_sent"
    EMAIL_OPENED = "email_opened"
    SMS_SENT = "sms_sent"

    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    APPOINTMENT_SCHEDULED = "appointment_scheduled"
    APPOINTMENT_OUTCOME = "appointment_outcome"

    NOTE_ADDED = "note_added"
    CONSENT_CHANGED = "consent_changed"
    AUDIENCE_SYNCED = "audience_synced"

    PROPOSAL_RAISED = "proposal_raised"
    PROPOSAL_DECIDED = "proposal_decided"


#: Activity types that represent genuine two-way contact with the lead. Used by
#: the scorer and by "untouched" detection — an outbound email that was never
#: opened is not contact.
ENGAGEMENT_ACTIVITIES: frozenset[ActivityType] = frozenset(
    {
        ActivityType.CALL_LOGGED,
        ActivityType.WHATSAPP_MESSAGE,
        ActivityType.EMAIL_OPENED,
        ActivityType.APPOINTMENT_OUTCOME,
    }
)


class CallDisposition(StrEnum):
    CONNECTED = "connected"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    SWITCHED_OFF = "switched_off"
    INVALID_NUMBER = "invalid_number"
    WRONG_NUMBER = "wrong_number"
    CALLBACK_REQUESTED = "callback_requested"


CONNECTED_DISPOSITIONS: frozenset[CallDisposition] = frozenset(
    {CallDisposition.CONNECTED, CallDisposition.CALLBACK_REQUESTED}
)
