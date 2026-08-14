"""Outbound consent gates.

Every outbound channel routes through :func:`check` before sending. Keeping the
rule in one function means adding a channel cannot accidentally skip it, and it
gives compliance a single place to audit under the DPDP Act.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.lead import Consent
from ..contracts.vocabulary import Channel


class ConsentDenied(RuntimeError):
    """Raised when an outbound send is attempted against a suppressed channel."""

    def __init__(self, channel: Channel, reason: str) -> None:
        super().__init__(f"{channel.value} suppressed: {reason}")
        self.channel = channel
        self.reason = reason


@dataclass(frozen=True)
class ConsentResult:
    allowed: bool
    reason: str


_FLAG_FOR_CHANNEL: dict[Channel, tuple[str, str]] = {
    Channel.CALL: ("do_not_call", "lead is on Do Not Call"),
    Channel.SMS: ("do_not_sms", "lead is on Do Not SMS"),
    Channel.EMAIL: ("do_not_email", "lead is on Do Not Email"),
    Channel.WHATSAPP: ("do_not_whatsapp", "lead has not opted in to WhatsApp"),
}


def check(consent: Consent, channel: Channel) -> ConsentResult:
    """Return whether an outbound message on ``channel`` is permitted."""
    gate = _FLAG_FOR_CHANNEL.get(channel)
    if gate is None:
        # walk_in / meeting / system are not outbound broadcast channels.
        return ConsentResult(allowed=True, reason="channel is not consent-gated")

    flag, reason = gate
    if getattr(consent, flag):
        return ConsentResult(allowed=False, reason=reason)

    if channel is Channel.WHATSAPP and consent.whatsapp_opt_in_at is None:
        return ConsentResult(allowed=False, reason="no recorded WhatsApp opt-in")

    return ConsentResult(allowed=True, reason="permitted")


def require(consent: Consent, channel: Channel) -> None:
    """Raise :class:`ConsentDenied` unless the channel is permitted."""
    result = check(consent, channel)
    if not result.allowed:
        raise ConsentDenied(channel, result.reason)


def allowed_channels(consent: Consent) -> list[Channel]:
    return [c for c in (Channel.CALL, Channel.WHATSAPP, Channel.EMAIL, Channel.SMS) if check(consent, c).allowed]


def marketable(consent: Consent) -> bool:
    """Whether a lead may be exported to an external ad audience.

    Stricter than per-channel consent: any suppression flag disqualifies the
    lead entirely, because we cannot control which channel an ad platform uses.
    """
    return not (consent.do_not_call or consent.do_not_sms or consent.do_not_email or consent.do_not_whatsapp)
