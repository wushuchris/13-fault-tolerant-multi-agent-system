"""Deterministic health and trust policy for Agent 13.

This module interprets observed faults. It does not retry work, reassign tasks,
quarantine outputs from execution, request corroboration, or recover a mission.
It only converts observations into auditable health and trust state changes.
"""

from __future__ import annotations

from .schemas import (
    AgentHealthState,
    AgentTrustState,
    FaultMode,
    FaultObservation,
    HealthStatus,
    ReliabilityAssessment,
    TrustTier,
)


POLICY_VERSION = "reliability-v1"


class ReliabilityPolicyError(ValueError):
    """Raised when reliability state cannot be safely evaluated."""


_TRUST_PENALTIES: dict[FaultMode, float] = {
    FaultMode.OFFLINE: 0.00,
    FaultMode.TIMEOUT: 0.05,
    FaultMode.MALFORMED_OUTPUT: 0.20,
    FaultMode.UNSUPPORTED_OUTPUT: 0.35,
    FaultMode.CONTRADICTORY_OUTPUT: 0.15,
    FaultMode.MISLEADING_OUTPUT: 0.70,
    FaultMode.ROLE_VIOLATION: 0.75,
}

_OPERATIONAL_FAILURES = {
    FaultMode.OFFLINE,
    FaultMode.TIMEOUT,
    FaultMode.MALFORMED_OUTPUT,
}


def initial_health_state(agent_id: str) -> AgentHealthState:
    """Create the healthy starting state for a peer."""

    return AgentHealthState(
        agent_id=agent_id,
        status=HealthStatus.HEALTHY,
        consecutive_failures=0,
        last_event_step=0,
    )


def initial_trust_state(agent_id: str) -> AgentTrustState:
    """Create the fully trusted starting state for a peer."""

    return AgentTrustState(
        agent_id=agent_id,
        tier=TrustTier.TRUSTED,
        score=1.0,
    )


def trust_tier_for_score(score: float) -> TrustTier:
    """Map an application-owned numeric score into an operational tier."""

    if score >= 0.90:
        return TrustTier.TRUSTED
    if score >= 0.70:
        return TrustTier.WATCH
    if score >= 0.40:
        return TrustTier.DEGRADED
    return TrustTier.QUARANTINED


def _health_after_fault(
    before: AgentHealthState,
    observation: FaultObservation,
) -> tuple[AgentHealthState, str]:
    mode = observation.fault_event.mode
    step = observation.fault_event.observed_at_step

    if mode is FaultMode.OFFLINE:
        return (
            AgentHealthState(
                agent_id=before.agent_id,
                status=HealthStatus.UNAVAILABLE,
                consecutive_failures=before.consecutive_failures + 1,
                last_event_step=step,
            ),
            "Peer did not respond and is currently unavailable.",
        )

    if mode is FaultMode.TIMEOUT:
        return (
            AgentHealthState(
                agent_id=before.agent_id,
                status=HealthStatus.DEGRADED,
                consecutive_failures=before.consecutive_failures + 1,
                last_event_step=step,
            ),
            "Peer exceeded the task response boundary and is operationally degraded.",
        )

    if mode is FaultMode.MALFORMED_OUTPUT:
        return (
            AgentHealthState(
                agent_id=before.agent_id,
                status=HealthStatus.DEGRADED,
                consecutive_failures=before.consecutive_failures + 1,
                last_event_step=step,
            ),
            "Peer responded, but its output failed structural validation.",
        )

    return (
        AgentHealthState(
            agent_id=before.agent_id,
            status=HealthStatus.HEALTHY,
            consecutive_failures=0,
            last_event_step=step,
        ),
        "Peer responded within the operational boundary; the observed fault is semantic.",
    )


def _trust_after_fault(
    before: AgentTrustState,
    observation: FaultObservation,
) -> tuple[AgentTrustState, str]:
    mode = observation.fault_event.mode
    penalty = _TRUST_PENALTIES[mode]
    score = round(max(0.0, before.score - penalty), 4)

    failed_verification_count = before.failed_verification_count
    contradiction_count = before.contradiction_count
    role_violation_count = before.role_violation_count

    if mode in {
        FaultMode.MALFORMED_OUTPUT,
        FaultMode.UNSUPPORTED_OUTPUT,
        FaultMode.MISLEADING_OUTPUT,
    }:
        failed_verification_count += 1

    if mode in {
        FaultMode.CONTRADICTORY_OUTPUT,
        FaultMode.MISLEADING_OUTPUT,
    }:
        contradiction_count += 1

    if mode is FaultMode.ROLE_VIOLATION:
        role_violation_count += 1

    after = AgentTrustState(
        agent_id=before.agent_id,
        tier=trust_tier_for_score(score),
        score=score,
        validated_work_count=before.validated_work_count,
        failed_verification_count=failed_verification_count,
        contradiction_count=contradiction_count,
        role_violation_count=role_violation_count,
    )

    reasons = {
        FaultMode.OFFLINE: (
            "Availability loss does not by itself imply dishonest or low-quality work."
        ),
        FaultMode.TIMEOUT: (
            "A timeout creates a small reliability penalty without treating the peer as untrustworthy."
        ),
        FaultMode.MALFORMED_OUTPUT: (
            "Structurally invalid work reduces trust because the peer failed the output contract."
        ),
        FaultMode.UNSUPPORTED_OUTPUT: (
            "A materially under-supported work product failed the evidence boundary."
        ),
        FaultMode.CONTRADICTORY_OUTPUT: (
            "A conflicting conclusion lowers trust provisionally but does not establish deception."
        ),
        FaultMode.MISLEADING_OUTPUT: (
            "The work product contradicts known synthetic evidence and receives a severe trust penalty."
        ),
        FaultMode.ROLE_VIOLATION: (
            "The peer crossed an explicit capability boundary and receives a severe trust penalty."
        ),
    }
    return after, reasons[mode]


def assess_fault(
    observation: FaultObservation,
    *,
    health_before: AgentHealthState | None = None,
    trust_before: AgentTrustState | None = None,
) -> ReliabilityAssessment:
    """Convert one observed fault into deterministic health and trust changes."""

    agent_id = observation.fault_event.target_agent_id
    health_before = health_before or initial_health_state(agent_id)
    trust_before = trust_before or initial_trust_state(agent_id)

    if health_before.agent_id != agent_id:
        raise ReliabilityPolicyError(
            "health state agent does not match the observed fault target"
        )
    if trust_before.agent_id != agent_id:
        raise ReliabilityPolicyError(
            "trust state agent does not match the observed fault target"
        )

    mode = observation.fault_event.mode
    if mode not in _TRUST_PENALTIES:
        raise ReliabilityPolicyError(f"no reliability policy for fault mode {mode.value}")

    health_after, health_reason = _health_after_fault(
        health_before,
        observation,
    )
    trust_after, trust_reason = _trust_after_fault(
        trust_before,
        observation,
    )

    return ReliabilityAssessment(
        assessment_id=f"assessment-{observation.injection_id}",
        fault_event_id=observation.fault_event.fault_event_id,
        agent_id=agent_id,
        policy_version=POLICY_VERSION,
        health_before=health_before,
        health_after=health_after,
        trust_before=trust_before,
        trust_after=trust_after,
        health_reason=health_reason,
        trust_reason=trust_reason,
    )
