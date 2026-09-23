import pytest

from fault_tolerant_agents.faults import run_fault_injection
from fault_tolerant_agents.reliability import (
    POLICY_VERSION,
    ReliabilityPolicyError,
    assess_fault,
    initial_health_state,
    initial_trust_state,
    trust_tier_for_score,
)
from fault_tolerant_agents.schemas import (
    AgentHealthState,
    AgentTrustState,
    FaultInjectionPlan,
    FaultMode,
    HealthStatus,
    TrustTier,
)


def make_observation(mode: FaultMode):
    plan = FaultInjectionPlan(
        injection_id=f"inject-{mode.value}",
        target_agent_id="analysis-a",
        task_id="task-analysis",
        mode=mode,
        trigger_step=2,
    )
    return run_fault_injection(plan).observation


@pytest.mark.parametrize(
    ("mode", "expected_health", "expected_score", "expected_tier"),
    [
        (FaultMode.OFFLINE, HealthStatus.UNAVAILABLE, 1.00, TrustTier.TRUSTED),
        (FaultMode.TIMEOUT, HealthStatus.DEGRADED, 0.95, TrustTier.TRUSTED),
        (FaultMode.MALFORMED_OUTPUT, HealthStatus.DEGRADED, 0.80, TrustTier.WATCH),
        (FaultMode.UNSUPPORTED_OUTPUT, HealthStatus.HEALTHY, 0.65, TrustTier.DEGRADED),
        (FaultMode.CONTRADICTORY_OUTPUT, HealthStatus.HEALTHY, 0.85, TrustTier.WATCH),
        (FaultMode.MISLEADING_OUTPUT, HealthStatus.HEALTHY, 0.30, TrustTier.QUARANTINED),
        (FaultMode.ROLE_VIOLATION, HealthStatus.HEALTHY, 0.25, TrustTier.QUARANTINED),
    ],
)
def test_fault_policy_has_explicit_health_and_trust_effects(
    mode: FaultMode,
    expected_health: HealthStatus,
    expected_score: float,
    expected_tier: TrustTier,
) -> None:
    assessment = assess_fault(make_observation(mode))

    assert assessment.policy_version == POLICY_VERSION
    assert assessment.health_after.status is expected_health
    assert assessment.trust_after.score == expected_score
    assert assessment.trust_after.tier is expected_tier


def test_offline_peer_can_remain_trusted_while_unavailable() -> None:
    assessment = assess_fault(make_observation(FaultMode.OFFLINE))

    assert assessment.health_after.status is HealthStatus.UNAVAILABLE
    assert assessment.trust_after.tier is TrustTier.TRUSTED
    assert assessment.trust_after.score == 1.0


def test_misleading_peer_can_be_healthy_but_quarantined_by_trust() -> None:
    assessment = assess_fault(make_observation(FaultMode.MISLEADING_OUTPUT))

    assert assessment.health_after.status is HealthStatus.HEALTHY
    assert assessment.trust_after.tier is TrustTier.QUARANTINED
    assert assessment.trust_after.score == 0.30


def test_semantic_fault_resets_operational_failure_streak() -> None:
    observation = make_observation(FaultMode.CONTRADICTORY_OUTPUT)
    before = AgentHealthState(
        agent_id="analysis-a",
        status=HealthStatus.DEGRADED,
        consecutive_failures=2,
        last_event_step=1,
    )

    assessment = assess_fault(observation, health_before=before)

    assert assessment.health_after.status is HealthStatus.HEALTHY
    assert assessment.health_after.consecutive_failures == 0


def test_repeated_timeouts_accumulate_without_immediate_quarantine() -> None:
    observation = make_observation(FaultMode.TIMEOUT)
    health = initial_health_state("analysis-a")
    trust = initial_trust_state("analysis-a")

    for _ in range(3):
        assessment = assess_fault(
            observation,
            health_before=health,
            trust_before=trust,
        )
        health = assessment.health_after
        trust = assessment.trust_after

    assert health.consecutive_failures == 3
    assert health.status is HealthStatus.DEGRADED
    assert trust.score == 0.85
    assert trust.tier is TrustTier.WATCH


def test_failed_verification_counter_tracks_quality_failures() -> None:
    malformed = assess_fault(make_observation(FaultMode.MALFORMED_OUTPUT))
    unsupported = assess_fault(
        make_observation(FaultMode.UNSUPPORTED_OUTPUT),
        trust_before=malformed.trust_after,
    )

    assert unsupported.trust_after.failed_verification_count == 2


def test_contradiction_counter_tracks_conflicting_and_misleading_work() -> None:
    contradictory = assess_fault(
        make_observation(FaultMode.CONTRADICTORY_OUTPUT)
    )
    misleading = assess_fault(
        make_observation(FaultMode.MISLEADING_OUTPUT),
        trust_before=contradictory.trust_after,
    )

    assert misleading.trust_after.contradiction_count == 2


def test_role_violation_counter_is_separate() -> None:
    assessment = assess_fault(make_observation(FaultMode.ROLE_VIOLATION))

    assert assessment.trust_after.role_violation_count == 1
    assert assessment.trust_after.failed_verification_count == 0


@pytest.mark.parametrize(
    ("score", "tier"),
    [
        (1.00, TrustTier.TRUSTED),
        (0.90, TrustTier.TRUSTED),
        (0.89, TrustTier.WATCH),
        (0.70, TrustTier.WATCH),
        (0.69, TrustTier.DEGRADED),
        (0.40, TrustTier.DEGRADED),
        (0.39, TrustTier.QUARANTINED),
        (0.00, TrustTier.QUARANTINED),
    ],
)
def test_trust_tier_thresholds_are_explicit(score: float, tier: TrustTier) -> None:
    assert trust_tier_for_score(score) is tier


def test_state_for_different_agent_is_rejected() -> None:
    observation = make_observation(FaultMode.TIMEOUT)
    wrong_health = initial_health_state("analysis-b")

    with pytest.raises(
        ReliabilityPolicyError,
        match="health state agent",
    ):
        assess_fault(observation, health_before=wrong_health)


def test_trust_state_for_different_agent_is_rejected() -> None:
    observation = make_observation(FaultMode.TIMEOUT)
    wrong_trust = AgentTrustState(
        agent_id="analysis-b",
        tier=TrustTier.TRUSTED,
        score=1.0,
    )

    with pytest.raises(
        ReliabilityPolicyError,
        match="trust state agent",
    ):
        assess_fault(observation, trust_before=wrong_trust)
