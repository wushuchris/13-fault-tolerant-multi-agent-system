import pytest
from pydantic import ValidationError

from fault_tolerant_agents.schemas import (
    AgentDescriptor,
    AgentHealthState,
    AgentRole,
    AgentTrustState,
    Capability,
    ConsensusResult,
    ConsensusStatus,
    FaultEvent,
    FaultMode,
    HealthStatus,
    MessageEnvelope,
    MessageType,
    MissionRecommendation,
    RecoveryAction,
    RecoveryEvent,
    TaskSpec,
    TrustTier,
    WorkProduct,
)


def test_agent_descriptor_accepts_role_appropriate_capability() -> None:
    agent = AgentDescriptor(
        agent_id="evidence-a",
        role=AgentRole.EVIDENCE,
        capabilities={Capability.EVIDENCE_REVIEW},
        display_name="Evidence Agent A",
    )

    assert Capability.EVIDENCE_REVIEW in agent.capabilities


def test_agent_descriptor_rejects_capability_outside_role() -> None:
    with pytest.raises(ValidationError):
        AgentDescriptor(
            agent_id="evidence-a",
            role=AgentRole.EVIDENCE,
            capabilities={Capability.CLAIM_VERIFICATION},
            display_name="Evidence Agent A",
        )


def test_health_and_trust_are_independent_states() -> None:
    health = AgentHealthState(
        agent_id="evidence-a",
        status=HealthStatus.UNAVAILABLE,
        consecutive_failures=1,
        last_event_step=4,
    )
    trust = AgentTrustState(
        agent_id="evidence-a",
        tier=TrustTier.TRUSTED,
        score=0.95,
    )

    assert health.status is HealthStatus.UNAVAILABLE
    assert trust.tier is TrustTier.TRUSTED


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_trust_score_rejects_out_of_bounds_values(score: float) -> None:
    with pytest.raises(ValidationError):
        AgentTrustState(
            agent_id="analysis-a",
            tier=TrustTier.WATCH,
            score=score,
        )


@pytest.mark.parametrize("redundancy", [0, 3])
def test_task_redundancy_is_bounded_for_mvp(redundancy: int) -> None:
    with pytest.raises(ValidationError):
        TaskSpec(
            task_id="task-001",
            mission_id="mission-001",
            required_capability=Capability.IMPACT_ANALYSIS,
            description="Assess the operational impact.",
            redundancy_required=redundancy,
        )


def test_work_product_requires_evidence_provenance() -> None:
    with pytest.raises(ValidationError):
        WorkProduct(
            work_product_id="product-001",
            task_id="task-001",
            producer_agent_id="analysis-a",
            capability=Capability.IMPACT_ANALYSIS,
            summary="Impact assessment",
            conclusion="Mitigation is warranted.",
            evidence_ids=(),
            confidence=0.8,
            created_at_step=3,
        )


def test_message_rejects_self_as_recipient() -> None:
    with pytest.raises(ValidationError):
        MessageEnvelope(
            message_id="message-001",
            correlation_id="mission-001",
            sender_agent_id="analysis-a",
            recipient_agent_ids=("analysis-a",),
            message_type=MessageType.WORK_PRODUCT,
            sequence=1,
        )


def test_fault_event_rejects_none_mode() -> None:
    with pytest.raises(ValidationError):
        FaultEvent(
            fault_event_id="fault-001",
            target_agent_id="analysis-a",
            mode=FaultMode.NONE,
            injected=True,
            observed_at_step=2,
            detail="No fault occurred.",
        )


def test_recovery_event_captures_application_owned_action() -> None:
    event = RecoveryEvent(
        recovery_event_id="recovery-001",
        trigger_fault_event_id="fault-001",
        action=RecoveryAction.SUBSTITUTE,
        task_id="task-001",
        affected_agent_ids=("analysis-a", "analysis-b"),
        occurred_at_step=4,
        detail="Backup analyst substituted for unavailable primary.",
    )

    assert event.action is RecoveryAction.SUBSTITUTE


def test_consensus_reached_requires_recommendation() -> None:
    with pytest.raises(ValidationError):
        ConsensusResult(
            consensus_id="consensus-001",
            mission_id="mission-001",
            status=ConsensusStatus.CONSENSUS_REACHED,
            recommendation=None,
            rationale="Evidence converged.",
        )


def test_nonfinal_consensus_cannot_publish_recommendation() -> None:
    with pytest.raises(ValidationError):
        ConsensusResult(
            consensus_id="consensus-001",
            mission_id="mission-001",
            status=ConsensusStatus.CORROBORATION_REQUIRED,
            recommendation=MissionRecommendation.MONITOR,
            rationale="Independent corroboration is still required.",
        )


def test_human_review_status_requires_review_flag() -> None:
    with pytest.raises(ValidationError):
        ConsensusResult(
            consensus_id="consensus-001",
            mission_id="mission-001",
            status=ConsensusStatus.HUMAN_REVIEW_REQUIRED,
            human_review_required=False,
            rationale="Trust is insufficient for safe publication.",
        )


def test_valid_consensus_can_publish_bounded_recommendation() -> None:
    result = ConsensusResult(
        consensus_id="consensus-001",
        mission_id="mission-001",
        status=ConsensusStatus.CONSENSUS_REACHED,
        recommendation=MissionRecommendation.MITIGATE,
        supporting_work_product_ids=("product-001", "product-002"),
        rationale="Two independently verified analyses support mitigation.",
    )

    assert result.recommendation is MissionRecommendation.MITIGATE


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentHealthState(
            agent_id="verification-a",
            status=HealthStatus.HEALTHY,
            consecutive_failures=0,
            last_event_step=0,
            invented_field="not allowed",
        )
