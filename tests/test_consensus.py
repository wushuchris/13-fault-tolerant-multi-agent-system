import pytest

from fault_tolerant_agents.baseline import run_healthy_mission
from fault_tolerant_agents.consensus import (
    MIN_SUPPORT_MARGIN,
    MIN_SUPPORT_SCORE,
    evaluate_consensus,
    evaluate_recovered_mission,
)
from fault_tolerant_agents.schemas import (
    AgentTrustState,
    Capability,
    ConsensusStatus,
    FaultInjectionPlan,
    FaultMode,
    MissionRecommendation,
    MissionStatus,
    TrustTier,
    WorkProduct,
)


def make_plan(
    mode: FaultMode,
    *,
    agent_id: str = "analysis-a",
    task_id: str = "task-analysis",
    step: int = 2,
) -> FaultInjectionPlan:
    return FaultInjectionPlan(
        injection_id=f"consensus-{task_id}-{mode.value}",
        target_agent_id=agent_id,
        task_id=task_id,
        mode=mode,
        trigger_step=step,
    )


@pytest.mark.parametrize(
    "mode",
    [
        FaultMode.OFFLINE,
        FaultMode.TIMEOUT,
        FaultMode.MALFORMED_OUTPUT,
        FaultMode.UNSUPPORTED_OUTPUT,
        FaultMode.CONTRADICTORY_OUTPUT,
        FaultMode.MISLEADING_OUTPUT,
        FaultMode.ROLE_VIOLATION,
    ],
)
def test_recovered_analysis_faults_can_reach_verified_consensus(
    mode: FaultMode,
) -> None:
    evaluation = evaluate_recovered_mission(make_plan(mode))

    assert evaluation.consensus.status is ConsensusStatus.CONSENSUS_REACHED
    assert evaluation.consensus.recommendation is MissionRecommendation.MITIGATE
    assert evaluation.mission_state.status is MissionStatus.COMPLETED
    assert evaluation.winning_support_score >= MIN_SUPPORT_SCORE
    assert {
        "analysis-b",
        "verification-a",
        "verification-b",
    }.issubset(set(evaluation.supporting_agent_ids))


@pytest.mark.parametrize(
    ("mode", "agent_id", "task_id", "step"),
    [
        (FaultMode.OFFLINE, "evidence-a", "task-evidence", 1),
        (FaultMode.MISLEADING_OUTPUT, "evidence-a", "task-evidence", 1),
        (FaultMode.OFFLINE, "verification-a", "task-verification", 3),
        (FaultMode.ROLE_VIOLATION, "verification-a", "task-verification", 3),
    ],
)
def test_faults_at_other_stages_can_still_publish_when_independent_support_remains(
    mode: FaultMode,
    agent_id: str,
    task_id: str,
    step: int,
) -> None:
    evaluation = evaluate_recovered_mission(
        make_plan(
            mode,
            agent_id=agent_id,
            task_id=task_id,
            step=step,
        )
    )

    assert evaluation.consensus.status is ConsensusStatus.CONSENSUS_REACHED
    assert evaluation.consensus.recommendation is MissionRecommendation.MITIGATE


def test_failed_recovery_requires_human_review_and_never_publishes() -> None:
    evaluation = evaluate_recovered_mission(
        make_plan(FaultMode.OFFLINE),
        unavailable_agent_ids=frozenset({"analysis-b"}),
    )

    assert evaluation.consensus.status is ConsensusStatus.HUMAN_REVIEW_REQUIRED
    assert evaluation.consensus.human_review_required is True
    assert evaluation.consensus.recommendation is None
    assert evaluation.mission_state.status is MissionStatus.HUMAN_REVIEW


def _product(
    product_id: str,
    agent_id: str,
    capability: Capability,
    conclusion: str,
    *,
    evidence_ids: tuple[str, ...] = ("evidence-001", "evidence-002"),
) -> WorkProduct:
    return WorkProduct(
        work_product_id=product_id,
        task_id=(
            "task-analysis"
            if capability is Capability.IMPACT_ANALYSIS
            else "task-verification"
        ),
        producer_agent_id=agent_id,
        capability=capability,
        summary="Synthetic consensus test product.",
        conclusion=conclusion,
        evidence_ids=evidence_ids,
        confidence=0.9,
        created_at_step=1,
    )


def _trust(agent_id: str, score: float) -> AgentTrustState:
    if score >= 0.90:
        tier = TrustTier.TRUSTED
    elif score >= 0.70:
        tier = TrustTier.WATCH
    elif score >= 0.40:
        tier = TrustTier.DEGRADED
    else:
        tier = TrustTier.QUARANTINED
    return AgentTrustState(
        agent_id=agent_id,
        tier=tier,
        score=score,
    )


def test_trust_weighting_can_defeat_raw_majority() -> None:
    products = (
        _product(
            "strong-analysis",
            "strong-analyst",
            Capability.IMPACT_ANALYSIS,
            "mitigate",
        ),
        _product(
            "strong-verification",
            "strong-verifier",
            Capability.CLAIM_VERIFICATION,
            "supported:mitigate",
        ),
        _product(
            "weak-analysis-a",
            "weak-a",
            Capability.IMPACT_ANALYSIS,
            "continue",
        ),
        _product(
            "weak-analysis-b",
            "weak-b",
            Capability.IMPACT_ANALYSIS,
            "continue",
        ),
        _product(
            "weak-verification",
            "weak-c",
            Capability.CLAIM_VERIFICATION,
            "supported:continue",
        ),
    )
    trust = (
        _trust("strong-analyst", 1.0),
        _trust("strong-verifier", 1.0),
        _trust("weak-a", 0.4),
        _trust("weak-b", 0.4),
        _trust("weak-c", 0.4),
    )

    evaluation = evaluate_consensus(
        mission_id="mission-test",
        allowed_recommendations=frozenset(MissionRecommendation),
        required_evidence_ids=frozenset({"evidence-001", "evidence-002"}),
        work_products=products,
        trust_states=trust,
    )

    assert len(products) == 5
    assert evaluation.consensus.status is ConsensusStatus.CONSENSUS_REACHED
    assert evaluation.consensus.recommendation is MissionRecommendation.MITIGATE
    assert evaluation.winning_support_score == 2.0
    assert evaluation.runner_up_support_score == 1.2


def test_complete_evidence_but_low_trust_is_insufficient() -> None:
    products = (
        _product(
            "low-analysis",
            "low-analyst",
            Capability.IMPACT_ANALYSIS,
            "mitigate",
        ),
        _product(
            "low-verification",
            "low-verifier",
            Capability.CLAIM_VERIFICATION,
            "supported:mitigate",
        ),
    )
    trust = (
        _trust("low-analyst", 0.6),
        _trust("low-verifier", 0.6),
    )

    evaluation = evaluate_consensus(
        mission_id="mission-test",
        allowed_recommendations=frozenset(MissionRecommendation),
        required_evidence_ids=frozenset({"evidence-001", "evidence-002"}),
        work_products=products,
        trust_states=trust,
    )

    assert evaluation.consensus.status is ConsensusStatus.INSUFFICIENT_TRUST
    assert evaluation.consensus.recommendation is None


def test_missing_evidence_is_not_outvoted() -> None:
    products = (
        _product(
            "analysis-one",
            "analyst-a",
            Capability.IMPACT_ANALYSIS,
            "mitigate",
            evidence_ids=("evidence-001",),
        ),
        _product(
            "verify-one",
            "verifier-a",
            Capability.CLAIM_VERIFICATION,
            "supported:mitigate",
            evidence_ids=("evidence-001",),
        ),
    )
    trust = (
        _trust("analyst-a", 1.0),
        _trust("verifier-a", 1.0),
    )

    evaluation = evaluate_consensus(
        mission_id="mission-test",
        allowed_recommendations=frozenset(MissionRecommendation),
        required_evidence_ids=frozenset({"evidence-001", "evidence-002"}),
        work_products=products,
        trust_states=trust,
    )

    assert evaluation.consensus.status is ConsensusStatus.INSUFFICIENT_EVIDENCE
    assert set(evaluation.excluded_work_product_ids) == {
        "analysis-one",
        "verify-one",
    }


def test_analysis_only_support_requires_verification() -> None:
    products = (
        _product(
            "analysis-a",
            "analyst-a",
            Capability.IMPACT_ANALYSIS,
            "mitigate",
        ),
        _product(
            "analysis-b",
            "analyst-b",
            Capability.IMPACT_ANALYSIS,
            "mitigate",
        ),
    )
    trust = (
        _trust("analyst-a", 1.0),
        _trust("analyst-b", 1.0),
    )

    evaluation = evaluate_consensus(
        mission_id="mission-test",
        allowed_recommendations=frozenset(MissionRecommendation),
        required_evidence_ids=frozenset({"evidence-001", "evidence-002"}),
        work_products=products,
        trust_states=trust,
    )

    assert evaluation.consensus.status is ConsensusStatus.CORROBORATION_REQUIRED


def test_close_competing_support_requires_corroboration() -> None:
    products = (
        _product(
            "mitigate-analysis",
            "analyst-a",
            Capability.IMPACT_ANALYSIS,
            "mitigate",
        ),
        _product(
            "mitigate-verify",
            "verifier-a",
            Capability.CLAIM_VERIFICATION,
            "supported:mitigate",
        ),
        _product(
            "continue-analysis",
            "analyst-b",
            Capability.IMPACT_ANALYSIS,
            "continue",
        ),
        _product(
            "continue-verify",
            "verifier-b",
            Capability.CLAIM_VERIFICATION,
            "supported:continue",
        ),
    )
    trust = (
        _trust("analyst-a", 1.0),
        _trust("verifier-a", 1.0),
        _trust("analyst-b", 0.9),
        _trust("verifier-b", 0.9),
    )

    evaluation = evaluate_consensus(
        mission_id="mission-test",
        allowed_recommendations=frozenset(MissionRecommendation),
        required_evidence_ids=frozenset({"evidence-001", "evidence-002"}),
        work_products=products,
        trust_states=trust,
    )

    assert evaluation.winning_support_score == 2.0
    assert evaluation.runner_up_support_score == 1.8
    assert evaluation.winning_support_score - evaluation.runner_up_support_score < MIN_SUPPORT_MARGIN
    assert evaluation.consensus.status is ConsensusStatus.CORROBORATION_REQUIRED


def test_quarantined_work_has_zero_publication_authority() -> None:
    products = (
        _product(
            "bad-analysis",
            "bad-agent",
            Capability.IMPACT_ANALYSIS,
            "pause",
        ),
        _product(
            "good-analysis",
            "good-agent",
            Capability.IMPACT_ANALYSIS,
            "mitigate",
        ),
        _product(
            "good-verify",
            "good-verifier",
            Capability.CLAIM_VERIFICATION,
            "supported:mitigate",
        ),
    )
    trust = (
        _trust("bad-agent", 0.2),
        _trust("good-agent", 1.0),
        _trust("good-verifier", 1.0),
    )

    evaluation = evaluate_consensus(
        mission_id="mission-test",
        allowed_recommendations=frozenset(MissionRecommendation),
        required_evidence_ids=frozenset({"evidence-001", "evidence-002"}),
        work_products=products,
        trust_states=trust,
        quarantined_agent_ids=frozenset({"bad-agent"}),
    )

    assert evaluation.consensus.recommendation is MissionRecommendation.MITIGATE
    assert "bad-analysis" in evaluation.excluded_work_product_ids


def test_healthy_baseline_satisfies_new_consensus_policy() -> None:
    healthy = run_healthy_mission()
    evaluation = evaluate_consensus(
        mission_id=healthy.mission.mission_id,
        allowed_recommendations=healthy.mission.allowed_recommendations,
        required_evidence_ids=frozenset(
            item.evidence_id for item in healthy.evidence
        ),
        work_products=healthy.work_products,
        trust_states=healthy.trust_states,
        task_statuses=healthy.state.task_statuses,
    )

    assert evaluation.consensus.status is ConsensusStatus.CONSENSUS_REACHED
    assert evaluation.consensus.recommendation is MissionRecommendation.MITIGATE
    assert evaluation.winning_support_score == 4.0


def test_recovered_consensus_is_replayable() -> None:
    plan = make_plan(FaultMode.MISLEADING_OUTPUT)

    first = evaluate_recovered_mission(plan)
    second = evaluate_recovered_mission(plan)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
