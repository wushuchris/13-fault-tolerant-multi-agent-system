"""Trust-aware consensus and publication authority for Agent 13.

Consensus is deterministic and application-owned. Work products do not gain
authority simply because they are numerous. Publication requires complete
evidence lineage, sufficient trust-weighted support, cross-role support from
analysis and verification, and a clear margin over competing recommendations.
"""

from __future__ import annotations

from collections import defaultdict

from .baseline import run_healthy_mission
from .faults import run_fault_injection
from .recovery import recover_single_fault
from .schemas import (
    AgentTrustState,
    Capability,
    ConsensusEvaluation,
    ConsensusResult,
    ConsensusStatus,
    FaultInjectionPlan,
    MissionRecommendation,
    MissionState,
    MissionStatus,
    RecoveryStatus,
    TaskStatus,
    TrustTier,
    WorkProduct,
)


CONSENSUS_POLICY_VERSION = "consensus-v1"
MIN_SUPPORT_SCORE = 2.0
MIN_SUPPORT_MARGIN = 0.5


class ConsensusPolicyError(ValueError):
    """Raised when consensus inputs violate deterministic policy assumptions."""


def _parse_recommendation(product: WorkProduct) -> MissionRecommendation | None:
    conclusion = product.conclusion.strip().lower()
    for recommendation in MissionRecommendation:
        if conclusion == recommendation.value:
            return recommendation
        if conclusion == f"supported:{recommendation.value}":
            return recommendation
    return None


def _support_role(product: WorkProduct) -> str | None:
    if product.capability is Capability.IMPACT_ANALYSIS:
        return "analysis"
    if product.capability is Capability.CLAIM_VERIFICATION:
        return "verification"
    return None


def evaluate_consensus(
    *,
    mission_id: str,
    allowed_recommendations: frozenset[MissionRecommendation],
    required_evidence_ids: frozenset[str],
    work_products: tuple[WorkProduct, ...],
    trust_states: tuple[AgentTrustState, ...],
    rejected_work_product_ids: frozenset[str] = frozenset(),
    quarantined_agent_ids: frozenset[str] = frozenset(),
    task_statuses: dict[str, TaskStatus] | None = None,
) -> ConsensusEvaluation:
    """Evaluate publication authority without using simple majority voting."""

    trust_by_agent = {state.agent_id: state for state in trust_states}
    if len(trust_by_agent) != len(trust_states):
        raise ConsensusPolicyError("duplicate trust state for agent")

    score_by_rec: dict[MissionRecommendation, float] = defaultdict(float)
    supporters_by_rec: dict[MissionRecommendation, set[str]] = defaultdict(set)
    roles_by_rec: dict[MissionRecommendation, set[str]] = defaultdict(set)
    products_by_rec: dict[MissionRecommendation, list[str]] = defaultdict(list)
    excluded: set[str] = set()
    eligible: set[str] = set()
    parseable_complete_exists = False
    parseable_incomplete_exists = False

    for product in work_products:
        recommendation = _parse_recommendation(product)
        if recommendation is None or recommendation not in allowed_recommendations:
            continue

        complete_evidence = required_evidence_ids.issubset(set(product.evidence_ids))
        if complete_evidence:
            parseable_complete_exists = True
        else:
            parseable_incomplete_exists = True
            excluded.add(product.work_product_id)
            continue

        if product.work_product_id in rejected_work_product_ids:
            excluded.add(product.work_product_id)
            continue

        if product.producer_agent_id in quarantined_agent_ids:
            excluded.add(product.work_product_id)
            continue

        trust = trust_by_agent.get(product.producer_agent_id)
        if trust is None or trust.tier is TrustTier.QUARANTINED:
            excluded.add(product.work_product_id)
            continue

        role = _support_role(product)
        if role is None:
            excluded.add(product.work_product_id)
            continue

        # One peer contributes at most once to a recommendation, even if duplicate
        # work products are present.
        if product.producer_agent_id in supporters_by_rec[recommendation]:
            excluded.add(product.work_product_id)
            continue

        supporters_by_rec[recommendation].add(product.producer_agent_id)
        roles_by_rec[recommendation].add(role)
        products_by_rec[recommendation].append(product.work_product_id)
        score_by_rec[recommendation] = round(
            score_by_rec[recommendation] + trust.score,
            4,
        )
        eligible.add(product.work_product_id)

    ranked = sorted(
        score_by_rec.items(),
        key=lambda item: (-item[1], item[0].value),
    )
    winning_rec = ranked[0][0] if ranked else None
    winning_score = ranked[0][1] if ranked else 0.0
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = round(winning_score - runner_up_score, 4)

    status: ConsensusStatus
    rationale: str
    recommendation: MissionRecommendation | None = None

    if not parseable_complete_exists:
        status = ConsensusStatus.INSUFFICIENT_EVIDENCE
        rationale = (
            "No candidate recommendation has complete evidence provenance."
            if parseable_incomplete_exists
            else "No publishable recommendation-bearing work product is available."
        )
    elif winning_rec is None or winning_score < MIN_SUPPORT_SCORE:
        status = ConsensusStatus.INSUFFICIENT_TRUST
        rationale = (
            "Complete evidence exists, but eligible trust-weighted support is "
            "below the publication threshold."
        )
    elif not {"analysis", "verification"}.issubset(roles_by_rec[winning_rec]):
        status = ConsensusStatus.CORROBORATION_REQUIRED
        rationale = (
            "The leading recommendation lacks independent cross-role support "
            "from both analysis and verification."
        )
    elif runner_up_score > 0 and margin < MIN_SUPPORT_MARGIN:
        status = ConsensusStatus.CORROBORATION_REQUIRED
        rationale = (
            "Competing recommendations remain too close under trust-weighted "
            "support to publish safely."
        )
    else:
        status = ConsensusStatus.CONSENSUS_REACHED
        recommendation = winning_rec
        rationale = (
            "Complete evidence, cross-role verification, sufficient weighted "
            "support, and publication margin requirements are satisfied."
        )

    supporting_ids = (
        tuple(products_by_rec[winning_rec])
        if winning_rec is not None
        else ()
    )
    dissenting_ids: list[str] = []
    for rec, product_ids in products_by_rec.items():
        if rec != winning_rec:
            dissenting_ids.extend(product_ids)

    # Rejected recommendation-bearing products remain visible as dissent/audit
    # artifacts even though they contribute zero authority.
    for product in work_products:
        parsed = _parse_recommendation(product)
        if (
            parsed is not None
            and winning_rec is not None
            and parsed != winning_rec
            and product.work_product_id in rejected_work_product_ids
        ):
            dissenting_ids.append(product.work_product_id)

    consensus = ConsensusResult(
        consensus_id=f"consensus-{mission_id}",
        mission_id=mission_id,
        status=status,
        recommendation=recommendation,
        supporting_work_product_ids=supporting_ids,
        dissenting_work_product_ids=tuple(dict.fromkeys(dissenting_ids)),
        human_review_required=False,
        rationale=rationale,
    )

    state = MissionState(
        mission_id=mission_id,
        status=(
            MissionStatus.COMPLETED
            if status is ConsensusStatus.CONSENSUS_REACHED
            else MissionStatus.ACTIVE
        ),
        task_statuses=task_statuses or {},
        quarantined_agent_ids=quarantined_agent_ids,
        current_recommendation=recommendation,
        current_step=4 if status is ConsensusStatus.CONSENSUS_REACHED else 3,
    )

    return ConsensusEvaluation(
        policy_version=CONSENSUS_POLICY_VERSION,
        consensus=consensus,
        mission_state=state,
        winning_support_score=winning_score,
        runner_up_support_score=runner_up_score,
        required_support_score=MIN_SUPPORT_SCORE,
        required_margin=MIN_SUPPORT_MARGIN,
        supporting_agent_ids=(
            tuple(sorted(supporters_by_rec[winning_rec]))
            if winning_rec is not None
            else ()
        ),
        eligible_work_product_ids=tuple(sorted(eligible)),
        excluded_work_product_ids=tuple(sorted(excluded)),
    )


def _dedupe_by_producer(
    products: tuple[WorkProduct, ...],
) -> tuple[WorkProduct, ...]:
    by_agent: dict[str, WorkProduct] = {}
    for product in products:
        by_agent[product.producer_agent_id] = product
    return tuple(by_agent[key] for key in sorted(by_agent))


def _regenerate_analysis_products(
    evidence_product_ids: tuple[str, ...],
) -> tuple[WorkProduct, ...]:
    healthy = run_healthy_mission()
    templates = tuple(
        product
        for product in healthy.work_products
        if product.task_id == "task-analysis"
    )
    return tuple(
        template.model_copy(
            update={
                "work_product_id": (
                    f"post-recovery-task-analysis-{template.producer_agent_id}"
                ),
                "input_work_product_ids": evidence_product_ids,
                "created_at_step": 4,
            }
        )
        for template in templates
    )


def _regenerate_verification_products(
    analysis_product_ids: tuple[str, ...],
) -> tuple[WorkProduct, ...]:
    healthy = run_healthy_mission()
    templates = tuple(
        product
        for product in healthy.work_products
        if product.task_id == "task-verification"
    )
    return tuple(
        template.model_copy(
            update={
                "work_product_id": (
                    f"post-recovery-task-verification-{template.producer_agent_id}"
                ),
                "input_work_product_ids": analysis_product_ids,
                "created_at_step": 5,
            }
        )
        for template in templates
    )


def _human_review_evaluation(
    plan: FaultInjectionPlan,
    rejected_ids: frozenset[str],
    quarantined_ids: frozenset[str],
) -> ConsensusEvaluation:
    healthy = run_healthy_mission()
    consensus = ConsensusResult(
        consensus_id=f"consensus-{healthy.mission.mission_id}",
        mission_id=healthy.mission.mission_id,
        status=ConsensusStatus.HUMAN_REVIEW_REQUIRED,
        recommendation=None,
        human_review_required=True,
        rationale=(
            "Recovery could not restore sufficient independent capability for "
            "safe automated publication."
        ),
    )
    state = MissionState(
        mission_id=healthy.mission.mission_id,
        status=MissionStatus.HUMAN_REVIEW,
        task_statuses={
            task.task_id: (
                TaskStatus.FAILED
                if task.task_id == plan.task_id
                else (
                    TaskStatus.COMPLETED
                    if healthy.tasks.index(task) + 1 < plan.trigger_step
                    else TaskStatus.OPEN
                )
            )
            for task in healthy.tasks
        },
        quarantined_agent_ids=quarantined_ids,
        current_recommendation=None,
        current_step=plan.trigger_step,
    )
    return ConsensusEvaluation(
        policy_version=CONSENSUS_POLICY_VERSION,
        consensus=consensus,
        mission_state=state,
        winning_support_score=0.0,
        runner_up_support_score=0.0,
        required_support_score=MIN_SUPPORT_SCORE,
        required_margin=MIN_SUPPORT_MARGIN,
        supporting_agent_ids=(),
        eligible_work_product_ids=(),
        excluded_work_product_ids=tuple(sorted(rejected_ids)),
    )


def evaluate_recovered_mission(
    plan: FaultInjectionPlan,
    *,
    retry_succeeds: bool = True,
    unavailable_agent_ids: frozenset[str] = frozenset(),
) -> ConsensusEvaluation:
    """Continue a recovered single-fault mission through publication policy."""

    recovery = recover_single_fault(
        plan,
        retry_succeeds=retry_succeeds,
        unavailable_agent_ids=unavailable_agent_ids,
    )
    healthy = run_healthy_mission()

    if recovery.status is RecoveryStatus.HUMAN_REVIEW_REQUIRED:
        return _human_review_evaluation(
            plan,
            frozenset(recovery.rejected_work_product_ids),
            recovery.quarantined_agent_ids,
        )

    trust_by_agent = {state.agent_id: state for state in healthy.trust_states}
    trust_by_agent[plan.target_agent_id] = recovery.trust_after_recovery
    trust_states = tuple(
        trust_by_agent[agent_id]
        for agent_id in sorted(trust_by_agent)
    )

    healthy_evidence_products = tuple(
        product
        for product in healthy.work_products
        if product.task_id == "task-evidence"
    )
    healthy_analysis_products = tuple(
        product
        for product in healthy.work_products
        if product.task_id == "task-analysis"
    )
    healthy_verification_products = tuple(
        product
        for product in healthy.work_products
        if product.task_id == "task-verification"
    )

    if plan.task_id == "task-evidence":
        evidence_products = _dedupe_by_producer(
            tuple(
                product
                for product in healthy_evidence_products
                if product.producer_agent_id != plan.target_agent_id
            )
            + recovery.recovery_work_products
        )
        analysis_products = _regenerate_analysis_products(
            tuple(product.work_product_id for product in evidence_products)
        )
        verification_products = _regenerate_verification_products(
            tuple(product.work_product_id for product in analysis_products)
        )
    elif plan.task_id == "task-analysis":
        evidence_products = healthy_evidence_products
        analysis_products = _dedupe_by_producer(
            tuple(
                product
                for product in healthy_analysis_products
                if product.producer_agent_id != plan.target_agent_id
            )
            + recovery.recovery_work_products
        )
        verification_products = _regenerate_verification_products(
            tuple(product.work_product_id for product in analysis_products)
        )
    elif plan.task_id == "task-verification":
        evidence_products = healthy_evidence_products
        analysis_products = healthy_analysis_products
        verification_products = _dedupe_by_producer(
            tuple(
                product
                for product in healthy_verification_products
                if product.producer_agent_id != plan.target_agent_id
            )
            + recovery.recovery_work_products
        )
    else:
        raise ConsensusPolicyError(
            f"unknown recovery task {plan.task_id}"
        )

    fault_product = run_fault_injection(plan).observation.work_product
    all_products = (
        evidence_products
        + analysis_products
        + verification_products
        + ((fault_product,) if fault_product is not None else ())
    )

    required_evidence_ids = frozenset(
        item.evidence_id for item in healthy.evidence
    )
    task_statuses = {
        task.task_id: TaskStatus.COMPLETED for task in healthy.tasks
    }

    return evaluate_consensus(
        mission_id=healthy.mission.mission_id,
        allowed_recommendations=healthy.mission.allowed_recommendations,
        required_evidence_ids=required_evidence_ids,
        work_products=all_products,
        trust_states=trust_states,
        rejected_work_product_ids=frozenset(
            recovery.rejected_work_product_ids
        ),
        quarantined_agent_ids=recovery.quarantined_agent_ids,
        task_statuses=task_statuses,
    )
