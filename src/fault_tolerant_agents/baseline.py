"""Deterministic healthy-operation baseline for Agent 13.

This module intentionally models only normal operation. It establishes the
control case that later fault-injection and recovery scenarios will be compared
against. No trust mutation, failure injection, retry, quarantine, recovery, or
LLM inference occurs here.
"""

from __future__ import annotations

from collections.abc import Iterable

from .schemas import (
    AgentDescriptor,
    AgentHealthState,
    AgentRole,
    AgentTrustState,
    AuditEvent,
    Capability,
    ConsensusResult,
    ConsensusStatus,
    EvidenceItem,
    HealthStatus,
    MissionRecommendation,
    MissionRequest,
    MissionState,
    MissionStatus,
    StrictModel,
    TaskAssignment,
    TaskBid,
    TaskSpec,
    TaskStatus,
    TrustTier,
    WorkProduct,
)


class BaselineConfigurationError(ValueError):
    """Raised when a healthy baseline cannot be executed safely."""


class HealthyMissionRun(StrictModel):
    mission: MissionRequest
    agents: tuple[AgentDescriptor, ...]
    evidence: tuple[EvidenceItem, ...]
    tasks: tuple[TaskSpec, ...]
    bids: tuple[TaskBid, ...]
    assignments: tuple[TaskAssignment, ...]
    work_products: tuple[WorkProduct, ...]
    health_states: tuple[AgentHealthState, ...]
    trust_states: tuple[AgentTrustState, ...]
    consensus: ConsensusResult
    state: MissionState
    audit_events: tuple[AuditEvent, ...]


def build_default_team() -> tuple[AgentDescriptor, ...]:
    """Return the six-peer redundant team used by the MVP."""

    return (
        AgentDescriptor(
            agent_id="evidence-a",
            role=AgentRole.EVIDENCE,
            capabilities={Capability.EVIDENCE_REVIEW},
            display_name="Evidence Agent A",
        ),
        AgentDescriptor(
            agent_id="evidence-b",
            role=AgentRole.EVIDENCE,
            capabilities={Capability.EVIDENCE_REVIEW},
            display_name="Evidence Agent B",
        ),
        AgentDescriptor(
            agent_id="analysis-a",
            role=AgentRole.ANALYSIS,
            capabilities={Capability.IMPACT_ANALYSIS},
            display_name="Analysis Agent A",
        ),
        AgentDescriptor(
            agent_id="analysis-b",
            role=AgentRole.ANALYSIS,
            capabilities={Capability.IMPACT_ANALYSIS},
            display_name="Analysis Agent B",
        ),
        AgentDescriptor(
            agent_id="verification-a",
            role=AgentRole.VERIFICATION,
            capabilities={Capability.CLAIM_VERIFICATION},
            display_name="Verification Agent A",
        ),
        AgentDescriptor(
            agent_id="verification-b",
            role=AgentRole.VERIFICATION,
            capabilities={Capability.CLAIM_VERIFICATION},
            display_name="Verification Agent B",
        ),
    )


def build_healthy_mission() -> tuple[
    MissionRequest,
    tuple[EvidenceItem, ...],
    tuple[TaskSpec, ...],
]:
    """Create a public-safe synthetic operations incident.

    The scenario is intentionally structured so the healthy deterministic policy
    reaches MITIGATE. Later steps will run the same scenario with missing,
    malformed, contradictory, or misleading peer behavior.
    """

    mission = MissionRequest(
        mission_id="mission-001",
        title="Northstar Fulfillment Motor Alert",
        objective=(
            "Assess a synthetic fulfillment-center equipment incident and "
            "publish one bounded operational recommendation."
        ),
    )

    evidence = (
        EvidenceItem(
            evidence_id="evidence-001",
            source_id="source-telemetry",
            source_label="Sorter telemetry",
            content=(
                "A drive-motor temperature alert reduced sorter throughput by "
                "18 percent during the observation window."
            ),
            observed_at_step=0,
        ),
        EvidenceItem(
            evidence_id="evidence-002",
            source_id="source-safety",
            source_label="Safety controls",
            content=(
                "Safety interlocks remain normal and the synthetic incident "
                "contains no personnel injury or hazardous-material release."
            ),
            observed_at_step=0,
        ),
        EvidenceItem(
            evidence_id="evidence-003",
            source_id="source-continuity",
            source_label="Continuity status",
            content=(
                "A backup conveyor is available and can carry reduced workload "
                "while the affected motor is isolated."
            ),
            observed_at_step=0,
        ),
        EvidenceItem(
            evidence_id="evidence-004",
            source_id="source-maintenance",
            source_label="Maintenance status",
            content=(
                "A replacement motor is onsite and a controlled 30-minute "
                "replacement window is available."
            ),
            observed_at_step=0,
        ),
    )

    evidence_ids = tuple(item.evidence_id for item in evidence)
    tasks = (
        TaskSpec(
            task_id="task-evidence",
            mission_id=mission.mission_id,
            required_capability=Capability.EVIDENCE_REVIEW,
            description="Independently review the supplied incident evidence.",
            evidence_ids=evidence_ids,
            redundancy_required=2,
        ),
        TaskSpec(
            task_id="task-analysis",
            mission_id=mission.mission_id,
            required_capability=Capability.IMPACT_ANALYSIS,
            description=(
                "Independently assess the operational impact using the "
                "validated evidence package."
            ),
            evidence_ids=evidence_ids,
            depends_on_task_ids=("task-evidence",),
            redundancy_required=2,
        ),
        TaskSpec(
            task_id="task-verification",
            mission_id=mission.mission_id,
            required_capability=Capability.CLAIM_VERIFICATION,
            description=(
                "Independently verify the analysis conclusions and evidence "
                "lineage before publication."
            ),
            evidence_ids=evidence_ids,
            depends_on_task_ids=("task-analysis",),
            redundancy_required=2,
        ),
    )
    return mission, evidence, tasks


def create_bids(
    agents: Iterable[AgentDescriptor],
    tasks: Iterable[TaskSpec],
) -> tuple[TaskBid, ...]:
    """Create deterministic bids from every eligible healthy peer."""

    bids: list[TaskBid] = []
    for task in tasks:
        eligible = sorted(
            (
                agent
                for agent in agents
                if task.required_capability in agent.capabilities
            ),
            key=lambda agent: agent.agent_id,
        )
        for rank, agent in enumerate(eligible, start=1):
            bids.append(
                TaskBid(
                    bid_id=f"bid-{task.task_id}-{agent.agent_id}",
                    task_id=task.task_id,
                    agent_id=agent.agent_id,
                    capability=task.required_capability,
                    confidence=0.95,
                    normalized_cost=0.10 + (rank - 1) * 0.01,
                    available=True,
                )
            )
    return tuple(bids)


def settle_assignments(
    tasks: Iterable[TaskSpec],
    bids: Iterable[TaskBid],
) -> tuple[TaskAssignment, ...]:
    """Settle healthy bids with deterministic application-owned ordering."""

    bid_list = tuple(bids)
    assignments: list[TaskAssignment] = []

    for task in tasks:
        candidates = [
            bid
            for bid in bid_list
            if bid.task_id == task.task_id
            and bid.capability is task.required_capability
            and bid.available
        ]
        candidates.sort(
            key=lambda bid: (
                -bid.confidence,
                bid.normalized_cost,
                bid.agent_id,
            )
        )

        if len(candidates) < task.redundancy_required:
            raise BaselineConfigurationError(
                f"{task.task_id} requires {task.redundancy_required} eligible "
                f"agents but only {len(candidates)} bid"
            )

        for slot, bid in enumerate(
            candidates[: task.redundancy_required],
            start=1,
        ):
            assignments.append(
                TaskAssignment(
                    assignment_id=f"assignment-{task.task_id}-{slot}",
                    task_id=task.task_id,
                    agent_id=bid.agent_id,
                    capability=bid.capability,
                    assignment_slot=slot,
                )
            )

    return tuple(assignments)


def _healthy_recommendation(evidence_ids: set[str]) -> MissionRecommendation:
    """Return the deterministic recommendation for the synthetic fixture.

    This is deliberately a narrow application policy, not general reasoning.
    All four independently sourced facts are required to publish MITIGATE.
    """

    required = {
        "evidence-001",
        "evidence-002",
        "evidence-003",
        "evidence-004",
    }
    if required.issubset(evidence_ids):
        return MissionRecommendation.MITIGATE
    raise BaselineConfigurationError(
        "healthy scenario is missing evidence required by the deterministic policy"
    )


def _products_for_task(
    products: Iterable[WorkProduct],
    task_id: str,
) -> tuple[WorkProduct, ...]:
    return tuple(product for product in products if product.task_id == task_id)


def _make_work_product(
    task: TaskSpec,
    assignment: TaskAssignment,
    evidence: tuple[EvidenceItem, ...],
    existing_products: tuple[WorkProduct, ...],
    step: int,
) -> WorkProduct:
    evidence_ids = tuple(item.evidence_id for item in evidence)
    if not set(task.evidence_ids).issubset(set(evidence_ids)):
        raise BaselineConfigurationError(
            f"{task.task_id} references evidence unavailable to the mission"
        )

    if task.required_capability is Capability.EVIDENCE_REVIEW:
        return WorkProduct(
            work_product_id=f"wp-{task.task_id}-{assignment.agent_id}",
            task_id=task.task_id,
            producer_agent_id=assignment.agent_id,
            capability=assignment.capability,
            summary="Reviewed all four independently supplied incident records.",
            conclusion="The evidence package is complete for downstream analysis.",
            evidence_ids=task.evidence_ids,
            confidence=0.98,
            created_at_step=step,
        )

    if task.required_capability is Capability.IMPACT_ANALYSIS:
        upstream = _products_for_task(existing_products, "task-evidence")
        if len(upstream) < 2:
            raise BaselineConfigurationError(
                "analysis requires two completed evidence-review work products"
            )
        recommendation = _healthy_recommendation(set(task.evidence_ids))
        return WorkProduct(
            work_product_id=f"wp-{task.task_id}-{assignment.agent_id}",
            task_id=task.task_id,
            producer_agent_id=assignment.agent_id,
            capability=assignment.capability,
            summary=(
                "The incident reduces capacity but preserves safety controls, "
                "backup throughput, and a bounded repair path."
            ),
            conclusion=recommendation.value,
            evidence_ids=task.evidence_ids,
            input_work_product_ids=tuple(
                product.work_product_id for product in upstream
            ),
            confidence=0.92,
            created_at_step=step,
        )

    if task.required_capability is Capability.CLAIM_VERIFICATION:
        upstream = _products_for_task(existing_products, "task-analysis")
        if len(upstream) < 2:
            raise BaselineConfigurationError(
                "verification requires two completed analysis work products"
            )
        conclusions = {product.conclusion for product in upstream}
        if conclusions != {MissionRecommendation.MITIGATE.value}:
            raise BaselineConfigurationError(
                "healthy verification expected independently convergent analyses"
            )
        if any(
            not set(product.evidence_ids).issuperset(task.evidence_ids)
            for product in upstream
        ):
            raise BaselineConfigurationError(
                "analysis work product is missing required evidence lineage"
            )
        return WorkProduct(
            work_product_id=f"wp-{task.task_id}-{assignment.agent_id}",
            task_id=task.task_id,
            producer_agent_id=assignment.agent_id,
            capability=assignment.capability,
            summary=(
                "Both independent analyses preserve the required evidence "
                "lineage and converge on the same bounded recommendation."
            ),
            conclusion=f"supported:{MissionRecommendation.MITIGATE.value}",
            evidence_ids=task.evidence_ids,
            input_work_product_ids=tuple(
                product.work_product_id for product in upstream
            ),
            confidence=0.97,
            created_at_step=step,
        )

    raise BaselineConfigurationError(
        f"no healthy handler for capability {task.required_capability.value}"
    )


def run_healthy_mission() -> HealthyMissionRun:
    """Execute the deterministic six-peer healthy baseline."""

    mission, evidence, tasks = build_healthy_mission()
    agents = build_default_team()
    bids = create_bids(agents, tasks)
    assignments = settle_assignments(tasks, bids)

    health_states = tuple(
        AgentHealthState(
            agent_id=agent.agent_id,
            status=HealthStatus.HEALTHY,
            consecutive_failures=0,
            last_event_step=0,
        )
        for agent in agents
    )
    trust_states = tuple(
        AgentTrustState(
            agent_id=agent.agent_id,
            tier=TrustTier.TRUSTED,
            score=1.0,
        )
        for agent in agents
    )

    task_statuses = {task.task_id: TaskStatus.OPEN for task in tasks}
    completed_tasks: set[str] = set()
    products: list[WorkProduct] = []
    audit_events: list[AuditEvent] = []
    sequence = 0
    step = 0

    def audit(
        event_type: str,
        detail: str,
        actor_id: str | None = None,
        subject_id: str | None = None,
    ) -> None:
        nonlocal sequence
        audit_events.append(
            AuditEvent(
                audit_event_id=f"audit-{sequence:03d}",
                mission_id=mission.mission_id,
                sequence=sequence,
                event_type=event_type,
                actor_id=actor_id,
                subject_id=subject_id,
                detail=detail,
            )
        )
        sequence += 1

    audit("mission_started", "Healthy baseline mission started.")

    for task in tasks:
        missing_dependencies = set(task.depends_on_task_ids) - completed_tasks
        if missing_dependencies:
            raise BaselineConfigurationError(
                f"{task.task_id} cannot run before dependencies "
                f"{sorted(missing_dependencies)}"
            )

        task_assignments = tuple(
            assignment
            for assignment in assignments
            if assignment.task_id == task.task_id
        )
        if len(task_assignments) != task.redundancy_required:
            raise BaselineConfigurationError(
                f"{task.task_id} assignment count does not satisfy redundancy"
            )

        task_statuses[task.task_id] = TaskStatus.IN_PROGRESS
        step += 1
        for assignment in task_assignments:
            audit(
                "task_assigned",
                f"{assignment.agent_id} assigned to {task.task_id}.",
                actor_id=assignment.agent_id,
                subject_id=task.task_id,
            )
            product = _make_work_product(
                task=task,
                assignment=assignment,
                evidence=evidence,
                existing_products=tuple(products),
                step=step,
            )
            products.append(product)
            audit(
                "work_product_recorded",
                f"{product.work_product_id} recorded for {task.task_id}.",
                actor_id=product.producer_agent_id,
                subject_id=product.work_product_id,
            )

        task_statuses[task.task_id] = TaskStatus.COMPLETED
        completed_tasks.add(task.task_id)
        audit(
            "task_completed",
            f"{task.task_id} completed with required redundancy.",
            subject_id=task.task_id,
        )

    verification_products = _products_for_task(products, "task-verification")
    expected = f"supported:{MissionRecommendation.MITIGATE.value}"
    if len(verification_products) != 2 or {
        product.conclusion for product in verification_products
    } != {expected}:
        raise BaselineConfigurationError(
            "healthy baseline requires two independent supported verifications"
        )

    consensus = ConsensusResult(
        consensus_id="consensus-001",
        mission_id=mission.mission_id,
        status=ConsensusStatus.CONSENSUS_REACHED,
        recommendation=MissionRecommendation.MITIGATE,
        supporting_work_product_ids=tuple(
            product.work_product_id for product in verification_products
        ),
        rationale=(
            "Two independent analyses were independently verified against the "
            "complete evidence package."
        ),
    )
    audit(
        "consensus_reached",
        "Healthy peers reached verified consensus on MITIGATE.",
        subject_id=consensus.consensus_id,
    )
    audit("mission_completed", "Healthy baseline mission completed.")

    state = MissionState(
        mission_id=mission.mission_id,
        status=MissionStatus.COMPLETED,
        task_statuses=task_statuses,
        current_recommendation=consensus.recommendation,
        current_step=step,
    )

    return HealthyMissionRun(
        mission=mission,
        agents=agents,
        evidence=evidence,
        tasks=tasks,
        bids=bids,
        assignments=assignments,
        work_products=tuple(products),
        health_states=health_states,
        trust_states=trust_states,
        consensus=consensus,
        state=state,
        audit_events=tuple(audit_events),
    )
