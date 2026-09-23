"""Bounded recovery policy for Agent 13.

The recovery layer consumes a deterministic fault observation plus the health
and trust assessment from Step 6. It may retry, corroborate, substitute,
quarantine, or escalate. It does not perform final mission consensus; a
successful result returns the mission to ACTIVE so the next stage can continue.
"""

from __future__ import annotations

from .baseline import run_healthy_mission
from .faults import run_fault_injection
from .reliability import assess_fault
from .schemas import (
    AgentHealthState,
    FaultInjectionPlan,
    FaultMode,
    HealthStatus,
    MissionState,
    MissionStatus,
    RecoveryAction,
    RecoveryEvent,
    RecoveryResult,
    RecoveryStatus,
    TaskStatus,
    TrustTier,
    WorkProduct,
)


class RecoveryPolicyError(ValueError):
    """Raised when a recovery plan cannot be evaluated safely."""


def _reference_product(task_id: str, agent_id: str) -> WorkProduct:
    """Fetch the deterministic healthy work product for a known peer/task."""

    healthy = run_healthy_mission()
    matches = tuple(
        product
        for product in healthy.work_products
        if product.task_id == task_id
        and product.producer_agent_id == agent_id
    )
    if len(matches) != 1:
        raise RecoveryPolicyError(
            f"expected one healthy reference product for {agent_id}/{task_id}"
        )
    return matches[0]


def _eligible_counterpart(
    plan: FaultInjectionPlan,
    unavailable_agent_ids: frozenset[str],
) -> str | None:
    """Return the other assigned peer for the same capability, if usable."""

    healthy = run_healthy_mission()
    candidates = sorted(
        assignment.agent_id
        for assignment in healthy.assignments
        if assignment.task_id == plan.task_id
        and assignment.agent_id != plan.target_agent_id
        and assignment.agent_id not in unavailable_agent_ids
    )
    return candidates[0] if candidates else None


def _copy_recovery_product(
    source: WorkProduct,
    *,
    prefix: str,
    step: int,
) -> WorkProduct:
    """Create an auditable recovery artifact from a deterministic healthy result."""

    return source.model_copy(
        update={
            "work_product_id": f"{prefix}-{source.task_id}-{source.producer_agent_id}",
            "created_at_step": step,
        }
    )


def _rejected_ids(plan: FaultInjectionPlan) -> tuple[str, ...]:
    run = run_fault_injection(plan)
    product = run.observation.work_product
    return (product.work_product_id,) if product is not None else ()


def _task_statuses(
    plan: FaultInjectionPlan,
    *,
    recovered: bool,
) -> dict[str, TaskStatus]:
    healthy = run_healthy_mission()
    statuses: dict[str, TaskStatus] = {}
    trigger = plan.trigger_step
    for step, task in enumerate(healthy.tasks, start=1):
        if step < trigger:
            statuses[task.task_id] = TaskStatus.COMPLETED
        elif step == trigger:
            statuses[task.task_id] = (
                TaskStatus.COMPLETED if recovered else TaskStatus.FAILED
            )
        else:
            statuses[task.task_id] = TaskStatus.OPEN
    return statuses


def recover_single_fault(
    plan: FaultInjectionPlan,
    *,
    retry_succeeds: bool = True,
    unavailable_agent_ids: frozenset[str] = frozenset(),
) -> RecoveryResult:
    """Attempt bounded deterministic recovery from one injected fault.

    The function intentionally supports only a single observed fault. Additional
    unavailable peers may be supplied as recovery constraints so the policy can
    demonstrate safe human escalation when no independent capability remains.
    """

    fault_run = run_fault_injection(plan)
    observation = fault_run.observation
    assessment = assess_fault(observation)
    counterpart = _eligible_counterpart(plan, unavailable_agent_ids)

    events: list[RecoveryEvent] = []
    products: list[WorkProduct] = []
    accepted_ids: list[str] = []
    rejected_ids = list(_rejected_ids(plan))
    quarantined: set[str] = set()

    def add_event(
        action: RecoveryAction,
        affected: tuple[str, ...],
        detail: str,
    ) -> None:
        events.append(
            RecoveryEvent(
                recovery_event_id=f"recovery-event-{len(events):02d}",
                trigger_fault_event_id=observation.fault_event.fault_event_id,
                action=action,
                task_id=plan.task_id,
                affected_agent_ids=affected,
                occurred_at_step=plan.trigger_step + len(events) + 1,
                detail=detail,
            )
        )

    def accept(product: WorkProduct) -> None:
        products.append(product)
        accepted_ids.append(product.work_product_id)

    mode = plan.mode
    health_after_recovery = assessment.health_after
    trust_after_recovery = assessment.trust_after

    if mode is FaultMode.OFFLINE:
        if counterpart is not None:
            add_event(
                RecoveryAction.SUBSTITUTE,
                (plan.target_agent_id, counterpart),
                "Unavailable peer replaced by the independently assigned counterpart.",
            )
            accept(
                _copy_recovery_product(
                    _reference_product(plan.task_id, counterpart),
                    prefix="substitute",
                    step=plan.trigger_step + 1,
                )
            )
        else:
            add_event(
                RecoveryAction.ESCALATE,
                (plan.target_agent_id,),
                "No independent eligible peer remains for substitution.",
            )

    elif mode in {FaultMode.TIMEOUT, FaultMode.MALFORMED_OUTPUT}:
        add_event(
            RecoveryAction.RETRY,
            (plan.target_agent_id,),
            "One bounded retry is permitted for an operational or structural failure.",
        )
        if retry_succeeds:
            retry_product = _copy_recovery_product(
                _reference_product(plan.task_id, plan.target_agent_id),
                prefix="retry",
                step=plan.trigger_step + 1,
            )
            accept(retry_product)
            health_after_recovery = AgentHealthState(
                agent_id=plan.target_agent_id,
                status=HealthStatus.HEALTHY,
                consecutive_failures=0,
                last_event_step=plan.trigger_step + 1,
            )
        elif counterpart is not None:
            add_event(
                RecoveryAction.SUBSTITUTE,
                (plan.target_agent_id, counterpart),
                "Bounded retry failed; independent counterpart substituted.",
            )
            accept(
                _copy_recovery_product(
                    _reference_product(plan.task_id, counterpart),
                    prefix="substitute",
                    step=plan.trigger_step + 2,
                )
            )
        else:
            add_event(
                RecoveryAction.ESCALATE,
                (plan.target_agent_id,),
                "Bounded retry failed and no independent substitute remains.",
            )

    elif mode in {
        FaultMode.UNSUPPORTED_OUTPUT,
        FaultMode.CONTRADICTORY_OUTPUT,
    }:
        if counterpart is not None:
            add_event(
                RecoveryAction.CORROBORATE,
                (plan.target_agent_id, counterpart),
                "Independent counterpart asked to corroborate the disputed work.",
            )
            accept(
                _copy_recovery_product(
                    _reference_product(plan.task_id, counterpart),
                    prefix="corroboration",
                    step=plan.trigger_step + 1,
                )
            )
        else:
            add_event(
                RecoveryAction.CORROBORATE,
                (plan.target_agent_id,),
                "Corroboration requested but no independent peer is available.",
            )
            add_event(
                RecoveryAction.ESCALATE,
                (plan.target_agent_id,),
                "Disputed work cannot be resolved without independent corroboration.",
            )

    elif mode in {
        FaultMode.MISLEADING_OUTPUT,
        FaultMode.ROLE_VIOLATION,
    }:
        quarantined.add(plan.target_agent_id)
        add_event(
            RecoveryAction.QUARANTINE,
            (plan.target_agent_id,),
            "Peer and disputed work excluded from recovery authority.",
        )
        if counterpart is not None:
            add_event(
                RecoveryAction.SUBSTITUTE,
                (plan.target_agent_id, counterpart),
                "Independent counterpart substituted for quarantined peer.",
            )
            accept(
                _copy_recovery_product(
                    _reference_product(plan.task_id, counterpart),
                    prefix="substitute",
                    step=plan.trigger_step + 2,
                )
            )
        else:
            add_event(
                RecoveryAction.ESCALATE,
                (plan.target_agent_id,),
                "Quarantined peer has no independent eligible substitute.",
            )

    else:
        raise RecoveryPolicyError(f"no recovery policy for {mode.value}")

    recovered = bool(accepted_ids)
    status = (
        RecoveryStatus.RECOVERED
        if recovered
        else RecoveryStatus.HUMAN_REVIEW_REQUIRED
    )
    mission_status = (
        MissionStatus.ACTIVE if recovered else MissionStatus.HUMAN_REVIEW
    )

    if trust_after_recovery.tier is TrustTier.QUARANTINED:
        quarantined.add(plan.target_agent_id)

    mission_state = MissionState(
        mission_id=fault_run.mission.mission_id,
        status=mission_status,
        task_statuses=_task_statuses(plan, recovered=recovered),
        quarantined_agent_ids=frozenset(quarantined),
        current_recommendation=None,
        current_step=(
            events[-1].occurred_at_step
            if events
            else plan.trigger_step
        ),
    )

    return RecoveryResult(
        recovery_id=f"recovery-{plan.injection_id}",
        status=status,
        assessment=assessment,
        recovery_events=tuple(events),
        recovery_work_products=tuple(products),
        accepted_work_product_ids=tuple(accepted_ids),
        rejected_work_product_ids=tuple(rejected_ids),
        quarantined_agent_ids=frozenset(quarantined),
        health_after_recovery=health_after_recovery,
        trust_after_recovery=trust_after_recovery,
        mission_state=mission_state,
    )
