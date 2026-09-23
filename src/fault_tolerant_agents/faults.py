"""Deterministic fault injection for Agent 13.

This module creates known failure conditions at a specific assigned peer/task
boundary. It does not update trust, mutate health state, retry, reassign,
quarantine, recover, or invoke an LLM. Later control-plane steps will consume
these observations.
"""

from __future__ import annotations

from pydantic import ValidationError

from .baseline import (
    BaselineConfigurationError,
    build_default_team,
    build_healthy_mission,
    create_bids,
    settle_assignments,
)
from .schemas import (
    AgentDescriptor,
    AuditEvent,
    Capability,
    FaultEvent,
    FaultInjectionPlan,
    FaultMode,
    FaultObservation,
    MissionRequest,
    MissionState,
    MissionStatus,
    StrictModel,
    TaskAssignment,
    TaskSpec,
    TaskStatus,
    WorkProduct,
)


class FaultInjectionRun(StrictModel):
    """A reproducible single-fault experiment with no recovery behavior."""

    mission: MissionRequest
    agents: tuple[AgentDescriptor, ...]
    tasks: tuple[TaskSpec, ...]
    assignments: tuple[TaskAssignment, ...]
    plan: FaultInjectionPlan
    observation: FaultObservation
    state: MissionState
    audit_events: tuple[AuditEvent, ...]


def _assignment_for_plan(
    assignments: tuple[TaskAssignment, ...],
    plan: FaultInjectionPlan,
) -> TaskAssignment:
    matches = tuple(
        assignment
        for assignment in assignments
        if assignment.task_id == plan.task_id
        and assignment.agent_id == plan.target_agent_id
    )
    if len(matches) != 1:
        raise BaselineConfigurationError(
            "fault plan must target exactly one existing task assignment"
        )
    return matches[0]


def _task_for_plan(
    tasks: tuple[TaskSpec, ...],
    plan: FaultInjectionPlan,
) -> tuple[TaskSpec, int]:
    for step, task in enumerate(tasks, start=1):
        if task.task_id == plan.task_id:
            if step != plan.trigger_step:
                raise BaselineConfigurationError(
                    f"{plan.task_id} executes at step {step}, not "
                    f"{plan.trigger_step}"
                )
            return task, step
    raise BaselineConfigurationError(
        f"fault plan references unknown task {plan.task_id}"
    )


def _alternate_capability(required: Capability) -> Capability:
    for candidate in Capability:
        if candidate is not required:
            return candidate
    raise BaselineConfigurationError("no alternate capability is available")


def _structured_fault_product(
    task: TaskSpec,
    assignment: TaskAssignment,
    mode: FaultMode,
    step: int,
) -> WorkProduct:
    base = {
        "work_product_id": f"faulty-{task.task_id}-{assignment.agent_id}",
        "task_id": task.task_id,
        "producer_agent_id": assignment.agent_id,
        "capability": assignment.capability,
        "summary": "Injected deterministic fault artifact.",
        "conclusion": "mitigate",
        "evidence_ids": task.evidence_ids,
        "confidence": 0.90,
        "created_at_step": step,
    }

    if mode is FaultMode.UNSUPPORTED_OUTPUT:
        return WorkProduct(
            **{
                **base,
                "summary": (
                    "Recommendation uses only one source from the required "
                    "evidence package."
                ),
                "evidence_ids": (task.evidence_ids[0],),
            }
        )

    if mode is FaultMode.CONTRADICTORY_OUTPUT:
        return WorkProduct(
            **{
                **base,
                "summary": (
                    "The peer reaches an alternate conclusion from the same "
                    "supplied evidence."
                ),
                "conclusion": "continue",
            }
        )

    if mode is FaultMode.MISLEADING_OUTPUT:
        return WorkProduct(
            **{
                **base,
                "summary": (
                    "Safety interlocks failed and no backup conveyor is "
                    "available."
                ),
                "conclusion": "pause",
            }
        )

    if mode is FaultMode.ROLE_VIOLATION:
        return WorkProduct(
            **{
                **base,
                "capability": _alternate_capability(task.required_capability),
                "summary": (
                    "The peer produced structurally valid work under a "
                    "capability outside its assigned task boundary."
                ),
                "conclusion": "out-of-role",
            }
        )

    raise BaselineConfigurationError(
        f"{mode.value} is not a structured semantic fault"
    )


def _observe_fault(
    plan: FaultInjectionPlan,
    task: TaskSpec,
    assignment: TaskAssignment,
    step: int,
) -> FaultObservation:
    event = FaultEvent(
        fault_event_id=f"event-{plan.injection_id}",
        target_agent_id=plan.target_agent_id,
        mode=plan.mode,
        task_id=plan.task_id,
        injected=True,
        observed_at_step=step,
        detail=f"Injected {plan.mode.value} at {plan.task_id}.",
    )

    if plan.mode in {FaultMode.OFFLINE, FaultMode.TIMEOUT}:
        return FaultObservation(
            injection_id=plan.injection_id,
            assignment_id=assignment.assignment_id,
            fault_event=event,
        )

    if plan.mode is FaultMode.MALFORMED_OUTPUT:
        raw_payload: dict[str, object] = {
            "work_product_id": f"faulty-{task.task_id}-{assignment.agent_id}",
            "task_id": task.task_id,
            "producer_agent_id": assignment.agent_id,
            "capability": assignment.capability.value,
            "summary": "Malformed injected payload.",
            "conclusion": "mitigate",
            "evidence_ids": [],
            "confidence": "high",
            "created_at_step": step,
        }
        try:
            WorkProduct.model_validate(raw_payload)
        except ValidationError as exc:
            validation_error = str(exc).replace("\n", " ")[:500]
        else:
            raise BaselineConfigurationError(
                "malformed fault unexpectedly produced a valid work product"
            )
        return FaultObservation(
            injection_id=plan.injection_id,
            assignment_id=assignment.assignment_id,
            fault_event=event,
            raw_payload=raw_payload,
            validation_error=validation_error,
        )

    product = _structured_fault_product(
        task=task,
        assignment=assignment,
        mode=plan.mode,
        step=step,
    )
    return FaultObservation(
        injection_id=plan.injection_id,
        assignment_id=assignment.assignment_id,
        fault_event=event,
        work_product=product,
    )


def run_fault_injection(plan: FaultInjectionPlan) -> FaultInjectionRun:
    """Inject exactly one deterministic fault and stop without recovery."""

    mission, _, tasks = build_healthy_mission()
    agents = build_default_team()
    bids = create_bids(agents, tasks)
    assignments = settle_assignments(tasks, bids)

    task, step = _task_for_plan(tasks, plan)
    assignment = _assignment_for_plan(assignments, plan)
    observation = _observe_fault(
        plan=plan,
        task=task,
        assignment=assignment,
        step=step,
    )

    task_statuses: dict[str, TaskStatus] = {}
    for index, candidate in enumerate(tasks, start=1):
        if index < step:
            task_statuses[candidate.task_id] = TaskStatus.COMPLETED
        elif index == step:
            task_statuses[candidate.task_id] = TaskStatus.FAILED
        else:
            task_statuses[candidate.task_id] = TaskStatus.OPEN

    state = MissionState(
        mission_id=mission.mission_id,
        status=MissionStatus.FAILED_SAFE,
        task_statuses=task_statuses,
        current_recommendation=None,
        current_step=step,
    )

    audit_events = (
        AuditEvent(
            audit_event_id="fault-audit-000",
            mission_id=mission.mission_id,
            sequence=0,
            event_type="mission_started",
            detail="Fault-injection experiment started.",
        ),
        AuditEvent(
            audit_event_id="fault-audit-001",
            mission_id=mission.mission_id,
            sequence=1,
            event_type="fault_injected",
            actor_id=plan.target_agent_id,
            subject_id=plan.task_id,
            detail=f"Injected {plan.mode.value}.",
        ),
        AuditEvent(
            audit_event_id="fault-audit-002",
            mission_id=mission.mission_id,
            sequence=2,
            event_type="fault_observed",
            actor_id=plan.target_agent_id,
            subject_id=observation.fault_event.fault_event_id,
            detail=observation.fault_event.detail,
        ),
        AuditEvent(
            audit_event_id="fault-audit-003",
            mission_id=mission.mission_id,
            sequence=3,
            event_type="mission_failed_safe",
            subject_id=plan.task_id,
            detail=(
                "Experiment stopped after observing the injected fault; "
                "recovery is intentionally disabled in Step 5."
            ),
        ),
    )

    return FaultInjectionRun(
        mission=mission,
        agents=agents,
        tasks=tasks,
        assignments=assignments,
        plan=plan,
        observation=observation,
        state=state,
        audit_events=audit_events,
    )
