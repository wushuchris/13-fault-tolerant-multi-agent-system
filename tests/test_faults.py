import pytest

from fault_tolerant_agents.baseline import BaselineConfigurationError
from fault_tolerant_agents.faults import run_fault_injection
from fault_tolerant_agents.schemas import (
    Capability,
    FaultInjectionPlan,
    FaultMode,
    MissionStatus,
    TaskStatus,
)


def make_plan(
    mode: FaultMode,
    *,
    agent_id: str = "analysis-a",
    task_id: str = "task-analysis",
    trigger_step: int = 2,
) -> FaultInjectionPlan:
    return FaultInjectionPlan(
        injection_id=f"inject-{mode.value}",
        target_agent_id=agent_id,
        task_id=task_id,
        mode=mode,
        trigger_step=trigger_step,
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
def test_every_supported_fault_mode_is_reproducible(mode: FaultMode) -> None:
    first = run_fault_injection(make_plan(mode))
    second = run_fault_injection(make_plan(mode))

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.observation.fault_event.mode is mode
    assert first.observation.fault_event.injected is True


@pytest.mark.parametrize("mode", [FaultMode.OFFLINE, FaultMode.TIMEOUT])
def test_no_response_faults_produce_no_artifact(mode: FaultMode) -> None:
    run = run_fault_injection(make_plan(mode))

    assert run.observation.raw_payload is None
    assert run.observation.work_product is None
    assert run.observation.validation_error is None


def test_malformed_output_is_preserved_as_raw_invalid_payload() -> None:
    run = run_fault_injection(make_plan(FaultMode.MALFORMED_OUTPUT))

    assert run.observation.raw_payload is not None
    assert run.observation.work_product is None
    assert run.observation.validation_error is not None
    assert "validation error" in run.observation.validation_error.lower()


def test_unsupported_output_is_structurally_valid_but_under_cited() -> None:
    run = run_fault_injection(make_plan(FaultMode.UNSUPPORTED_OUTPUT))
    product = run.observation.work_product

    assert product is not None
    assert len(product.evidence_ids) == 1
    assert len(run.tasks[1].evidence_ids) == 4


def test_contradictory_output_is_structurally_valid_and_disagrees() -> None:
    run = run_fault_injection(make_plan(FaultMode.CONTRADICTORY_OUTPUT))
    product = run.observation.work_product

    assert product is not None
    assert product.conclusion == "continue"


def test_misleading_output_contains_known_false_synthetic_claim() -> None:
    run = run_fault_injection(make_plan(FaultMode.MISLEADING_OUTPUT))
    product = run.observation.work_product

    assert product is not None
    assert "interlocks failed" in product.summary.lower()
    assert product.conclusion == "pause"


def test_role_violation_uses_capability_outside_assignment_boundary() -> None:
    run = run_fault_injection(make_plan(FaultMode.ROLE_VIOLATION))
    product = run.observation.work_product

    assert product is not None
    assert product.capability is not Capability.IMPACT_ANALYSIS


def test_fault_run_fails_safe_without_publishing_recommendation() -> None:
    run = run_fault_injection(make_plan(FaultMode.TIMEOUT))

    assert run.state.status is MissionStatus.FAILED_SAFE
    assert run.state.current_recommendation is None
    assert run.state.task_statuses == {
        "task-evidence": TaskStatus.COMPLETED,
        "task-analysis": TaskStatus.FAILED,
        "task-verification": TaskStatus.OPEN,
    }
    assert run.audit_events[-1].event_type == "mission_failed_safe"


def test_fault_plan_must_target_existing_assignment() -> None:
    plan = make_plan(
        FaultMode.OFFLINE,
        agent_id="evidence-a",
        task_id="task-analysis",
    )

    with pytest.raises(
        BaselineConfigurationError,
        match="existing task assignment",
    ):
        run_fault_injection(plan)


def test_fault_plan_step_must_match_task_stage() -> None:
    plan = make_plan(FaultMode.TIMEOUT, trigger_step=3)

    with pytest.raises(
        BaselineConfigurationError,
        match="executes at step 2",
    ):
        run_fault_injection(plan)
