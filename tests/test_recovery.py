import pytest

from fault_tolerant_agents.recovery import recover_single_fault
from fault_tolerant_agents.schemas import (
    FaultInjectionPlan,
    FaultMode,
    HealthStatus,
    MissionStatus,
    RecoveryAction,
    RecoveryStatus,
    TaskStatus,
    TrustTier,
)


def make_plan(mode: FaultMode) -> FaultInjectionPlan:
    return FaultInjectionPlan(
        injection_id=f"recover-{mode.value}",
        target_agent_id="analysis-a",
        task_id="task-analysis",
        mode=mode,
        trigger_step=2,
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
def test_default_single_fault_scenarios_recover(mode: FaultMode) -> None:
    result = recover_single_fault(make_plan(mode))

    assert result.status is RecoveryStatus.RECOVERED
    assert result.mission_state.status is MissionStatus.ACTIVE
    assert result.mission_state.current_recommendation is None
    assert result.accepted_work_product_ids
    assert result.mission_state.task_statuses["task-analysis"] is TaskStatus.COMPLETED


def test_offline_uses_independent_substitute_without_erasing_unavailability() -> None:
    result = recover_single_fault(make_plan(FaultMode.OFFLINE))

    assert [event.action for event in result.recovery_events] == [
        RecoveryAction.SUBSTITUTE
    ]
    assert result.recovery_work_products[0].producer_agent_id == "analysis-b"
    assert result.health_after_recovery.status is HealthStatus.UNAVAILABLE
    assert result.trust_after_recovery.tier is TrustTier.TRUSTED


def test_timeout_uses_one_bounded_retry_and_restores_health_only() -> None:
    result = recover_single_fault(make_plan(FaultMode.TIMEOUT))

    assert [event.action for event in result.recovery_events] == [
        RecoveryAction.RETRY
    ]
    assert result.recovery_work_products[0].producer_agent_id == "analysis-a"
    assert result.health_after_recovery.status is HealthStatus.HEALTHY
    assert result.trust_after_recovery.score == 0.95


def test_malformed_output_retry_does_not_instantly_restore_trust() -> None:
    result = recover_single_fault(make_plan(FaultMode.MALFORMED_OUTPUT))

    assert result.health_after_recovery.status is HealthStatus.HEALTHY
    assert result.trust_after_recovery.tier is TrustTier.WATCH
    assert result.trust_after_recovery.score == 0.80


@pytest.mark.parametrize(
    "mode",
    [FaultMode.UNSUPPORTED_OUTPUT, FaultMode.CONTRADICTORY_OUTPUT],
)
def test_disputed_semantic_work_requires_independent_corroboration(
    mode: FaultMode,
) -> None:
    result = recover_single_fault(make_plan(mode))

    assert [event.action for event in result.recovery_events] == [
        RecoveryAction.CORROBORATE
    ]
    assert result.recovery_work_products[0].producer_agent_id == "analysis-b"
    assert result.rejected_work_product_ids == (
        "faulty-task-analysis-analysis-a",
    )


@pytest.mark.parametrize(
    "mode",
    [FaultMode.MISLEADING_OUTPUT, FaultMode.ROLE_VIOLATION],
)
def test_severe_semantic_fault_quarantines_then_substitutes(
    mode: FaultMode,
) -> None:
    result = recover_single_fault(make_plan(mode))

    assert [event.action for event in result.recovery_events] == [
        RecoveryAction.QUARANTINE,
        RecoveryAction.SUBSTITUTE,
    ]
    assert result.quarantined_agent_ids == frozenset({"analysis-a"})
    assert result.mission_state.quarantined_agent_ids == frozenset(
        {"analysis-a"}
    )
    assert result.recovery_work_products[0].producer_agent_id == "analysis-b"
    assert result.trust_after_recovery.tier is TrustTier.QUARANTINED


@pytest.mark.parametrize(
    "mode",
    [FaultMode.TIMEOUT, FaultMode.MALFORMED_OUTPUT],
)
def test_failed_retry_falls_back_to_independent_substitute(
    mode: FaultMode,
) -> None:
    result = recover_single_fault(make_plan(mode), retry_succeeds=False)

    assert [event.action for event in result.recovery_events] == [
        RecoveryAction.RETRY,
        RecoveryAction.SUBSTITUTE,
    ]
    assert result.recovery_work_products[0].producer_agent_id == "analysis-b"


def test_no_substitute_after_offline_escalates_to_human_review() -> None:
    result = recover_single_fault(
        make_plan(FaultMode.OFFLINE),
        unavailable_agent_ids=frozenset({"analysis-b"}),
    )

    assert result.status is RecoveryStatus.HUMAN_REVIEW_REQUIRED
    assert result.mission_state.status is MissionStatus.HUMAN_REVIEW
    assert result.mission_state.current_recommendation is None
    assert result.accepted_work_product_ids == ()
    assert result.recovery_events[-1].action is RecoveryAction.ESCALATE
    assert result.mission_state.task_statuses["task-analysis"] is TaskStatus.FAILED


def test_failed_retry_without_substitute_escalates() -> None:
    result = recover_single_fault(
        make_plan(FaultMode.TIMEOUT),
        retry_succeeds=False,
        unavailable_agent_ids=frozenset({"analysis-b"}),
    )

    assert [event.action for event in result.recovery_events] == [
        RecoveryAction.RETRY,
        RecoveryAction.ESCALATE,
    ]
    assert result.status is RecoveryStatus.HUMAN_REVIEW_REQUIRED


def test_unresolved_contradiction_without_peer_escalates() -> None:
    result = recover_single_fault(
        make_plan(FaultMode.CONTRADICTORY_OUTPUT),
        unavailable_agent_ids=frozenset({"analysis-b"}),
    )

    assert [event.action for event in result.recovery_events] == [
        RecoveryAction.CORROBORATE,
        RecoveryAction.ESCALATE,
    ]
    assert result.status is RecoveryStatus.HUMAN_REVIEW_REQUIRED


def test_quarantined_peer_without_substitute_escalates() -> None:
    result = recover_single_fault(
        make_plan(FaultMode.MISLEADING_OUTPUT),
        unavailable_agent_ids=frozenset({"analysis-b"}),
    )

    assert [event.action for event in result.recovery_events] == [
        RecoveryAction.QUARANTINE,
        RecoveryAction.ESCALATE,
    ]
    assert result.status is RecoveryStatus.HUMAN_REVIEW_REQUIRED
    assert result.quarantined_agent_ids == frozenset({"analysis-a"})


def test_recovery_is_replayable() -> None:
    plan = make_plan(FaultMode.MISLEADING_OUTPUT)

    first = recover_single_fault(plan)
    second = recover_single_fault(plan)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
