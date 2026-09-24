from fault_tolerant_agents.metrics import (
    build_default_reliability_suite,
    build_fault_report,
    build_healthy_report,
)
from fault_tolerant_agents.schemas import (
    FaultInjectionPlan,
    FaultMode,
    RecoveryAction,
)


def make_plan(mode: FaultMode) -> FaultInjectionPlan:
    return FaultInjectionPlan(
        injection_id=f"metrics-{mode.value}",
        target_agent_id="analysis-a",
        task_id="task-analysis",
        mode=mode,
        trigger_step=2,
    )


def test_healthy_baseline_has_zero_recovery_overhead() -> None:
    report = build_healthy_report()
    metrics = report.metrics

    assert metrics.mission_success is True
    assert metrics.safe_outcome is True
    assert metrics.recovery_attempted is False
    assert metrics.recovery_success is False
    assert metrics.human_escalation is False
    assert metrics.fault_mode is None
    assert metrics.publication_time_steps == 3
    assert metrics.recovery_overhead_steps == 0
    assert metrics.winning_support_score == 4.0
    assert metrics.consensus_support_margin == 4.0


def test_fault_detection_is_immediate_in_deterministic_harness() -> None:
    report = build_fault_report(make_plan(FaultMode.TIMEOUT))

    assert report.metrics.fault_detection_steps == 0


def test_timeout_metrics_capture_retry_and_persistent_trust_penalty() -> None:
    report = build_fault_report(make_plan(FaultMode.TIMEOUT))
    metrics = report.metrics

    assert metrics.mission_success is True
    assert metrics.recovery_success is True
    assert metrics.retry_count == 1
    assert metrics.substitution_count == 0
    assert metrics.initial_trust_score == 1.0
    assert metrics.post_fault_trust_score == 0.95
    assert metrics.final_trust_score == 0.95
    assert metrics.recovery_time_steps == 1
    assert metrics.recovery_overhead_steps == 1


def test_offline_metrics_capture_substitution() -> None:
    report = build_fault_report(make_plan(FaultMode.OFFLINE))

    assert report.metrics.substitution_count == 1
    assert report.metrics.retry_count == 0
    assert report.metrics.final_trust_score == 1.0


def test_unsupported_output_metrics_capture_corroboration() -> None:
    report = build_fault_report(make_plan(FaultMode.UNSUPPORTED_OUTPUT))

    assert report.metrics.corroboration_count == 1
    assert report.metrics.post_fault_trust_score == 0.65


def test_misleading_output_metrics_capture_quarantine_and_trust_collapse() -> None:
    report = build_fault_report(make_plan(FaultMode.MISLEADING_OUTPUT))
    metrics = report.metrics

    assert metrics.quarantine_count == 1
    assert metrics.substitution_count == 1
    assert metrics.post_fault_trust_score == 0.30
    assert metrics.final_trust_score == 0.30
    assert metrics.mission_success is True


def test_failed_recovery_is_recorded_as_safe_human_escalation() -> None:
    report = build_fault_report(
        make_plan(FaultMode.OFFLINE),
        unavailable_agent_ids=frozenset({"analysis-b"}),
    )
    metrics = report.metrics

    assert metrics.mission_success is False
    assert metrics.safe_outcome is True
    assert metrics.recovery_success is False
    assert metrics.human_escalation is True
    assert metrics.escalation_count == 1
    assert metrics.winning_support_score == 0.0


def test_audit_trail_is_contiguous_and_matches_metric_count() -> None:
    report = build_fault_report(make_plan(FaultMode.ROLE_VIOLATION))

    assert [event.sequence for event in report.audit_events] == list(
        range(len(report.audit_events))
    )
    assert report.metrics.audit_event_count == len(report.audit_events)
    assert report.audit_events[0].event_type == "scenario_started"
    assert report.audit_events[-1].event_type == "mission_completed"


def test_trust_trajectory_preserves_fault_history_after_recovery() -> None:
    report = build_fault_report(make_plan(FaultMode.MALFORMED_OUTPUT))
    scores = [snapshot.score for snapshot in report.metrics.trust_trajectory]

    assert scores == [1.0, 0.8, 0.8]


def test_default_suite_contains_healthy_seven_faults_and_escalation() -> None:
    suite = build_default_reliability_suite()

    assert len(suite) == 9
    assert suite[0].scenario_id == "healthy-baseline"
    assert sum(report.metrics.mission_success for report in suite) == 8
    assert sum(report.metrics.human_escalation for report in suite) == 1


def test_reliability_reports_are_replayable() -> None:
    first = build_default_reliability_suite()
    second = build_default_reliability_suite()

    assert [report.model_dump(mode="json") for report in first] == [
        report.model_dump(mode="json") for report in second
    ]
