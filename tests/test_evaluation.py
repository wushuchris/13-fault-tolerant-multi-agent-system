from collections import Counter

from fault_tolerant_agents.evaluation import run_evaluation_suite
from fault_tolerant_agents.schemas import (
    ConsensusStatus,
    EvaluationCategory,
)


def test_suite_meets_required_scenario_counts() -> None:
    report = run_evaluation_suite()
    counts = Counter(
        result.scenario.category for result in report.results
    )

    assert counts == {
        EvaluationCategory.NORMAL_OPERATION: 3,
        EvaluationCategory.MISSING_INFORMATION: 3,
        EvaluationCategory.CONFLICTING_INFORMATION: 3,
        EvaluationCategory.FAILED_AGENT: 3,
        EvaluationCategory.MISLEADING_AGENT: 3,
        EvaluationCategory.CENTRALIZED_BASELINE: 1,
    }
    assert len(report.results) == 16


def test_all_evaluation_scenarios_match_expected_safe_behavior() -> None:
    report = run_evaluation_suite()

    assert all(result.passed for result in report.results)
    assert all(result.safe_outcome for result in report.results)


def test_missing_information_never_publishes() -> None:
    report = run_evaluation_suite()
    missing = [
        result
        for result in report.results
        if result.scenario.category
        is EvaluationCategory.MISSING_INFORMATION
    ]

    assert all(result.mission_success is False for result in missing)
    assert all(
        result.observed_consensus_status
        in {
            ConsensusStatus.INSUFFICIENT_EVIDENCE,
            ConsensusStatus.INSUFFICIENT_TRUST,
        }
        for result in missing
    )


def test_conflicting_information_includes_both_escalation_and_resolution() -> None:
    report = run_evaluation_suite()
    conflicts = [
        result
        for result in report.results
        if result.scenario.category
        is EvaluationCategory.CONFLICTING_INFORMATION
    ]

    assert sum(
        result.observed_consensus_status
        is ConsensusStatus.CORROBORATION_REQUIRED
        for result in conflicts
    ) == 2
    assert sum(result.mission_success for result in conflicts) == 1


def test_failed_agents_include_recovery_and_safe_escalation() -> None:
    report = run_evaluation_suite()
    failed_agents = [
        result
        for result in report.results
        if result.scenario.category is EvaluationCategory.FAILED_AGENT
    ]

    assert sum(result.mission_success for result in failed_agents) == 2
    assert sum(result.human_escalation for result in failed_agents) == 1


def test_misleading_agents_include_quarantine_and_safe_escalation_paths() -> None:
    report = run_evaluation_suite()
    misleading = [
        result
        for result in report.results
        if result.scenario.category
        is EvaluationCategory.MISLEADING_AGENT
    ]

    assert sum(result.mission_success for result in misleading) == 2
    assert sum(result.human_escalation for result in misleading) == 1


def test_role_violation_is_detected_but_safely_contained() -> None:
    report = run_evaluation_suite()
    role_report = next(
        item
        for item in report.role_coherence_reports
        if item.scenario_id == "misleading-role-violation"
    )
    result = next(
        item
        for item in report.results
        if item.scenario.scenario_id == "misleading-role-violation"
    )

    assert role_report.violations == 1
    assert result.role_coherent is True
    assert result.passed is True


def test_centralized_baseline_comparison_shows_redundancy_delta() -> None:
    report = run_evaluation_suite()
    comparison = report.centralized_comparison

    assert comparison.centralized_mission_success is False
    assert comparison.fault_tolerant_mission_success is True
    assert comparison.centralized_safe_outcome is True
    assert comparison.fault_tolerant_safe_outcome is True
    assert comparison.baseline_delta == 1


def test_summary_reports_recovery_and_event_metrics() -> None:
    summary = run_evaluation_suite().summary

    assert summary.average_recovery_time_steps is not None
    assert summary.average_recovery_time_steps > 0
    assert summary.average_audit_event_count > 0
    assert summary.role_coherence_rate == 1.0
    assert summary.safe_outcome_rate == 1.0


def test_release_checklist_passes() -> None:
    summary = run_evaluation_suite().summary

    assert summary.release_ready is True
    assert summary.release_failures == ()
    assert summary.failed_scenarios == 0
    assert summary.passed_scenarios == summary.total_scenarios


def test_evaluation_suite_is_replayable() -> None:
    first = run_evaluation_suite()
    second = run_evaluation_suite()

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
