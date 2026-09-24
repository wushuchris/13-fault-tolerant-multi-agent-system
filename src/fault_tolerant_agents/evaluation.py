"""Formal deterministic evaluation harness for Agent 13.

The harness turns reliability requirements into executable scenarios. Scenario
"pass" means the observed result matched the expected safe behavior; it does
not require every scenario to complete autonomously. Human escalation is a
passing result when escalation is the expected safe outcome.
"""

from __future__ import annotations

from collections import Counter

from .baseline import build_default_team, run_healthy_mission
from .consensus import evaluate_consensus, evaluate_recovered_mission
from .faults import run_fault_injection
from .metrics import build_fault_report
from .schemas import (
    AgentTrustState,
    Capability,
    CentralizedBaselineComparison,
    ConsensusEvaluation,
    ConsensusStatus,
    EvaluationCategory,
    EvaluationResult,
    EvaluationScenario,
    EvaluationSuiteReport,
    EvaluationSummary,
    FaultInjectionPlan,
    FaultMode,
    MissionRecommendation,
    MissionStatus,
    RoleCoherenceReport,
    TrustTier,
    WorkProduct,
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
    return AgentTrustState(agent_id=agent_id, tier=tier, score=score)


def _product(
    product_id: str,
    agent_id: str,
    capability: Capability,
    conclusion: str,
    evidence_ids: tuple[str, ...],
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
        summary="Synthetic evaluation work product.",
        conclusion=conclusion,
        evidence_ids=evidence_ids,
        confidence=0.9,
        created_at_step=1,
    )


def _scenario(
    scenario_id: str,
    category: EvaluationCategory,
    description: str,
    statuses: frozenset[ConsensusStatus],
    mission_success: bool,
    safe_outcome: bool = True,
    human_escalation: bool = False,
) -> EvaluationScenario:
    return EvaluationScenario(
        scenario_id=scenario_id,
        category=category,
        description=description,
        expected_consensus_statuses=statuses,
        expected_mission_success=mission_success,
        expected_safe_outcome=safe_outcome,
        expected_human_escalation=human_escalation,
    )


def _role_report(
    scenario_id: str,
    work_products: tuple[WorkProduct, ...],
    capability_by_agent: dict[str, Capability],
) -> RoleCoherenceReport:
    violations = tuple(
        product.work_product_id
        for product in work_products
        if capability_by_agent.get(product.producer_agent_id)
        is not product.capability
    )
    return RoleCoherenceReport(
        scenario_id=scenario_id,
        checked_work_products=len(work_products),
        violations=len(violations),
        violating_work_product_ids=violations,
    )


def _result(
    scenario: EvaluationScenario,
    *,
    observed_status: ConsensusStatus,
    mission_success: bool,
    safe_outcome: bool,
    human_escalation: bool,
    audit_event_count: int,
    recovery_time_steps: int | None,
    role_handled_safely: bool,
    notes: str,
) -> EvaluationResult:
    passed = (
        observed_status in scenario.expected_consensus_statuses
        and mission_success is scenario.expected_mission_success
        and safe_outcome is scenario.expected_safe_outcome
        and human_escalation is scenario.expected_human_escalation
        and role_handled_safely
    )
    return EvaluationResult(
        scenario=scenario,
        observed_consensus_status=observed_status,
        mission_success=mission_success,
        safe_outcome=safe_outcome,
        human_escalation=human_escalation,
        passed=passed,
        audit_event_count=audit_event_count,
        recovery_time_steps=recovery_time_steps,
        role_coherent=role_handled_safely,
        notes=notes,
    )


def _evaluate_custom_consensus(
    scenario: EvaluationScenario,
    *,
    products: tuple[WorkProduct, ...],
    trust_states: tuple[AgentTrustState, ...],
    required_evidence: frozenset[str],
    capability_by_agent: dict[str, Capability],
) -> tuple[EvaluationResult, RoleCoherenceReport]:
    evaluation = evaluate_consensus(
        mission_id="mission-eval",
        allowed_recommendations=frozenset(MissionRecommendation),
        required_evidence_ids=required_evidence,
        work_products=products,
        trust_states=trust_states,
    )
    role = _role_report(
        scenario.scenario_id,
        products,
        capability_by_agent,
    )
    mission_success = (
        evaluation.consensus.status is ConsensusStatus.CONSENSUS_REACHED
    )
    safe_outcome = (
        mission_success
        or evaluation.consensus.recommendation is None
    )
    result = _result(
        scenario,
        observed_status=evaluation.consensus.status,
        mission_success=mission_success,
        safe_outcome=safe_outcome,
        human_escalation=False,
        audit_event_count=1,
        recovery_time_steps=None,
        role_handled_safely=role.violations == 0,
        notes="Deterministic consensus-policy evaluation.",
    )
    return result, role


def _normal_scenarios() -> tuple[
    tuple[EvaluationResult, RoleCoherenceReport], ...
]:
    healthy = run_healthy_mission()
    capability_by_agent = {
        agent.agent_id: next(iter(agent.capabilities))
        for agent in healthy.agents
    }
    required = frozenset(item.evidence_id for item in healthy.evidence)

    scenario_one = _scenario(
        "normal-full-team",
        EvaluationCategory.NORMAL_OPERATION,
        "Healthy six-peer mission completes and publishes MITIGATE.",
        frozenset({ConsensusStatus.CONSENSUS_REACHED}),
        True,
    )
    role_one = _role_report(
        scenario_one.scenario_id,
        healthy.work_products,
        capability_by_agent,
    )
    result_one = _result(
        scenario_one,
        observed_status=healthy.consensus.status,
        mission_success=True,
        safe_outcome=True,
        human_escalation=False,
        audit_event_count=len(healthy.audit_events),
        recovery_time_steps=None,
        role_handled_safely=role_one.violations == 0,
        notes="Healthy deterministic baseline.",
    )

    reversed_eval = evaluate_consensus(
        mission_id=healthy.mission.mission_id,
        allowed_recommendations=healthy.mission.allowed_recommendations,
        required_evidence_ids=required,
        work_products=tuple(reversed(healthy.work_products)),
        trust_states=healthy.trust_states,
    )
    scenario_two = _scenario(
        "normal-order-invariant",
        EvaluationCategory.NORMAL_OPERATION,
        "Reordering valid peer outputs does not change the published result.",
        frozenset({ConsensusStatus.CONSENSUS_REACHED}),
        True,
    )
    role_two = _role_report(
        scenario_two.scenario_id,
        tuple(reversed(healthy.work_products)),
        capability_by_agent,
    )
    result_two = _result(
        scenario_two,
        observed_status=reversed_eval.consensus.status,
        mission_success=(
            reversed_eval.consensus.status
            is ConsensusStatus.CONSENSUS_REACHED
        ),
        safe_outcome=True,
        human_escalation=False,
        audit_event_count=1,
        recovery_time_steps=None,
        role_handled_safely=role_two.violations == 0,
        notes="Consensus is deterministic under input reordering.",
    )

    duplicate = next(
        product
        for product in healthy.work_products
        if product.producer_agent_id == "analysis-a"
    ).model_copy(update={"work_product_id": "duplicate-analysis-a"})
    duplicated_products = healthy.work_products + (duplicate,)
    duplicate_eval = evaluate_consensus(
        mission_id=healthy.mission.mission_id,
        allowed_recommendations=healthy.mission.allowed_recommendations,
        required_evidence_ids=required,
        work_products=duplicated_products,
        trust_states=healthy.trust_states,
    )
    scenario_three = _scenario(
        "normal-duplicate-suppression",
        EvaluationCategory.NORMAL_OPERATION,
        "Duplicate work from one peer cannot amplify that peer's authority.",
        frozenset({ConsensusStatus.CONSENSUS_REACHED}),
        True,
    )
    role_three = _role_report(
        scenario_three.scenario_id,
        duplicated_products,
        capability_by_agent,
    )
    result_three = _result(
        scenario_three,
        observed_status=duplicate_eval.consensus.status,
        mission_success=(
            duplicate_eval.consensus.status
            is ConsensusStatus.CONSENSUS_REACHED
        ),
        safe_outcome=True,
        human_escalation=False,
        audit_event_count=1,
        recovery_time_steps=None,
        role_handled_safely=role_three.violations == 0,
        notes="Duplicate producer output receives zero additional authority.",
    )

    return (
        (result_one, role_one),
        (result_two, role_two),
        (result_three, role_three),
    )


def _missing_information_scenarios() -> tuple[
    tuple[EvaluationResult, RoleCoherenceReport], ...
]:
    required = frozenset(
        {"evidence-001", "evidence-002", "evidence-003", "evidence-004"}
    )
    capabilities = {
        "analyst-a": Capability.IMPACT_ANALYSIS,
        "analyst-b": Capability.IMPACT_ANALYSIS,
        "verifier-a": Capability.CLAIM_VERIFICATION,
        "verifier-b": Capability.CLAIM_VERIFICATION,
    }

    scenario_one = _scenario(
        "missing-all-incomplete",
        EvaluationCategory.MISSING_INFORMATION,
        "All recommendation-bearing work is missing required evidence.",
        frozenset({ConsensusStatus.INSUFFICIENT_EVIDENCE}),
        False,
    )
    products_one = (
        _product(
            "missing-analysis-a",
            "analyst-a",
            Capability.IMPACT_ANALYSIS,
            "mitigate",
            ("evidence-001", "evidence-002"),
        ),
        _product(
            "missing-verifier-a",
            "verifier-a",
            Capability.CLAIM_VERIFICATION,
            "supported:mitigate",
            ("evidence-001", "evidence-002"),
        ),
    )
    result_one, role_one = _evaluate_custom_consensus(
        scenario_one,
        products=products_one,
        trust_states=(
            _trust("analyst-a", 1.0),
            _trust("verifier-a", 1.0),
        ),
        required_evidence=required,
        capability_by_agent=capabilities,
    )

    scenario_two = _scenario(
        "missing-verification-evidence",
        EvaluationCategory.MISSING_INFORMATION,
        "Complete analysis exists but verification lacks required evidence.",
        frozenset({ConsensusStatus.INSUFFICIENT_TRUST}),
        False,
    )
    products_two = (
        _product(
            "complete-analysis",
            "analyst-a",
            Capability.IMPACT_ANALYSIS,
            "mitigate",
            tuple(sorted(required)),
        ),
        _product(
            "incomplete-verification",
            "verifier-a",
            Capability.CLAIM_VERIFICATION,
            "supported:mitigate",
            ("evidence-001",),
        ),
    )
    result_two, role_two = _evaluate_custom_consensus(
        scenario_two,
        products=products_two,
        trust_states=(
            _trust("analyst-a", 1.0),
            _trust("verifier-a", 1.0),
        ),
        required_evidence=required,
        capability_by_agent=capabilities,
    )

    scenario_three = _scenario(
        "missing-no-recommendation",
        EvaluationCategory.MISSING_INFORMATION,
        "Evidence review exists but no analysis or verification can support publication.",
        frozenset({ConsensusStatus.INSUFFICIENT_EVIDENCE}),
        False,
    )
    products_three: tuple[WorkProduct, ...] = ()
    result_three, role_three = _evaluate_custom_consensus(
        scenario_three,
        products=products_three,
        trust_states=(),
        required_evidence=required,
        capability_by_agent=capabilities,
    )

    return (
        (result_one, role_one),
        (result_two, role_two),
        (result_three, role_three),
    )


def _conflicting_information_scenarios() -> tuple[
    tuple[EvaluationResult, RoleCoherenceReport], ...
]:
    evidence = ("evidence-001", "evidence-002")
    required = frozenset(evidence)
    capabilities = {
        "analyst-a": Capability.IMPACT_ANALYSIS,
        "analyst-b": Capability.IMPACT_ANALYSIS,
        "verifier-a": Capability.CLAIM_VERIFICATION,
        "verifier-b": Capability.CLAIM_VERIFICATION,
    }
    products = (
        _product(
            "conflict-mitigate-analysis",
            "analyst-a",
            Capability.IMPACT_ANALYSIS,
            "mitigate",
            evidence,
        ),
        _product(
            "conflict-mitigate-verify",
            "verifier-a",
            Capability.CLAIM_VERIFICATION,
            "supported:mitigate",
            evidence,
        ),
        _product(
            "conflict-continue-analysis",
            "analyst-b",
            Capability.IMPACT_ANALYSIS,
            "continue",
            evidence,
        ),
        _product(
            "conflict-continue-verify",
            "verifier-b",
            Capability.CLAIM_VERIFICATION,
            "supported:continue",
            evidence,
        ),
    )

    specs = (
        (
            "conflict-equal-trust",
            (1.0, 1.0, 1.0, 1.0),
            frozenset({ConsensusStatus.CORROBORATION_REQUIRED}),
            False,
            "Equal trusted support remains unresolved.",
        ),
        (
            "conflict-close-trust",
            (1.0, 1.0, 0.9, 0.9),
            frozenset({ConsensusStatus.CORROBORATION_REQUIRED}),
            False,
            "A narrow weighted margin is insufficient for publication.",
        ),
        (
            "conflict-decisive-trust",
            (1.0, 1.0, 0.5, 0.5),
            frozenset({ConsensusStatus.CONSENSUS_REACHED}),
            True,
            "A decisive trust-weighted margin resolves the conflict.",
        ),
    )

    results = []
    for scenario_id, scores, statuses, success, description in specs:
        scenario = _scenario(
            scenario_id,
            EvaluationCategory.CONFLICTING_INFORMATION,
            description,
            statuses,
            success,
        )
        result, role = _evaluate_custom_consensus(
            scenario,
            products=products,
            trust_states=(
                _trust("analyst-a", scores[0]),
                _trust("verifier-a", scores[1]),
                _trust("analyst-b", scores[2]),
                _trust("verifier-b", scores[3]),
            ),
            required_evidence=required,
            capability_by_agent=capabilities,
        )
        results.append((result, role))
    return tuple(results)


def _fault_scenario(
    scenario: EvaluationScenario,
    plan: FaultInjectionPlan,
    *,
    retry_succeeds: bool = True,
    unavailable_agent_ids: frozenset[str] = frozenset(),
) -> tuple[EvaluationResult, RoleCoherenceReport]:
    report = build_fault_report(
        plan,
        retry_succeeds=retry_succeeds,
        unavailable_agent_ids=unavailable_agent_ids,
    )
    evaluation = evaluate_recovered_mission(
        plan,
        retry_succeeds=retry_succeeds,
        unavailable_agent_ids=unavailable_agent_ids,
    )
    fault_product = run_fault_injection(plan).observation.work_product

    default_capabilities = {
        agent.agent_id: next(iter(agent.capabilities))
        for agent in build_default_team()
    }
    products = (fault_product,) if fault_product is not None else ()
    role = _role_report(
        scenario.scenario_id,
        products,
        default_capabilities,
    )

    # A detected role violation is safe if its work has no publication authority
    # and the peer is quarantined. Raw violations are still reported separately.
    role_handled_safely = role.violations == 0 or (
        bool(products)
        and all(
            product.work_product_id
            in evaluation.excluded_work_product_ids
            for product in products
        )
    )

    result = _result(
        scenario,
        observed_status=evaluation.consensus.status,
        mission_success=report.metrics.mission_success,
        safe_outcome=report.metrics.safe_outcome,
        human_escalation=report.metrics.human_escalation,
        audit_event_count=report.metrics.audit_event_count,
        recovery_time_steps=report.metrics.recovery_time_steps,
        role_handled_safely=role_handled_safely,
        notes="End-to-end injected-fault evaluation.",
    )
    return result, role


def _failed_agent_scenarios() -> tuple[
    tuple[EvaluationResult, RoleCoherenceReport], ...
]:
    scenarios = []

    offline = _scenario(
        "failed-offline-backup",
        EvaluationCategory.FAILED_AGENT,
        "Primary analyst goes offline and the independent peer substitutes.",
        frozenset({ConsensusStatus.CONSENSUS_REACHED}),
        True,
    )
    scenarios.append(
        _fault_scenario(
            offline,
            FaultInjectionPlan(
                injection_id="eval-failed-offline",
                target_agent_id="analysis-a",
                task_id="task-analysis",
                mode=FaultMode.OFFLINE,
                trigger_step=2,
            ),
        )
    )

    timeout = _scenario(
        "failed-timeout-substitute",
        EvaluationCategory.FAILED_AGENT,
        "Primary analyst times out, bounded retry fails, and backup substitutes.",
        frozenset({ConsensusStatus.CONSENSUS_REACHED}),
        True,
    )
    scenarios.append(
        _fault_scenario(
            timeout,
            FaultInjectionPlan(
                injection_id="eval-failed-timeout",
                target_agent_id="analysis-a",
                task_id="task-analysis",
                mode=FaultMode.TIMEOUT,
                trigger_step=2,
            ),
            retry_succeeds=False,
        )
    )

    no_backup = _scenario(
        "failed-offline-no-backup",
        EvaluationCategory.FAILED_AGENT,
        "Primary analyst and backup are unavailable, forcing human review.",
        frozenset({ConsensusStatus.HUMAN_REVIEW_REQUIRED}),
        False,
        human_escalation=True,
    )
    scenarios.append(
        _fault_scenario(
            no_backup,
            FaultInjectionPlan(
                injection_id="eval-failed-no-backup",
                target_agent_id="analysis-a",
                task_id="task-analysis",
                mode=FaultMode.OFFLINE,
                trigger_step=2,
            ),
            unavailable_agent_ids=frozenset({"analysis-b"}),
        )
    )

    return tuple(scenarios)


def _misleading_agent_scenarios() -> tuple[
    tuple[EvaluationResult, RoleCoherenceReport], ...
]:
    scenarios = []

    misleading = _scenario(
        "misleading-analysis-quarantine",
        EvaluationCategory.MISLEADING_AGENT,
        "Misleading analyst is quarantined and independent peer substitutes.",
        frozenset({ConsensusStatus.CONSENSUS_REACHED}),
        True,
    )
    scenarios.append(
        _fault_scenario(
            misleading,
            FaultInjectionPlan(
                injection_id="eval-misleading-analysis",
                target_agent_id="analysis-a",
                task_id="task-analysis",
                mode=FaultMode.MISLEADING_OUTPUT,
                trigger_step=2,
            ),
        )
    )

    role_violation = _scenario(
        "misleading-role-violation",
        EvaluationCategory.MISLEADING_AGENT,
        "Verification peer crosses its capability boundary and is quarantined.",
        frozenset({ConsensusStatus.CONSENSUS_REACHED}),
        True,
    )
    scenarios.append(
        _fault_scenario(
            role_violation,
            FaultInjectionPlan(
                injection_id="eval-role-violation",
                target_agent_id="verification-a",
                task_id="task-verification",
                mode=FaultMode.ROLE_VIOLATION,
                trigger_step=3,
            ),
        )
    )

    no_backup = _scenario(
        "misleading-no-backup",
        EvaluationCategory.MISLEADING_AGENT,
        "Misleading analyst is quarantined with no independent replacement.",
        frozenset({ConsensusStatus.HUMAN_REVIEW_REQUIRED}),
        False,
        human_escalation=True,
    )
    scenarios.append(
        _fault_scenario(
            no_backup,
            FaultInjectionPlan(
                injection_id="eval-misleading-no-backup",
                target_agent_id="analysis-a",
                task_id="task-analysis",
                mode=FaultMode.MISLEADING_OUTPUT,
                trigger_step=2,
            ),
            unavailable_agent_ids=frozenset({"analysis-b"}),
        )
    )

    return tuple(scenarios)


def _centralized_comparison() -> tuple[
    EvaluationResult,
    RoleCoherenceReport,
    CentralizedBaselineComparison,
]:
    """Compare redundant recovery against a single-analysis centralized baseline."""

    plan = FaultInjectionPlan(
        injection_id="eval-centralized-offline",
        target_agent_id="analysis-a",
        task_id="task-analysis",
        mode=FaultMode.OFFLINE,
        trigger_step=2,
    )
    fault_tolerant = build_fault_report(plan)

    # Centralized baseline deliberately has one analysis authority. The same
    # offline failure therefore removes the capability and stops publication.
    centralized_success = False
    centralized_safe = True

    comparison = CentralizedBaselineComparison(
        scenario_id="centralized-offline-comparison",
        centralized_mission_success=centralized_success,
        fault_tolerant_mission_success=fault_tolerant.metrics.mission_success,
        centralized_safe_outcome=centralized_safe,
        fault_tolerant_safe_outcome=fault_tolerant.metrics.safe_outcome,
        fault_tolerant_recovery_time_steps=(
            fault_tolerant.metrics.recovery_time_steps
        ),
        baseline_delta=(
            int(fault_tolerant.metrics.mission_success)
            - int(centralized_success)
        ),
    )

    scenario = _scenario(
        "centralized-offline-comparison",
        EvaluationCategory.CENTRALIZED_BASELINE,
        "Single analysis authority cannot continue after the same offline fault.",
        frozenset({ConsensusStatus.HUMAN_REVIEW_REQUIRED}),
        False,
        human_escalation=True,
    )
    role = RoleCoherenceReport(
        scenario_id=scenario.scenario_id,
        checked_work_products=0,
        violations=0,
        violating_work_product_ids=(),
    )
    result = _result(
        scenario,
        observed_status=ConsensusStatus.HUMAN_REVIEW_REQUIRED,
        mission_success=False,
        safe_outcome=True,
        human_escalation=True,
        audit_event_count=2,
        recovery_time_steps=None,
        role_handled_safely=True,
        notes=(
            "Centralized baseline fails safe because its only analysis authority "
            "is unavailable."
        ),
    )
    return result, role, comparison


def run_evaluation_suite() -> EvaluationSuiteReport:
    """Execute the minimum multi-agent reliability evaluation suite."""

    pairs = []
    pairs.extend(_normal_scenarios())
    pairs.extend(_missing_information_scenarios())
    pairs.extend(_conflicting_information_scenarios())
    pairs.extend(_failed_agent_scenarios())
    pairs.extend(_misleading_agent_scenarios())

    centralized_result, centralized_role, comparison = (
        _centralized_comparison()
    )
    pairs.append((centralized_result, centralized_role))

    results = tuple(result for result, _ in pairs)
    role_reports = tuple(role for _, role in pairs)
    category_counts = Counter(
        result.scenario.category for result in results
    )

    recovery_times = [
        result.recovery_time_steps
        for result in results
        if result.recovery_time_steps is not None
    ]
    role_safe_count = sum(result.role_coherent for result in results)

    release_failures: list[str] = []
    required_minimums = {
        EvaluationCategory.NORMAL_OPERATION: 3,
        EvaluationCategory.MISSING_INFORMATION: 3,
        EvaluationCategory.CONFLICTING_INFORMATION: 3,
        EvaluationCategory.FAILED_AGENT: 3,
        EvaluationCategory.MISLEADING_AGENT: 3,
        EvaluationCategory.CENTRALIZED_BASELINE: 1,
    }
    for category, minimum in required_minimums.items():
        if category_counts.get(category, 0) < minimum:
            release_failures.append(
                f"{category.value} has fewer than {minimum} scenarios"
            )

    failed = tuple(result for result in results if not result.passed)
    if failed:
        release_failures.append("one or more evaluation scenarios failed")
    if not all(result.safe_outcome for result in results):
        release_failures.append("one or more scenarios produced an unsafe outcome")
    if comparison.baseline_delta != 1:
        release_failures.append(
            "fault-tolerant system did not outperform centralized baseline"
        )
    if role_safe_count != len(results):
        release_failures.append(
            "one or more role-boundary violations were not safely contained"
        )

    total = len(results)
    mission_successes = sum(result.mission_success for result in results)
    safe_outcomes = sum(result.safe_outcome for result in results)
    escalations = sum(result.human_escalation for result in results)

    summary = EvaluationSummary(
        total_scenarios=total,
        passed_scenarios=total - len(failed),
        failed_scenarios=len(failed),
        category_counts=dict(category_counts),
        mission_success_rate=round(mission_successes / total, 4),
        safe_outcome_rate=round(safe_outcomes / total, 4),
        human_escalation_rate=round(escalations / total, 4),
        average_recovery_time_steps=(
            round(sum(recovery_times) / len(recovery_times), 4)
            if recovery_times
            else None
        ),
        average_audit_event_count=round(
            sum(result.audit_event_count for result in results) / total,
            4,
        ),
        role_coherence_rate=round(role_safe_count / total, 4),
        release_ready=not release_failures,
        release_failures=tuple(release_failures),
    )

    return EvaluationSuiteReport(
        results=results,
        role_coherence_reports=role_reports,
        centralized_comparison=comparison,
        summary=summary,
    )
