"""Deterministic audit reporting and reliability metrics for Agent 13."""

from __future__ import annotations

from .baseline import run_healthy_mission
from .consensus import evaluate_recovered_mission
from .faults import run_fault_injection
from .recovery import recover_single_fault
from .schemas import (
    AuditEvent,
    ConsensusStatus,
    FaultInjectionPlan,
    FaultMode,
    MissionStatus,
    RecoveryAction,
    RecoveryStatus,
    ReliabilityMetrics,
    ReliabilityReport,
    TrustSnapshot,
)


def _count_action(recovery_events, action: RecoveryAction) -> int:
    return sum(event.action is action for event in recovery_events)


def build_healthy_report() -> ReliabilityReport:
    """Create metrics for the deterministic no-fault baseline."""

    run = run_healthy_mission()
    trust = run.trust_states[0]

    metrics = ReliabilityMetrics(
        scenario_id="healthy-baseline",
        fault_mode=None,
        mission_success=True,
        safe_outcome=True,
        recovery_attempted=False,
        recovery_success=False,
        human_escalation=False,
        fault_detection_steps=None,
        recovery_time_steps=None,
        publication_time_steps=run.state.current_step,
        recovery_overhead_steps=0,
        retry_count=0,
        substitution_count=0,
        corroboration_count=0,
        quarantine_count=0,
        escalation_count=0,
        audit_event_count=len(run.audit_events),
        initial_trust_score=trust.score,
        post_fault_trust_score=trust.score,
        final_trust_score=trust.score,
        trust_trajectory=(
            TrustSnapshot(
                step=0,
                agent_id=trust.agent_id,
                score=trust.score,
                tier=trust.tier,
                reason="Healthy baseline begins with fully trusted peers.",
            ),
            TrustSnapshot(
                step=run.state.current_step,
                agent_id=trust.agent_id,
                score=trust.score,
                tier=trust.tier,
                reason="No fault occurred, so trust remained unchanged.",
            ),
        ),
        winning_support_score=4.0,
        consensus_support_margin=4.0,
    )
    return ReliabilityReport(
        scenario_id="healthy-baseline",
        audit_events=run.audit_events,
        metrics=metrics,
    )


def _build_fault_audit(
    plan: FaultInjectionPlan,
    *,
    recovery,
    consensus,
) -> tuple[AuditEvent, ...]:
    """Compose one audit sequence spanning fault, assessment, recovery, and publication."""

    fault_run = run_fault_injection(plan)
    observation = fault_run.observation
    assessment = recovery.assessment

    events: list[AuditEvent] = []

    def add(
        event_type: str,
        detail: str,
        *,
        actor_id: str | None = None,
        subject_id: str | None = None,
    ) -> None:
        sequence = len(events)
        events.append(
            AuditEvent(
                audit_event_id=f"report-audit-{sequence:03d}",
                mission_id=fault_run.mission.mission_id,
                sequence=sequence,
                event_type=event_type,
                actor_id=actor_id,
                subject_id=subject_id,
                detail=detail,
            )
        )

    add(
        "scenario_started",
        f"Reliability scenario {plan.injection_id} started.",
    )
    add(
        "fault_injected",
        f"Injected {plan.mode.value} at {plan.task_id}.",
        actor_id=plan.target_agent_id,
        subject_id=plan.task_id,
    )
    add(
        "fault_observed",
        observation.fault_event.detail,
        actor_id=plan.target_agent_id,
        subject_id=observation.fault_event.fault_event_id,
    )
    add(
        "reliability_assessed",
        (
            f"Health became {assessment.health_after.status.value}; trust became "
            f"{assessment.trust_after.tier.value} at score "
            f"{assessment.trust_after.score:.2f}."
        ),
        actor_id=plan.target_agent_id,
        subject_id=assessment.assessment_id,
    )

    for recovery_event in recovery.recovery_events:
        add(
            f"recovery_{recovery_event.action.value}",
            recovery_event.detail,
            actor_id=plan.target_agent_id,
            subject_id=recovery_event.recovery_event_id,
        )

    add(
        "consensus_evaluated",
        (
            f"Consensus status: {consensus.consensus.status.value}; winning "
            f"support score: {consensus.winning_support_score:.2f}."
        ),
        subject_id=consensus.consensus.consensus_id,
    )

    if consensus.consensus.status is ConsensusStatus.CONSENSUS_REACHED:
        add(
            "mission_completed",
            (
                "Trust-aware publication policy approved recommendation "
                f"{consensus.consensus.recommendation.value}."
            ),
            subject_id=consensus.consensus.consensus_id,
        )
    else:
        add(
            "human_review_required",
            "Automated publication stopped and the mission was escalated safely.",
            subject_id=consensus.consensus.consensus_id,
        )

    return tuple(events)


def build_fault_report(
    plan: FaultInjectionPlan,
    *,
    retry_succeeds: bool = True,
    unavailable_agent_ids: frozenset[str] = frozenset(),
) -> ReliabilityReport:
    """Measure one deterministic degraded scenario end to end."""

    healthy = run_healthy_mission()
    fault_run = run_fault_injection(plan)
    recovery = recover_single_fault(
        plan,
        retry_succeeds=retry_succeeds,
        unavailable_agent_ids=unavailable_agent_ids,
    )
    consensus = evaluate_recovered_mission(
        plan,
        retry_succeeds=retry_succeeds,
        unavailable_agent_ids=unavailable_agent_ids,
    )
    audit_events = _build_fault_audit(
        plan,
        recovery=recovery,
        consensus=consensus,
    )

    observed_step = fault_run.observation.fault_event.observed_at_step
    final_recovery_step = recovery.recovery_events[-1].occurred_at_step
    publication_step = consensus.mission_state.current_step
    support_margin = round(
        max(
            0.0,
            consensus.winning_support_score
            - consensus.runner_up_support_score,
        ),
        4,
    )

    mission_success = (
        consensus.consensus.status is ConsensusStatus.CONSENSUS_REACHED
        and consensus.mission_state.status is MissionStatus.COMPLETED
    )
    human_escalation = (
        consensus.consensus.status is ConsensusStatus.HUMAN_REVIEW_REQUIRED
        or any(
            event.action is RecoveryAction.ESCALATE
            for event in recovery.recovery_events
        )
    )

    final_step = max(
        final_recovery_step,
        publication_step,
    )

    metrics = ReliabilityMetrics(
        scenario_id=plan.injection_id,
        fault_mode=plan.mode,
        mission_success=mission_success,
        safe_outcome=(
            mission_success
            or consensus.consensus.status
            is ConsensusStatus.HUMAN_REVIEW_REQUIRED
        ),
        recovery_attempted=True,
        recovery_success=recovery.status is RecoveryStatus.RECOVERED,
        human_escalation=human_escalation,
        fault_detection_steps=max(0, observed_step - plan.trigger_step),
        recovery_time_steps=max(0, final_recovery_step - observed_step),
        publication_time_steps=publication_step,
        recovery_overhead_steps=max(
            0,
            final_step - healthy.state.current_step,
        ),
        retry_count=_count_action(
            recovery.recovery_events,
            RecoveryAction.RETRY,
        ),
        substitution_count=_count_action(
            recovery.recovery_events,
            RecoveryAction.SUBSTITUTE,
        ),
        corroboration_count=_count_action(
            recovery.recovery_events,
            RecoveryAction.CORROBORATE,
        ),
        quarantine_count=_count_action(
            recovery.recovery_events,
            RecoveryAction.QUARANTINE,
        ),
        escalation_count=_count_action(
            recovery.recovery_events,
            RecoveryAction.ESCALATE,
        ),
        audit_event_count=len(audit_events),
        initial_trust_score=recovery.assessment.trust_before.score,
        post_fault_trust_score=recovery.assessment.trust_after.score,
        final_trust_score=recovery.trust_after_recovery.score,
        trust_trajectory=(
            TrustSnapshot(
                step=0,
                agent_id=plan.target_agent_id,
                score=recovery.assessment.trust_before.score,
                tier=recovery.assessment.trust_before.tier,
                reason="Initial trust before the injected fault.",
            ),
            TrustSnapshot(
                step=observed_step,
                agent_id=plan.target_agent_id,
                score=recovery.assessment.trust_after.score,
                tier=recovery.assessment.trust_after.tier,
                reason="Trust immediately after deterministic fault assessment.",
            ),
            TrustSnapshot(
                step=final_recovery_step,
                agent_id=plan.target_agent_id,
                score=recovery.trust_after_recovery.score,
                tier=recovery.trust_after_recovery.tier,
                reason=(
                    "Trust after bounded recovery; successful recovery does not "
                    "erase the observed reliability event."
                ),
            ),
        ),
        winning_support_score=consensus.winning_support_score,
        consensus_support_margin=support_margin,
    )

    return ReliabilityReport(
        scenario_id=plan.injection_id,
        audit_events=audit_events,
        metrics=metrics,
    )


def build_default_reliability_suite() -> tuple[ReliabilityReport, ...]:
    """Return the standard healthy, degraded, and escalation comparison set."""

    reports: list[ReliabilityReport] = [build_healthy_report()]
    for mode in (
        FaultMode.OFFLINE,
        FaultMode.TIMEOUT,
        FaultMode.MALFORMED_OUTPUT,
        FaultMode.UNSUPPORTED_OUTPUT,
        FaultMode.CONTRADICTORY_OUTPUT,
        FaultMode.MISLEADING_OUTPUT,
        FaultMode.ROLE_VIOLATION,
    ):
        plan = FaultInjectionPlan(
            injection_id=f"suite-{mode.value}",
            target_agent_id="analysis-a",
            task_id="task-analysis",
            mode=mode,
            trigger_step=2,
        )
        reports.append(build_fault_report(plan))

    escalation_plan = FaultInjectionPlan(
        injection_id="suite-offline-no-backup",
        target_agent_id="analysis-a",
        task_id="task-analysis",
        mode=FaultMode.OFFLINE,
        trigger_step=2,
    )
    reports.append(
        build_fault_report(
            escalation_plan,
            unavailable_agent_ids=frozenset({"analysis-b"}),
        )
    )

    return tuple(reports)
