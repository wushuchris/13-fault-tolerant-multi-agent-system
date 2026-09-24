"""Presentation helpers for the Agent 13 Hugging Face demo.

This module translates deterministic domain objects into UI-friendly rows and
summaries. It does not make trust, recovery, or publication decisions.
"""

from __future__ import annotations

import os

from .baseline import build_default_team, run_healthy_mission
from .consensus import evaluate_recovered_mission
from .evaluation import run_evaluation_suite
from .llm_adapter import (
    HuggingFaceOpenAIClient,
    LLMAdapterError,
    invoke_specialist,
    model_id_from_env,
)
from .metrics import build_fault_report, build_healthy_report
from .recovery import recover_single_fault
from .reliability import initial_health_state, initial_trust_state
from .schemas import (
    FaultInjectionPlan,
    FaultMode,
    HealthStatus,
    SpecialistTaskPacket,
)


SCENARIOS = {
    "Healthy Mission": {
        "description": "All six peers operate normally and publish a verified recommendation.",
        "mode": None,
    },
    "Offline Agent": {
        "description": "Analysis Agent A disappears and the independent analysis peer substitutes.",
        "mode": FaultMode.OFFLINE,
    },
    "Timeout": {
        "description": "Analysis Agent A times out and succeeds on one bounded retry.",
        "mode": FaultMode.TIMEOUT,
    },
    "Malformed Output": {
        "description": "Analysis Agent A returns invalid structure and succeeds on one bounded retry.",
        "mode": FaultMode.MALFORMED_OUTPUT,
    },
    "Unsupported Output": {
        "description": "Analysis Agent A returns under-supported work and triggers independent corroboration.",
        "mode": FaultMode.UNSUPPORTED_OUTPUT,
    },
    "Contradictory Agent": {
        "description": "Analysis Agent A disagrees with its peer and triggers independent corroboration.",
        "mode": FaultMode.CONTRADICTORY_OUTPUT,
    },
    "Misleading Agent": {
        "description": "Analysis Agent A contradicts known evidence, is quarantined, and is replaced.",
        "mode": FaultMode.MISLEADING_OUTPUT,
    },
    "Role Violation": {
        "description": "Verification Agent A crosses its capability boundary, is quarantined, and is replaced.",
        "mode": FaultMode.ROLE_VIOLATION,
    },
    "No Backup Available": {
        "description": "Analysis Agent A is offline and Analysis Agent B is also unavailable, forcing human review.",
        "mode": FaultMode.OFFLINE,
        "no_backup": True,
    },
}


def scenario_choices() -> list[str]:
    return list(SCENARIOS)


def _fault_plan(name: str) -> tuple[FaultInjectionPlan, frozenset[str]]:
    config = SCENARIOS[name]
    mode = config["mode"]
    if mode is None:
        raise ValueError("healthy scenario does not use a fault plan")

    if name == "Role Violation":
        return (
            FaultInjectionPlan(
                injection_id="ui-role-violation",
                target_agent_id="verification-a",
                task_id="task-verification",
                mode=mode,
                trigger_step=3,
            ),
            frozenset(),
        )

    unavailable = (
        frozenset({"analysis-b"})
        if config.get("no_backup")
        else frozenset()
    )
    return (
        FaultInjectionPlan(
            injection_id=f"ui-{mode.value}",
            target_agent_id="analysis-a",
            task_id="task-analysis",
            mode=mode,
            trigger_step=2,
        ),
        unavailable,
    )


def _agent_rows(
    *,
    target_agent_id: str | None = None,
    target_health=None,
    target_trust=None,
    quarantined: frozenset[str] = frozenset(),
    unavailable_agent_ids: frozenset[str] = frozenset(),
) -> list[list[object]]:
    rows: list[list[object]] = []
    for agent in build_default_team():
        health = initial_health_state(agent.agent_id)
        trust = initial_trust_state(agent.agent_id)

        if agent.agent_id in unavailable_agent_ids:
            health = health.model_copy(
                update={
                    "status": HealthStatus.UNAVAILABLE,
                    "consecutive_failures": 1,
                }
            )
        if agent.agent_id == target_agent_id:
            if target_health is not None:
                health = target_health
            if target_trust is not None:
                trust = target_trust

        rows.append(
            [
                agent.display_name,
                agent.role.value,
                health.status.value,
                trust.tier.value,
                round(trust.score, 2),
                "yes" if agent.agent_id in quarantined else "no",
            ]
        )
    return rows


def _metrics_rows(metrics) -> list[list[object]]:
    return [
        ["Mission success", "yes" if metrics.mission_success else "no"],
        ["Safe outcome", "yes" if metrics.safe_outcome else "no"],
        ["Recovery success", "yes" if metrics.recovery_success else "no"],
        ["Human escalation", "yes" if metrics.human_escalation else "no"],
        ["Detection steps", metrics.fault_detection_steps if metrics.fault_detection_steps is not None else "n/a"],
        ["Recovery time", metrics.recovery_time_steps if metrics.recovery_time_steps is not None else "n/a"],
        ["Recovery overhead", metrics.recovery_overhead_steps],
        ["Retries", metrics.retry_count],
        ["Substitutions", metrics.substitution_count],
        ["Corroborations", metrics.corroboration_count],
        ["Quarantines", metrics.quarantine_count],
        ["Escalations", metrics.escalation_count],
        ["Audit events", metrics.audit_event_count],
        ["Final trust score", round(metrics.final_trust_score, 2)],
        ["Winning support", round(metrics.winning_support_score, 2)],
        ["Consensus margin", round(metrics.consensus_support_margin, 2)],
    ]


def _audit_rows(events) -> list[list[object]]:
    return [
        [
            event.sequence,
            event.event_type,
            event.actor_id or "system",
            event.detail,
        ]
        for event in events
    ]


def run_demo_scenario(name: str) -> dict[str, object]:
    if name not in SCENARIOS:
        raise ValueError("unknown demo scenario")

    description = SCENARIOS[name]["description"]

    if name == "Healthy Mission":
        healthy = run_healthy_mission()
        report = build_healthy_report()
        recommendation = healthy.consensus.recommendation
        summary = (
            "### Mission completed\n"
            f"**Scenario:** {name}\n\n"
            f"{description}\n\n"
            f"**Outcome:** {healthy.consensus.status.value}\n\n"
            f"**Recommendation:** {recommendation.value if recommendation else 'none'}"
        )
        return {
            "summary": summary,
            "agents": _agent_rows(),
            "recovery": [["none", "No recovery action required.", "system"]],
            "metrics": _metrics_rows(report.metrics),
            "audit": _audit_rows(report.audit_events),
            "consensus": healthy.consensus.model_dump(mode="json"),
        }

    plan, unavailable = _fault_plan(name)
    recovery = recover_single_fault(
        plan,
        unavailable_agent_ids=unavailable,
    )
    evaluation = evaluate_recovered_mission(
        plan,
        unavailable_agent_ids=unavailable,
    )
    report = build_fault_report(
        plan,
        unavailable_agent_ids=unavailable,
    )

    recommendation = evaluation.consensus.recommendation
    outcome_heading = (
        "Mission completed"
        if report.metrics.mission_success
        else "Human review required"
    )
    summary = (
        f"### {outcome_heading}\n"
        f"**Scenario:** {name}\n\n"
        f"{description}\n\n"
        f"**Consensus:** {evaluation.consensus.status.value}\n\n"
        f"**Recommendation:** {recommendation.value if recommendation else 'none'}\n\n"
        f"**Safe outcome:** {'yes' if report.metrics.safe_outcome else 'no'}"
    )

    recovery_rows = [
        [
            event.action.value,
            event.detail,
            ", ".join(event.affected_agent_ids),
        ]
        for event in recovery.recovery_events
    ]

    return {
        "summary": summary,
        "agents": _agent_rows(
            target_agent_id=plan.target_agent_id,
            target_health=recovery.health_after_recovery,
            target_trust=recovery.trust_after_recovery,
            quarantined=recovery.quarantined_agent_ids,
            unavailable_agent_ids=unavailable,
        ),
        "recovery": recovery_rows,
        "metrics": _metrics_rows(report.metrics),
        "audit": _audit_rows(report.audit_events),
        "consensus": evaluation.consensus.model_dump(mode="json"),
    }


def evaluation_dashboard() -> dict[str, object]:
    report = run_evaluation_suite()
    summary = report.summary

    headline = (
        "### Automated evaluation gate: PASS"
        if summary.release_ready
        else "### Automated evaluation gate: FAIL"
    )
    overview = (
        f"{headline}\n\n"
        f"**Scenarios:** {summary.passed_scenarios}/{summary.total_scenarios} passed  \n"
        f"**Safe outcome rate:** {summary.safe_outcome_rate:.0%}  \n"
        f"**Role containment rate:** {summary.role_coherence_rate:.0%}  \n"
        f"**Average recovery time:** {summary.average_recovery_time_steps} simulation steps  \n"
        f"**Centralized baseline delta:** {report.centralized_comparison.baseline_delta:+d}"
    )

    rows = [
        [
            result.scenario.scenario_id,
            result.scenario.category.value,
            "pass" if result.passed else "fail",
            result.observed_consensus_status.value,
            "yes" if result.mission_success else "no",
            "yes" if result.human_escalation else "no",
            result.recovery_time_steps if result.recovery_time_steps is not None else "n/a",
        ]
        for result in report.results
    ]

    return {
        "summary": overview,
        "results": rows,
        "comparison": report.centralized_comparison.model_dump(mode="json"),
    }


def live_inference_status() -> str:
    token_present = bool(os.getenv("HF_TOKEN", "").strip())
    model_present = bool(os.getenv("MODEL_ID", "").strip())
    if token_present and model_present:
        return (
            "### Live specialist inference: configured\n"
            "Runtime credentials were found in the environment. Secret values are never displayed."
        )
    return (
        "### Live specialist inference: not configured\n"
        "Add HF_TOKEN as a Hugging Face Space secret and MODEL_ID as a Space variable before using this tab."
    )


def run_live_analysis() -> tuple[str, dict[str, object]]:
    if not os.getenv("HF_TOKEN", "").strip() or not os.getenv("MODEL_ID", "").strip():
        return (
            "Live inference is not configured. Add HF_TOKEN in Hugging Face Space Secrets and MODEL_ID in Space Variables.",
            {},
        )

    healthy = run_healthy_mission()
    task = next(task for task in healthy.tasks if task.task_id == "task-analysis")
    assignment = next(
        assignment
        for assignment in healthy.assignments
        if assignment.agent_id == "analysis-a"
    )
    upstream = tuple(
        product
        for product in healthy.work_products
        if product.task_id == "task-evidence"
    )
    packet = SpecialistTaskPacket(
        packet_id="live-analysis-packet",
        task=task,
        assignment=assignment,
        evidence=healthy.evidence,
        upstream_work_products=upstream,
        allowed_recommendations=healthy.mission.allowed_recommendations,
    )

    try:
        product = invoke_specialist(
            packet=packet,
            client=HuggingFaceOpenAIClient.from_env(),
            model_id=model_id_from_env(),
            work_product_id="live-analysis-product",
            created_at_step=2,
        )
    except LLMAdapterError as exc:
        return (f"Live inference failed closed: {exc}", {})

    return (
        "Live specialist output passed the bounded application checks.",
        product.model_dump(mode="json"),
    )
