"""Presentation helpers for the Agent 13 Hugging Face demo.

This module translates deterministic domain objects into business-readable
stories plus optional engineering detail. It never changes trust, recovery,
consensus, or publication decisions.
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


BUSINESS_CASE = (
    "A fictional fulfillment center receives a sorter-motor temperature alert. "
    "Throughput falls 18%, but safety interlocks remain normal. A backup "
    "conveyor can carry reduced load, and a replacement motor is already onsite "
    "for a controlled 30-minute replacement window."
)

SCENARIOS = {
    "Healthy Mission": {
        "description": "Control case: every AI teammate performs its assigned job normally.",
        "failure_story": (
            "Nothing fails. This is the control case that shows what normal "
            "operation looks like."
        ),
        "response_story": (
            "Two evidence reviewers confirm the facts, two analysts assess the "
            "operational impact, and two independent verifiers check the recommendation."
        ),
        "why_it_matters": (
            "This establishes the normal result before we deliberately break "
            "individual members of the AI team."
        ),
        "mode": None,
    },
    "Offline Agent": {
        "description": "One analyst disappears before completing its assigned work.",
        "failure_story": (
            "Analysis Agent A becomes unavailable. It does not return an answer."
        ),
        "response_story": (
            "The system marks that analyst unavailable and uses the independently "
            "assigned Analysis Agent B instead. Going offline does not automatically "
            "make an agent untrustworthy."
        ),
        "why_it_matters": (
            "A single unavailable AI worker does not stop the organization when "
            "redundant capability exists."
        ),
        "mode": FaultMode.OFFLINE,
    },
    "Timeout": {
        "description": "One analyst does not respond within its allowed time window.",
        "failure_story": (
            "Analysis Agent A takes too long to respond and crosses the timeout boundary."
        ),
        "response_story": (
            "The system allows one bounded retry. The retry succeeds, health returns "
            "to normal, but the timeout remains in the audit history and slightly lowers trust."
        ),
        "why_it_matters": (
            "Recovery does not erase history. The system can keep moving while still "
            "remembering that reliability degraded."
        ),
        "mode": FaultMode.TIMEOUT,
    },
    "Malformed Output": {
        "description": "One analyst returns an answer that violates the required data contract.",
        "failure_story": (
            "Analysis Agent A returns a response that cannot pass the required structured-output checks."
        ),
        "response_story": (
            "The invalid artifact is rejected before it can influence the mission. "
            "One bounded retry succeeds, while the trust penalty remains."
        ),
        "why_it_matters": (
            "A fluent-looking AI answer is not accepted unless it satisfies the application's contract."
        ),
        "mode": FaultMode.MALFORMED_OUTPUT,
    },
    "Unsupported Output": {
        "description": "One analyst gives an answer without supporting it with the full evidence packet.",
        "failure_story": (
            "Analysis Agent A reaches a conclusion while omitting required evidence."
        ),
        "response_story": (
            "The system reduces the work's authority and asks the independent analyst "
            "to corroborate the conclusion using the complete evidence set."
        ),
        "why_it_matters": (
            "The system rewards evidence, not confidence or repetition."
        ),
        "mode": FaultMode.UNSUPPORTED_OUTPUT,
    },
    "Contradictory Agent": {
        "description": "Two AI analysts reach conflicting conclusions.",
        "failure_story": (
            "Analysis Agent A disagrees with the expected evidence-backed conclusion."
        ),
        "response_story": (
            "The system does not simply count votes. It requests independent corroboration "
            "and later uses evidence provenance, verification, and trust-weighted support."
        ),
        "why_it_matters": (
            "Multiple AI agents can disagree without forcing the application to guess."
        ),
        "mode": FaultMode.CONTRADICTORY_OUTPUT,
    },
    "Misleading Agent": {
        "description": "One analyst returns a plausible but demonstrably false operational story.",
        "failure_story": (
            "Analysis Agent A falsely claims that safety interlocks failed and that "
            "no backup conveyor is available, contradicting the approved evidence."
        ),
        "response_story": (
            "Trust drops sharply, the analyst is quarantined, its work receives zero "
            "publication authority, and Analysis Agent B substitutes."
        ),
        "why_it_matters": (
            "A bad AI answer can remain visible for audit while being prevented from "
            "influencing the final decision."
        ),
        "mode": FaultMode.MISLEADING_OUTPUT,
    },
    "Role Violation": {
        "description": "A verifier attempts to operate outside its assigned authority.",
        "failure_story": (
            "Verification Agent A crosses its role boundary instead of staying within verification."
        ),
        "response_story": (
            "The system detects the authority violation, quarantines the peer, and "
            "uses the independent verifier instead."
        ),
        "why_it_matters": (
            "Even a technically valid AI response cannot gain authority outside the role the application assigned."
        ),
        "mode": FaultMode.ROLE_VIOLATION,
    },
    "No Backup Available": {
        "description": "The primary analyst fails and the independent backup is also unavailable.",
        "failure_story": (
            "Analysis Agent A is offline, and Analysis Agent B is unavailable too. "
            "The organization no longer has trustworthy independent analysis capability."
        ),
        "response_story": (
            "The system refuses to invent consensus or publish a recommendation. "
            "It stops automation and sends the mission to a human reviewer."
        ),
        "why_it_matters": (
            "Fault tolerance includes knowing when not to automate. Safe escalation is "
            "better than manufacturing an answer."
        ),
        "mode": FaultMode.OFFLINE,
        "no_backup": True,
    },
}


ROLE_LABELS = {
    "evidence": "Evidence reviewer — confirms the approved facts",
    "analysis": "Impact analyst — recommends what the operation should do",
    "verification": "Independent verifier — checks the recommendation before publication",
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


def _team_story_rows(
    *,
    name: str,
    target_agent_id: str | None = None,
    target_health=None,
    target_trust=None,
    quarantined: frozenset[str] = frozenset(),
    unavailable_agent_ids: frozenset[str] = frozenset(),
) -> list[list[str]]:
    technical_rows = _agent_rows(
        target_agent_id=target_agent_id,
        target_health=target_health,
        target_trust=target_trust,
        quarantined=quarantined,
        unavailable_agent_ids=unavailable_agent_ids,
    )
    agents = build_default_team()
    result: list[list[str]] = []

    for agent, technical in zip(agents, technical_rows):
        status = "Normal"
        if agent.agent_id in quarantined:
            status = "Quarantined — cannot influence the final decision"
        elif agent.agent_id in unavailable_agent_ids:
            status = "Unavailable"
        elif target_agent_id == agent.agent_id and name in {"Timeout", "Malformed Output"}:
            status = "Recovered after one bounded retry"
        elif target_agent_id == agent.agent_id and technical[2] == "unavailable":
            status = "Unavailable — backup used"
        elif target_agent_id == agent.agent_id and name in {
            "Unsupported Output",
            "Contradictory Agent",
        }:
            status = "Work challenged — independent corroboration required"

        if name == "Offline Agent" and agent.agent_id == "analysis-b":
            status = "Backup analyst used"
        if name in {"Misleading Agent"} and agent.agent_id == "analysis-b":
            status = "Replacement analyst used"
        if name == "Role Violation" and agent.agent_id == "verification-b":
            status = "Replacement verifier used"

        result.append(
            [
                agent.display_name,
                ROLE_LABELS[agent.role.value],
                status,
            ]
        )
    return result


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


def _plain_decision(recommendation) -> str:
    if recommendation is None:
        return (
            "No automated decision. The system stopped and requested human review."
        )
    if recommendation.value == "mitigate":
        return (
            "MITIGATE — keep the operation running in a controlled way using the "
            "backup conveyor while the onsite motor is replaced."
        )
    return recommendation.value.upper()


def _story(
    name: str,
    *,
    recommendation,
    mission_success: bool,
    safe_outcome: bool,
) -> str:
    config = SCENARIOS[name]
    headline = (
        "✅ The AI team completed the mission safely"
        if mission_success
        else "🧑‍⚖️ The AI team stopped and asked for human review"
    )
    return (
        f"## {headline}\n\n"
        "### 1. Business situation\n"
        f"{BUSINESS_CASE}\n\n"
        "### 2. What went wrong\n"
        f"{config['failure_story']}\n\n"
        "### 3. How the system responded\n"
        f"{config['response_story']}\n\n"
        "### 4. Final outcome\n"
        f"**{_plain_decision(recommendation)}**\n\n"
        f"Safe outcome: **{'Yes' if safe_outcome else 'No'}**\n\n"
        "### 5. Why this matters\n"
        f"{config['why_it_matters']}"
    )


def _executive_rows(name: str, metrics, recommendation) -> list[list[str]]:
    recovery_action = "No recovery needed"
    if metrics.human_escalation:
        recovery_action = "Stopped automation and escalated to a human"
    elif metrics.quarantine_count:
        recovery_action = "Quarantined unreliable AI and used an independent replacement"
    elif metrics.substitution_count:
        recovery_action = "Used an independent backup AI"
    elif metrics.corroboration_count:
        recovery_action = "Requested independent corroboration"
    elif metrics.retry_count:
        recovery_action = "Allowed one bounded retry"

    return [
        ["Business issue", "Sorter motor alert; throughput down 18%"],
        ["Scenario", SCENARIOS[name]["description"]],
        ["System response", recovery_action],
        ["Final decision", _plain_decision(recommendation)],
        ["Human required?", "Yes" if metrics.human_escalation else "No"],
    ]


def run_demo_scenario(name: str) -> dict[str, object]:
    if name not in SCENARIOS:
        raise ValueError("unknown demo scenario")

    if name == "Healthy Mission":
        healthy = run_healthy_mission()
        report = build_healthy_report()
        recommendation = healthy.consensus.recommendation
        return {
            "story": _story(
                name,
                recommendation=recommendation,
                mission_success=True,
                safe_outcome=True,
            ),
            "executive": _executive_rows(name, report.metrics, recommendation),
            "team": _team_story_rows(name=name),
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

    recovery_rows = [
        [
            event.action.value,
            event.detail,
            ", ".join(event.affected_agent_ids),
        ]
        for event in recovery.recovery_events
    ]

    return {
        "story": _story(
            name,
            recommendation=recommendation,
            mission_success=report.metrics.mission_success,
            safe_outcome=report.metrics.safe_outcome,
        ),
        "executive": _executive_rows(name, report.metrics, recommendation),
        "team": _team_story_rows(
            name=name,
            target_agent_id=plan.target_agent_id,
            target_health=recovery.health_after_recovery,
            target_trust=recovery.trust_after_recovery,
            quarantined=recovery.quarantined_agent_ids,
            unavailable_agent_ids=unavailable,
        ),
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
        "## ✅ Automated reliability evaluation passed"
        if summary.release_ready
        else "## ❌ Automated reliability evaluation failed"
    )
    overview = (
        f"{headline}\n\n"
        "This is the system's reliability test suite—not a model benchmark. "
        "It deliberately creates missing information, disagreement, failed agents, "
        "misleading agents, and loss of backup capacity.\n\n"
        f"**{summary.passed_scenarios}/{summary.total_scenarios} scenarios behaved as expected.**  \n"
        f"**Safe outcome rate:** {summary.safe_outcome_rate:.0%}  \n"
        f"**Role-boundary containment:** {summary.role_coherence_rate:.0%}  \n"
        f"**Average recovery time:** {summary.average_recovery_time_steps} simulation steps  \n\n"
        "**Centralized comparison:** when the only analyst fails, a single-agent "
        "system must stop. This redundant system can continue when an independent backup exists."
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
            "### Live AI analyst: ready\n"
            "This button calls a real hosted language model for one bounded analyst job. "
            "The application then validates the answer before accepting it."
        )
    return (
        "### Live AI analyst: not configured\n"
        "Add HF_TOKEN as a Hugging Face Space secret and MODEL_ID as a Space variable before using this tab."
    )


def live_product_story(payload: dict[str, object]) -> str:
    if not payload:
        return ""
    evidence = payload.get("evidence_ids", [])
    return (
        "## ✅ The live AI analyst passed the application's checks\n\n"
        f"**What it recommended:** {str(payload.get('conclusion', 'none')).upper()}  \n"
        f"**Evidence used:** {len(evidence)} approved evidence items  \n"
        f"**Confidence:** {payload.get('confidence', 'n/a')}  \n\n"
        "**What the AI was *not* allowed to decide:** its identity, its role, its "
        "trust score, recovery actions, or whether the mission could be published. "
        "Those remain application-controlled."
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
