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
from .faults import run_fault_injection
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
    "A fictional asset manager is preparing an end-of-day portfolio valuation. "
    "Its primary market-data feed suddenly prices synthetic security NSTR-01 at "
    "$82.00, 18% below the prior validated price of $100.00. An approved "
    "secondary source reports $99.20, independent market indications cluster "
    "near $99 to $100, and no issuer filing or corporate action explains the "
    "18% move."
)

SCENARIOS = {
    "Healthy Team — Nothing Fails": {
        "description": "Control case: every AI teammate performs its assigned job normally.",
        "failure_story": (
            "Nothing fails. This is the control case that shows what normal "
            "operation looks like."
        ),
        "response_story": (
            "The six-agent team completes the normal evidence, analysis, and verification path."
        ),
        "response_steps": (
            "Two evidence reviewers independently confirm the approved pricing facts.",
            "Two analysts independently assess the valuation discrepancy.",
            "Two verifiers independently check the recommendation and evidence lineage.",
            "Only after verification does the application allow a recommendation to publish.",
        ),
        "why_it_matters": (
            "This establishes the normal result before we deliberately break "
            "individual members of the AI team."
        ),
        "mode": None,
    },
    "Agent Offline — Backup Takes Over": {
        "description": "One analyst disappears before completing its assigned work.",
        "failure_story": (
            "Analysis Agent A becomes unavailable. It does not return an answer."
        ),
        "response_story": (
            "The system marks the analyst unavailable and uses the independently assigned backup."
        ),
        "response_steps": (
            "Detect that Analysis Agent A did not return work.",
            "Mark the agent unavailable without treating absence as dishonesty.",
            "Use Analysis Agent B, which was independently assigned the same capability.",
            "Continue to independent verification before publication.",
        ),
        "why_it_matters": (
            "A single unavailable AI worker does not stop the organization when "
            "redundant capability exists."
        ),
        "mode": FaultMode.OFFLINE,
    },
    "Agent Timeout — One Bounded Retry": {
        "description": "One analyst does not respond within its allowed time window.",
        "failure_story": (
            "Analysis Agent A takes too long to respond and crosses the timeout boundary."
        ),
        "response_story": (
            "The system permits one bounded retry and preserves the reliability event in history."
        ),
        "response_steps": (
            "Detect that Analysis Agent A crossed the allowed response window.",
            "Allow exactly one bounded retry.",
            "Accept the retry only after the normal validation checks pass.",
            "Restore health but retain a small trust penalty and the audit record.",
        ),
        "why_it_matters": (
            "Recovery does not erase history. The system can keep moving while still "
            "remembering that reliability degraded."
        ),
        "mode": FaultMode.TIMEOUT,
    },
    "Malformed Answer — Reject and Retry": {
        "description": "One analyst returns an answer that violates the required data contract.",
        "failure_story": (
            "Analysis Agent A returns a response that cannot pass the required structured-output checks."
        ),
        "response_story": (
            "The invalid artifact is rejected before it can influence the mission."
        ),
        "response_steps": (
            "Reject the malformed answer at the structured-output boundary.",
            "Prevent the invalid artifact from entering mission state.",
            "Allow one bounded retry.",
            "Continue only after the replacement answer passes validation.",
        ),
        "why_it_matters": (
            "A fluent-looking AI answer is not accepted unless it satisfies the application's contract."
        ),
        "mode": FaultMode.MALFORMED_OUTPUT,
    },
    "Weak Evidence — Ask for Corroboration": {
        "description": "One analyst gives an answer without supporting it with the full evidence packet.",
        "failure_story": (
            "Analysis Agent A reaches a conclusion while omitting required evidence."
        ),
        "response_story": (
            "The system refuses to treat an under-supported answer as sufficient authority."
        ),
        "response_steps": (
            "Detect that required evidence is missing from the analyst's work.",
            "Reduce the work's authority instead of trusting the model's confidence.",
            "Ask the independent analyst to corroborate using the full approved evidence set.",
            "Publish only if the corroborated result later satisfies verification.",
        ),
        "why_it_matters": (
            "The system rewards evidence, not confidence or repetition."
        ),
        "mode": FaultMode.UNSUPPORTED_OUTPUT,
    },
    "Agents Disagree — Corroborate Before Acting": {
        "description": "Two AI analysts reach conflicting conclusions.",
        "failure_story": (
            "Analysis Agent A disagrees with the expected evidence-backed conclusion."
        ),
        "response_story": (
            "The system does not resolve disagreement by simple majority vote."
        ),
        "response_steps": (
            "Detect that the two analyses disagree.",
            "Request independent corroboration rather than guessing.",
            "Compare evidence provenance, trust, and verification support.",
            "Publish only when the evidence-backed support is decisive; otherwise escalate.",
        ),
        "why_it_matters": (
            "Multiple AI agents can disagree without forcing the application to guess."
        ),
        "mode": FaultMode.CONTRADICTORY_OUTPUT,
    },
    "Misleading Agent — Quarantine and Replace": {
        "description": "One analyst returns a plausible but demonstrably false operational story.",
        "failure_story": (
            "Analysis Agent A falsely claims that the issuer filed a default notice "
            "confirming the 18% decline and that no approved alternate price is "
            "available, contradicting the approved evidence."
        ),
        "response_story": (
            "The system contains the misleading analyst before its fabricated claim can affect valuation."
        ),
        "response_steps": (
            "Compare the claim with the approved pricing and issuer evidence.",
            "Drop the analyst's trust sharply when the claim is demonstrably unsupported.",
            "Quarantine the analyst so its work has zero publication authority.",
            "Use Analysis Agent B as the independent replacement.",
            "Require verification before the final valuation-control recommendation can publish.",
        ),
        "why_it_matters": (
            "A fabricated financial claim can remain visible for audit while being "
            "prevented from influencing the valuation decision."
        ),
        "mode": FaultMode.MISLEADING_OUTPUT,
    },
    "Role Violation — Block and Replace": {
        "description": "A verifier attempts to operate outside its assigned authority.",
        "failure_story": (
            "Verification Agent A crosses its role boundary instead of staying within verification."
        ),
        "response_story": (
            "The application blocks an AI worker that tries to exercise authority outside its assigned role."
        ),
        "response_steps": (
            "Detect that Verification Agent A crossed its assigned capability boundary.",
            "Reject the out-of-role artifact.",
            "Quarantine the verifier so it cannot influence publication.",
            "Use Verification Agent B to perform the authorized verification job.",
        ),
        "why_it_matters": (
            "Even a technically valid AI response cannot gain authority outside the role the application assigned."
        ),
        "mode": FaultMode.ROLE_VIOLATION,
    },
    "No Backup — Stop and Ask a Human": {
        "description": "The primary analyst fails and the independent backup is also unavailable.",
        "failure_story": (
            "Analysis Agent A is offline, and Analysis Agent B is unavailable too. "
            "The organization no longer has trustworthy independent analysis capability."
        ),
        "response_story": (
            "The system refuses to manufacture an answer when independent capability is gone."
        ),
        "response_steps": (
            "Detect that Analysis Agent A is unavailable.",
            "Discover that the independent backup analyst is unavailable too.",
            "Recognize that the mission no longer has enough trustworthy capability.",
            "Stop automated publication and send the case to a human reviewer.",
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
    "evidence": "Evidence reviewer — checks approved pricing and issuer facts",
    "analysis": "Impact analyst — recommends how to handle the valuation discrepancy",
    "verification": "Independent verifier — checks the recommendation before reporting",
}


def scenario_choices() -> list[str]:
    return list(SCENARIOS)


def _fault_plan(name: str) -> tuple[FaultInjectionPlan, frozenset[str]]:
    config = SCENARIOS[name]
    mode = config["mode"]
    if mode is None:
        raise ValueError("healthy scenario does not use a fault plan")

    if name == "Role Violation — Block and Replace":
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
        elif target_agent_id == agent.agent_id and name in {"Agent Timeout — One Bounded Retry", "Malformed Answer — Reject and Retry"}:
            status = "Recovered after one bounded retry"
        elif target_agent_id == agent.agent_id and technical[2] == "unavailable":
            status = "Unavailable — backup used"
        elif target_agent_id == agent.agent_id and name in {
            "Weak Evidence — Ask for Corroboration",
            "Agents Disagree — Corroborate Before Acting",
        }:
            status = "Work challenged — independent corroboration required"

        if name == "Agent Offline — Backup Takes Over" and agent.agent_id == "analysis-b":
            status = "Backup analyst used"
        if name in {"Misleading Agent — Quarantine and Replace"} and agent.agent_id == "analysis-b":
            status = "Replacement analyst used"
        if name == "Role Violation — Block and Replace" and agent.agent_id == "verification-b":
            status = "Replacement verifier used"

        result.append(
            [
                agent.display_name,
                ROLE_LABELS[agent.role.value],
                status,
            ]
        )
    return result



def _product_line(product, *, status: str, note: str = "") -> str:
    """Render one agent work product in plain English."""

    conclusion = product.conclusion.replace("supported:", "supports ").replace(
        "rejected:", "rejects "
    )
    evidence_count = len(product.evidence_ids)
    extra = f"  \n**System treatment:** {note}" if note else ""
    return (
        f"**Output:** {product.summary}  \n"
        f"**Conclusion:** `{conclusion}`  \n"
        f"**Evidence cited:** {evidence_count} approved items  \n"
        f"**Status:** {status}"
        f"{extra}"
    )


def _evidence_packet_markdown() -> str:
    healthy = run_healthy_mission()
    lines = [
        "### Evidence packet the team receives",
        "Before any agent can recommend an action, the application supplies the same approved evidence packet:",
        "",
    ]
    for item in healthy.evidence:
        lines.append(f"- **{item.source_label}:** {item.content}")
    return "\n".join(lines)


def _healthy_products_by_agent():
    healthy = run_healthy_mission()
    return {
        product.producer_agent_id: product
        for product in healthy.work_products
    }


def _work_walkthrough(
    name: str,
    *,
    plan: FaultInjectionPlan | None = None,
    recovery=None,
    unavailable_agent_ids: frozenset[str] = frozenset(),
) -> str:
    """Show the actual work moving through all six peers and the control plane."""

    healthy = run_healthy_mission()
    products = _healthy_products_by_agent()
    agents = {agent.agent_id: agent.display_name for agent in healthy.agents}

    faulty_product = None
    observation = None
    if plan is not None:
        observation = run_fault_injection(plan).observation
        faulty_product = observation.work_product

    lines = [
        "## Watch the work move through the team",
        "Each pair works independently. The arrows show when the application allows work to move to the next stage.",
        "",
        _evidence_packet_markdown(),
        "",
        "---",
        "### Phase 1 — Independent evidence review",
    ]

    for agent_id in ("evidence-a", "evidence-b"):
        if plan is not None and plan.target_agent_id == agent_id:
            if observation.work_product is None:
                status = "❌ No usable work returned"
                detail = (
                    f"**{agents[agent_id]}**  \n"
                    f"**Job:** Check the approved pricing and issuer facts.  \n"
                    f"**Result:** {plan.mode.value.replace('_', ' ')} — no accepted evidence-review product."
                )
            else:
                status = "❌ Rejected / contained"
                detail = (
                    f"**{agents[agent_id]}**  \n"
                    f"**Job:** Check the approved pricing and issuer facts.  \n"
                    + _product_line(
                        observation.work_product,
                        status=status,
                        note="The reliability layer does not let this faulty artifact become trusted evidence.",
                    )
                )
        else:
            product = products[agent_id]
            detail = (
                f"**{agents[agent_id]}**  \n"
                f"**Job:** Check the approved pricing and issuer facts.  \n"
                + _product_line(product, status="✅ Accepted")
            )
        lines.extend(["", detail])

    lines.extend([
        "",
        "**Gate 1:** Evidence review must complete before analysis is allowed to proceed.",
        "",
        "---",
        "### Phase 2 — Independent impact analysis",
    ])

    for agent_id in ("analysis-a", "analysis-b"):
        if agent_id in unavailable_agent_ids:
            detail = (
                f"**{agents[agent_id]}**  \n"
                "**Job:** Recommend how to handle the valuation discrepancy.  \n"
                "**Result:** No work returned.  \n"
                "**Status:** ⛔ Unavailable"
            )
        elif plan is not None and plan.target_agent_id == agent_id:
            mode = plan.mode
            if mode is FaultMode.OFFLINE:
                detail = (
                    f"**{agents[agent_id]}**  \n"
                    "**Job:** Recommend how to handle the valuation discrepancy.  \n"
                    "**Result:** No response.  \n"
                    "**Status:** ⛔ Unavailable"
                )
            elif mode is FaultMode.TIMEOUT:
                retry = next(
                    (
                        product
                        for product in recovery.recovery_work_products
                        if product.producer_agent_id == agent_id
                    ),
                    None,
                )
                detail = (
                    f"**{agents[agent_id]}**  \n"
                    "**First attempt:** Timed out before usable work arrived.  \n"
                    "**System action:** One bounded retry.  \n"
                    + (
                        _product_line(
                            retry,
                            status="✅ Retry accepted",
                            note=(
                                f"Trust remains {recovery.trust_after_recovery.score:.2f}; "
                                "successful recovery does not erase the timeout."
                            ),
                        )
                        if retry is not None
                        else "**Status:** ❌ Retry did not produce accepted work."
                    )
                )
            elif mode is FaultMode.MALFORMED_OUTPUT:
                retry = next(
                    (
                        product
                        for product in recovery.recovery_work_products
                        if product.producer_agent_id == agent_id
                    ),
                    None,
                )
                detail = (
                    f"**{agents[agent_id]}**  \n"
                    "**First attempt:** Returned malformed structured output.  \n"
                    "**System action:** Reject before mission state, then allow one bounded retry.  \n"
                    + (
                        _product_line(
                            retry,
                            status="✅ Retry accepted",
                            note=(
                                f"Trust remains {recovery.trust_after_recovery.tier.value} "
                                f"({recovery.trust_after_recovery.score:.2f})."
                            ),
                        )
                        if retry is not None
                        else "**Status:** ❌ No accepted retry."
                    )
                )
            elif faulty_product is not None:
                if mode is FaultMode.MISLEADING_OUTPUT:
                    note = (
                        f"Trust falls from {recovery.assessment.trust_before.score:.2f} "
                        f"to {recovery.assessment.trust_after.score:.2f}; agent is quarantined "
                        "and this work gets zero publication authority."
                    )
                    status = "🚫 Quarantined"
                elif mode is FaultMode.UNSUPPORTED_OUTPUT:
                    note = "Incomplete evidence lineage triggers independent corroboration."
                    status = "⚠️ Challenged"
                elif mode is FaultMode.CONTRADICTORY_OUTPUT:
                    note = "Conflicting conclusion triggers independent corroboration."
                    status = "⚠️ Disputed"
                else:
                    note = "Faulty work is contained by the reliability layer."
                    status = "❌ Rejected"
                detail = (
                    f"**{agents[agent_id]}**  \n"
                    "**Job:** Recommend how to handle the valuation discrepancy.  \n"
                    + _product_line(
                        faulty_product,
                        status=status,
                        note=note,
                    )
                )
            else:
                detail = (
                    f"**{agents[agent_id]}**  \n"
                    "**Status:** ❌ No accepted work product."
                )
        else:
            source = products[agent_id]
            replacement = None
            if recovery is not None:
                replacement = next(
                    (
                        product
                        for product in recovery.recovery_work_products
                        if product.producer_agent_id == agent_id
                    ),
                    None,
                )
            used = replacement or source
            status = "✅ Accepted independently"
            note = ""
            if replacement is not None:
                status = "✅ Used for recovery"
                if plan.mode in {
                    FaultMode.UNSUPPORTED_OUTPUT,
                    FaultMode.CONTRADICTORY_OUTPUT,
                }:
                    note = "This independent analysis provides the requested corroboration."
                else:
                    note = "This independent analysis replaces the failed or quarantined peer."
            detail = (
                f"**{agents[agent_id]}**  \n"
                "**Job:** Recommend how to handle the valuation discrepancy.  \n"
                + _product_line(used, status=status, note=note)
            )
        lines.extend(["", detail])

    if recovery is not None and recovery.status.value == "human_review_required":
        lines.extend([
            "",
            "**Gate 2 stops here:** The organization no longer has enough independent analysis capability.",
            "",
            "### Reliability protocol decision",
            "**STOP AUTOMATION → HUMAN REVIEW.** Verification is not allowed to manufacture a missing analysis layer.",
        ])
        return "\n".join(lines)

    lines.extend([
        "",
        "**Gate 2:** Only accepted analysis work can move to independent verification.",
        "",
        "---",
        "### Phase 3 — Independent verification",
    ])

    for agent_id in ("verification-a", "verification-b"):
        if plan is not None and plan.target_agent_id == agent_id:
            if faulty_product is not None and plan.mode is FaultMode.ROLE_VIOLATION:
                detail = (
                    f"**{agents[agent_id]}**  \n"
                    "**Job:** Check the recommendation before reporting.  \n"
                    + _product_line(
                        faulty_product,
                        status="🚫 Quarantined",
                        note=(
                            "The artifact used a capability outside this agent's assignment. "
                            "The application blocks it regardless of how fluent the answer looks."
                        ),
                    )
                )
            elif observation is not None and observation.work_product is None:
                detail = (
                    f"**{agents[agent_id]}**  \n"
                    "**Job:** Check the recommendation before reporting.  \n"
                    "**Status:** ❌ No usable verification returned."
                )
            else:
                detail = (
                    f"**{agents[agent_id]}**  \n"
                    + _product_line(products[agent_id], status="✅ Accepted")
                )
        else:
            product = products[agent_id]
            replacement = None
            if recovery is not None:
                replacement = next(
                    (
                        candidate
                        for candidate in recovery.recovery_work_products
                        if candidate.producer_agent_id == agent_id
                    ),
                    None,
                )
            used = replacement or product
            status = "✅ Verification accepted"
            note = ""
            if replacement is not None:
                status = "✅ Replacement verifier accepted"
                note = "This verifier replaces the quarantined peer."
            detail = (
                f"**{agents[agent_id]}**  \n"
                "**Job:** Check the recommendation and evidence lineage before reporting.  \n"
                + _product_line(used, status=status, note=note)
            )
        lines.extend(["", detail])

    lines.extend([
        "",
        "**Gate 3:** Verification does not publish by itself. The deterministic consensus policy still checks evidence completeness, trust, independence, and support margin.",
        "",
        "---",
        "### Phase 4 — Reliability protocol and publication",
    ])

    if recovery is None:
        lines.extend([
            "**Health / trust action:** None needed.",
            "**Consensus:** Both analyses converge on `MITIGATE`; both verifiers support it.",
            "**Publication:** ✅ Allowed.",
        ])
    else:
        actions = " → ".join(event.action.value.upper() for event in recovery.recovery_events)
        lines.extend([
            f"**Recovery path:** {actions}",
            f"**Target-agent trust after fault:** {recovery.trust_after_recovery.tier.value} ({recovery.trust_after_recovery.score:.2f})",
            "**Publication:** The recovered work still has to pass the normal trust-aware consensus gate.",
        ])

    return "\n".join(lines)

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
            "MITIGATE — quarantine the suspect primary price, use the approved "
            "secondary price under the documented exception process, and investigate "
            "the discrepancy before restoring the primary feed to automated use."
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
        "✅ The system completed the mission safely"
        if mission_success
        else "🧑‍⚖️ Automation stopped — asked for human review"
    )
    steps = "\n".join(
        f"{index}. {step}"
        for index, step in enumerate(config["response_steps"], start=1)
    )
    return (
        f"## {headline}\n\n"
        f"### Scenario\n**{name}**\n\n"
        f"{config['description']}\n\n"
        "### What failed\n"
        f"{config['failure_story']}\n\n"
        "### What the system did\n"
        f"{steps}\n\n"
        "### Business outcome\n"
        f"**{_plain_decision(recommendation)}**\n\n"
        f"Safe outcome: **{'Yes' if safe_outcome else 'No'}**\n\n"
        "### Why this matters\n"
        f"{config['why_it_matters']}"
    )


def _executive_rows(name: str, metrics, recommendation) -> list[list[str]]:
    recovery_action = "No recovery needed"
    if metrics.human_escalation:
        recovery_action = "Stop automation and send the case to a human"
    elif metrics.quarantine_count:
        recovery_action = "Quarantine the unreliable AI and use an independent replacement"
    elif metrics.substitution_count:
        recovery_action = "Use an independent backup AI"
    elif metrics.corroboration_count:
        recovery_action = "Request independent corroboration"
    elif metrics.retry_count:
        recovery_action = "Allow one bounded retry"

    return [
        ["Problem", "Primary price shows an unexplained 18% decline"],
        ["Response", recovery_action],
        ["Decision", _plain_decision(recommendation)],
        ["Human review", "Required" if metrics.human_escalation else "Not required"],
    ]


def run_demo_scenario(name: str) -> dict[str, object]:
    if name not in SCENARIOS:
        raise ValueError("unknown demo scenario")

    if name == "Healthy Team — Nothing Fails":
        healthy = run_healthy_mission()
        report = build_healthy_report()
        recommendation = healthy.consensus.recommendation
        return {
            "walkthrough": _work_walkthrough(name),
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
        "walkthrough": _work_walkthrough(
            name,
            plan=plan,
            recovery=recovery,
            unavailable_agent_ids=unavailable,
        ),
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
        "**What the AI was not allowed to decide:** its identity, its role, its "
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
