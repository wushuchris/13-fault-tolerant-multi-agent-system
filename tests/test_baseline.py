from collections import Counter

from fault_tolerant_agents.baseline import (
    build_default_team,
    build_healthy_mission,
    create_bids,
    run_healthy_mission,
    settle_assignments,
)
from fault_tolerant_agents.schemas import (
    AgentRole,
    ConsensusStatus,
    HealthStatus,
    MissionRecommendation,
    MissionStatus,
    TaskStatus,
    TrustTier,
)


def test_default_team_has_two_peers_per_role() -> None:
    team = build_default_team()

    assert len(team) == 6
    assert Counter(agent.role for agent in team) == {
        AgentRole.EVIDENCE: 2,
        AgentRole.ANALYSIS: 2,
        AgentRole.VERIFICATION: 2,
    }


def test_healthy_tasks_encode_dependency_chain() -> None:
    _, _, tasks = build_healthy_mission()

    assert tasks[0].depends_on_task_ids == ()
    assert tasks[1].depends_on_task_ids == ("task-evidence",)
    assert tasks[2].depends_on_task_ids == ("task-analysis",)
    assert all(task.redundancy_required == 2 for task in tasks)


def test_bidding_and_assignment_are_deterministic() -> None:
    _, _, tasks = build_healthy_mission()
    team = build_default_team()

    bids = create_bids(team, tasks)
    first = settle_assignments(tasks, bids)
    second = settle_assignments(tasks, bids)

    assert first == second
    assert len(bids) == 6
    assert len(first) == 6
    assert [assignment.agent_id for assignment in first] == [
        "evidence-a",
        "evidence-b",
        "analysis-a",
        "analysis-b",
        "verification-a",
        "verification-b",
    ]


def test_healthy_mission_completes_all_tasks() -> None:
    run = run_healthy_mission()

    assert run.state.status is MissionStatus.COMPLETED
    assert set(run.state.task_statuses.values()) == {TaskStatus.COMPLETED}
    assert run.state.quarantined_agent_ids == frozenset()


def test_healthy_mission_keeps_all_peers_healthy_and_trusted() -> None:
    run = run_healthy_mission()

    assert all(
        state.status is HealthStatus.HEALTHY
        for state in run.health_states
    )
    assert all(
        state.tier is TrustTier.TRUSTED and state.score == 1.0
        for state in run.trust_states
    )


def test_healthy_mission_produces_two_products_per_stage() -> None:
    run = run_healthy_mission()
    counts = Counter(product.task_id for product in run.work_products)

    assert counts == {
        "task-evidence": 2,
        "task-analysis": 2,
        "task-verification": 2,
    }


def test_downstream_work_products_preserve_lineage() -> None:
    run = run_healthy_mission()

    evidence_products = {
        product.work_product_id
        for product in run.work_products
        if product.task_id == "task-evidence"
    }
    analysis_products = {
        product.work_product_id
        for product in run.work_products
        if product.task_id == "task-analysis"
    }

    for product in run.work_products:
        if product.task_id == "task-analysis":
            assert set(product.input_work_product_ids) == evidence_products
        if product.task_id == "task-verification":
            assert set(product.input_work_product_ids) == analysis_products


def test_healthy_consensus_publishes_only_after_verification() -> None:
    run = run_healthy_mission()

    assert run.consensus.status is ConsensusStatus.CONSENSUS_REACHED
    assert run.consensus.recommendation is MissionRecommendation.MITIGATE
    assert run.consensus.human_review_required is False
    assert len(run.consensus.supporting_work_product_ids) == 2
    assert all(
        work_product_id.startswith("wp-task-verification-")
        for work_product_id in run.consensus.supporting_work_product_ids
    )


def test_audit_sequence_is_contiguous_and_ends_with_completion() -> None:
    run = run_healthy_mission()

    assert [event.sequence for event in run.audit_events] == list(
        range(len(run.audit_events))
    )
    assert run.audit_events[0].event_type == "mission_started"
    assert run.audit_events[-1].event_type == "mission_completed"


def test_healthy_mission_is_replayable() -> None:
    first = run_healthy_mission()
    second = run_healthy_mission()

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
