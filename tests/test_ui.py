import pytest

from fault_tolerant_agents.ui import (
    BUSINESS_CASE,
    evaluation_dashboard,
    live_inference_status,
    live_product_story,
    run_demo_scenario,
    run_live_analysis,
    scenario_choices,
)


def test_demo_exposes_expected_business_scenarios() -> None:
    assert scenario_choices() == [
        "Healthy Team — Nothing Fails",
        "Agent Offline — Backup Takes Over",
        "Agent Timeout — One Bounded Retry",
        "Malformed Answer — Reject and Retry",
        "Weak Evidence — Ask for Corroboration",
        "Agents Disagree — Corroborate Before Acting",
        "Misleading Agent — Quarantine and Replace",
        "Role Violation — Block and Replace",
        "No Backup — Stop and Ask a Human",
    ]


def test_business_case_is_explained_in_plain_language() -> None:
    assert "asset manager" in BUSINESS_CASE
    assert "18%" in BUSINESS_CASE
    assert "$99.20" in BUSINESS_CASE
    assert "corporate action" in BUSINESS_CASE


def test_healthy_demo_tells_a_complete_story() -> None:
    view = run_demo_scenario("Healthy Team — Nothing Fails")

    assert "Scenario" in view["story"]
    assert "What failed" in view["story"]
    assert "What the system did" in view["story"]
    assert "Business outcome" in view["story"]
    assert "Why this matters" in view["story"]
    assert "MITIGATE" in view["story"]
    assert len(view["team"]) == 6
    assert view["recovery"][0][0] == "none"


def test_misleading_demo_explains_quarantine_in_plain_english() -> None:
    view = run_demo_scenario("Misleading Agent — Quarantine and Replace")

    actions = [row[0] for row in view["recovery"]]
    analysis_a = next(
        row for row in view["agents"] if row[0] == "Analysis Agent A"
    )
    story_row = next(
        row for row in view["team"] if row[0] == "Analysis Agent A"
    )

    assert actions == ["quarantine", "substitute"]
    assert analysis_a[3] == "quarantined"
    assert analysis_a[5] == "yes"
    assert "Quarantined" in story_row[2]
    assert "zero publication authority" in view["story"]
    assert "default notice" in view["story"]
    assert "completed the mission safely" in view["story"]


def test_no_backup_demo_explains_safe_human_escalation() -> None:
    view = run_demo_scenario("No Backup — Stop and Ask a Human")

    assert "asked for human review" in view["story"]
    assert "No automated decision" in view["story"]
    assert view["consensus"]["human_review_required"] is True
    executive = dict(view["executive"])
    assert executive["Human review"] == "Required"


def test_role_violation_demo_targets_verification_peer() -> None:
    view = run_demo_scenario("Role Violation — Block and Replace")
    verification_a = next(
        row for row in view["agents"] if row[0] == "Verification Agent A"
    )
    verifier_story = next(
        row for row in view["team"] if row[0] == "Verification Agent A"
    )

    assert verification_a[3] == "quarantined"
    assert verification_a[5] == "yes"
    assert "Quarantined" in verifier_story[2]


def test_evaluation_dashboard_explains_what_is_being_tested() -> None:
    dashboard = evaluation_dashboard()

    assert "reliability test suite" in dashboard["summary"]
    assert "16/16" in dashboard["summary"]
    assert "Safe outcome rate" in dashboard["summary"]
    assert len(dashboard["results"]) == 16
    assert dashboard["comparison"]["baseline_delta"] == 1


def test_live_inference_status_does_not_expose_secret_values(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret-value-that-must-not-render")
    monkeypatch.setenv("MODEL_ID", "example/model")

    status = live_inference_status()

    assert "ready" in status
    assert "secret-value-that-must-not-render" not in status


def test_live_product_story_explains_engineering_boundary() -> None:
    story = live_product_story(
        {
            "conclusion": "mitigate",
            "evidence_ids": ["evidence-001", "evidence-002"],
            "confidence": 0.95,
        }
    )

    assert "passed the application's checks" in story
    assert "MITIGATE" in story
    assert "2 approved evidence items" in story
    assert "not allowed to decide" in story


def test_live_analysis_without_space_secrets_fails_cleanly(
    monkeypatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)

    message, payload = run_live_analysis()

    assert "not configured" in message
    assert payload == {}


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown demo scenario"):
        run_demo_scenario("Not A Scenario")


def test_root_gradio_app_constructs_without_launching() -> None:
    import app

    assert app.demo is not None
    assert app.demo.title == "Fault-Tolerant Multi-Agent System"



def test_guided_story_uses_numbered_system_response_steps() -> None:
    view = run_demo_scenario("Misleading Agent — Quarantine and Replace")

    assert "1. Compare the claim" in view["story"]
    assert "2. Drop the analyst's trust" in view["story"]
    assert "3. Quarantine the analyst" in view["story"]
    assert "4. Use Analysis Agent B" in view["story"]


def test_app_uses_narrow_reading_width() -> None:
    import app

    assert "max-width: 900px" in app.APP_CSS
