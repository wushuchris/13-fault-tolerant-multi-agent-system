import pytest

from fault_tolerant_agents.ui import (
    evaluation_dashboard,
    live_inference_status,
    run_demo_scenario,
    run_live_analysis,
    scenario_choices,
)


def test_demo_exposes_expected_business_scenarios() -> None:
    assert scenario_choices() == [
        "Healthy Mission",
        "Offline Agent",
        "Timeout",
        "Malformed Output",
        "Unsupported Output",
        "Contradictory Agent",
        "Misleading Agent",
        "Role Violation",
        "No Backup Available",
    ]


def test_healthy_demo_is_business_readable() -> None:
    view = run_demo_scenario("Healthy Mission")

    assert "Mission completed" in view["summary"]
    assert "mitigate" in view["summary"]
    assert len(view["agents"]) == 6
    assert view["recovery"][0][0] == "none"


def test_misleading_demo_shows_quarantine_and_recovery() -> None:
    view = run_demo_scenario("Misleading Agent")

    actions = [row[0] for row in view["recovery"]]
    analysis_a = next(
        row for row in view["agents"] if row[0] == "Analysis Agent A"
    )

    assert actions == ["quarantine", "substitute"]
    assert analysis_a[3] == "quarantined"
    assert analysis_a[5] == "yes"
    assert "Mission completed" in view["summary"]


def test_no_backup_demo_escalates_without_recommendation() -> None:
    view = run_demo_scenario("No Backup Available")

    assert "Human review required" in view["summary"]
    assert "Recommendation:** none" in view["summary"]
    assert view["consensus"]["human_review_required"] is True


def test_role_violation_demo_targets_verification_peer() -> None:
    view = run_demo_scenario("Role Violation")
    verification_a = next(
        row for row in view["agents"] if row[0] == "Verification Agent A"
    )

    assert verification_a[3] == "quarantined"
    assert verification_a[5] == "yes"


def test_evaluation_dashboard_exposes_release_gate() -> None:
    dashboard = evaluation_dashboard()

    assert "PASS" in dashboard["summary"]
    assert len(dashboard["results"]) == 16
    assert dashboard["comparison"]["baseline_delta"] == 1


def test_live_inference_status_does_not_expose_secret_values(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret-value-that-must-not-render")
    monkeypatch.setenv("MODEL_ID", "example/model")

    status = live_inference_status()

    assert "configured" in status
    assert "secret-value-that-must-not-render" not in status


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
