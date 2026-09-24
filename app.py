from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr

from fault_tolerant_agents.ui import (
    evaluation_dashboard,
    live_inference_status,
    run_demo_scenario,
    run_live_analysis,
    scenario_choices,
)


def demo_callback(name):
    view = run_demo_scenario(name)
    return (
        view["summary"],
        view["agents"],
        view["recovery"],
        view["metrics"],
        view["audit"],
        view["consensus"],
    )


def evaluation_callback():
    view = evaluation_dashboard()
    return view["summary"], view["results"], view["comparison"]


def live_callback():
    message, payload = run_live_analysis()
    return message, payload


with gr.Blocks(title="Fault-Tolerant Multi-Agent System") as demo:
    gr.Markdown(
        "# Fault-Tolerant Multi-Agent System\n"
        "### Resilient Operations Intelligence Team\n\n"
        "A six-peer AI organization that detects unreliable behavior, adjusts "
        "trust, corroborates disputed work, substitutes failed capability, and "
        "escalates when safe automation is no longer possible.\n\n"
        "**Core pattern:** Detect → Distrust → Corroborate → Substitute → Recover → Escalate"
    )

    with gr.Tab("Mission Demo"):
        gr.Markdown(
            "Choose a reliability scenario. The business outcome appears first; "
            "technical detail remains available underneath for auditability."
        )
        scenario = gr.Dropdown(
            choices=scenario_choices(),
            value="Healthy Mission",
            label="Scenario",
        )
        run_button = gr.Button("Run Scenario", variant="primary")
        outcome = gr.Markdown()

        gr.Markdown("### Team health and trust")
        agents = gr.Dataframe(
            headers=[
                "Agent",
                "Role",
                "Health",
                "Trust Tier",
                "Trust Score",
                "Quarantined",
            ],
            interactive=False,
        )

        gr.Markdown("### Recovery actions")
        recovery = gr.Dataframe(
            headers=["Action", "Detail", "Affected agents"],
            interactive=False,
        )

        gr.Markdown("### Reliability metrics")
        metrics = gr.Dataframe(
            headers=["Metric", "Value"],
            interactive=False,
        )

        with gr.Accordion("Audit trail", open=False):
            audit = gr.Dataframe(
                headers=["Sequence", "Event", "Actor", "Detail"],
                interactive=False,
            )

        with gr.Accordion("Consensus object", open=False):
            consensus = gr.JSON()

        run_button.click(
            fn=demo_callback,
            inputs=[scenario],
            outputs=[outcome, agents, recovery, metrics, audit, consensus],
        )
        demo.load(
            fn=demo_callback,
            inputs=[scenario],
            outputs=[outcome, agents, recovery, metrics, audit, consensus],
        )

    with gr.Tab("Evaluation"):
        gr.Markdown(
            "The deterministic evaluation harness covers 16 scenarios across "
            "normal operation, missing information, conflicts, failed agents, "
            "misleading agents, and a centralized baseline comparison."
        )
        eval_button = gr.Button("Run Evaluation Suite")
        eval_summary = gr.Markdown()
        eval_results = gr.Dataframe(
            headers=[
                "Scenario",
                "Category",
                "Result",
                "Consensus",
                "Mission Success",
                "Human Escalation",
                "Recovery Steps",
            ],
            interactive=False,
        )
        comparison = gr.JSON(label="Centralized baseline comparison")
        eval_button.click(
            fn=evaluation_callback,
            outputs=[eval_summary, eval_results, comparison],
        )

    with gr.Tab("Live Specialist"):
        gr.Markdown(
            "Optional bounded model inference. The model may draft an Analysis "
            "work product, but application code still owns identity, authority, "
            "trust, recovery, consensus, and publication."
        )
        live_status = gr.Markdown(value=live_inference_status())
        live_button = gr.Button("Run Bounded Analysis Specialist")
        live_message = gr.Markdown()
        live_product = gr.JSON(label="Validated WorkProduct")
        live_button.click(
            fn=live_callback,
            outputs=[live_message, live_product],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
