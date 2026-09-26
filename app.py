from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr

from fault_tolerant_agents.ui import (
    BUSINESS_CASE,
    evaluation_dashboard,
    live_inference_status,
    live_product_story,
    run_demo_scenario,
    run_live_analysis,
    scenario_choices,
)


APP_CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}
.prose, .markdown {
    line-height: 1.6;
}
"""


def demo_callback(name):
    view = run_demo_scenario(name)
    return (
        view["story"],
        view["executive"],
        view["team"],
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
    return message, live_product_story(payload), payload


with gr.Blocks(
    title="Fault-Tolerant Multi-Agent System",
    css=APP_CSS,
) as demo:
    gr.Markdown(
        "# Fault-Tolerant Multi-Agent System\n"
        "### A financial-services demo of resilient AI teamwork\n\n"
        "**What this agent is:** a reliability layer for an AI team. It is not a "
        "trading bot and it does not predict markets. The demo asks a simpler, "
        "important question: **what should happen when one AI worker fails, gives "
        "weak evidence, disagrees with its peers, or returns a misleading answer?**\n\n"
        "**How the six-agent team works:**\n"
        "1. **Two Evidence Reviewers** independently check the approved pricing and issuer facts.\n"
        "2. **Two Impact Analysts** independently recommend how to handle the valuation discrepancy.\n"
        "3. **Two Verifiers** independently check the recommendation before it can influence reporting.\n\n"
        "The AI workers produce work. **Application code—not the model—controls trust, "
        "quarantine, recovery, and publication authority.**"
    )

    with gr.Tab("1 · Guided Story"):
        gr.Markdown(
            "## The financial-services case\n"
            f"{BUSINESS_CASE}\n\n"
            "The normal control response is to quarantine the suspicious primary "
            "price, use the approved secondary price under the firm's exception "
            "process, and investigate the discrepancy.\n\n"
            "### Choose one thing to break\n"
            "The scenario names describe the lesson. For a first look, try "
            "**Misleading Agent — Quarantine and Replace**, then "
            "**No Backup — Stop and Ask a Human**."
        )
        scenario = gr.Dropdown(
            choices=scenario_choices(),
            value="Misleading Agent — Quarantine and Replace",
            label="Failure scenario",
        )
        run_button = gr.Button("Simulate this scenario", variant="primary")

        story = gr.Markdown()

        with gr.Accordion(
            "See the team state and engineering evidence",
            open=False,
        ):
            gr.Markdown(
                "Everything below is the implementation evidence behind the story. "
                "A non-technical viewer can stop above; an engineer can inspect the "
                "exact state transitions here."
            )

            gr.Markdown("#### Executive snapshot")
            executive = gr.Dataframe(
                headers=["Question", "Answer"],
                interactive=False,
            )

            gr.Markdown("#### Six-agent team")
            team = gr.Dataframe(
                headers=["AI teammate", "Job", "Status in this scenario"],
                interactive=False,
            )

            gr.Markdown("#### Health and trust state")
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

            gr.Markdown("#### Recovery actions")
            recovery = gr.Dataframe(
                headers=["Action", "Detail", "Affected agents"],
                interactive=False,
            )

            gr.Markdown("#### Reliability metrics")
            metrics = gr.Dataframe(
                headers=["Metric", "Value"],
                interactive=False,
            )

            gr.Markdown("#### Append-only audit trail")
            audit = gr.Dataframe(
                headers=["Sequence", "Event", "Actor", "Detail"],
                interactive=False,
            )

            gr.Markdown("#### Final consensus object")
            consensus = gr.JSON()

        outputs = [
            story,
            executive,
            team,
            agents,
            recovery,
            metrics,
            audit,
            consensus,
        ]
        run_button.click(
            fn=demo_callback,
            inputs=[scenario],
            outputs=outputs,
        )
        scenario.change(
            fn=demo_callback,
            inputs=[scenario],
            outputs=outputs,
        )
        demo.load(
            fn=demo_callback,
            inputs=[scenario],
            outputs=outputs,
        )

    with gr.Tab("2 · Reliability Tests"):
        gr.Markdown(
            "## Does it keep behaving safely when conditions change?\n"
            "The guided story shows one failure at a time. This tab runs the formal "
            "16-scenario reliability suite across healthy operation, missing evidence, "
            "agent disagreement, failed agents, misleading agents, and loss of backup "
            "capacity.\n\n"
            "A scenario can pass by **stopping and asking a human**. The goal is safe "
            "behavior, not automation at all costs."
        )
        eval_button = gr.Button("Run the 16-scenario reliability suite")
        eval_summary = gr.Markdown()
        with gr.Accordion("Engineering evaluation results", open=False):
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

    with gr.Tab("3 · Live AI Analyst"):
        gr.Markdown(
            "## Replace the deterministic analyst with a real hosted model\n"
            "The Guided Story uses deterministic outputs so every failure is "
            "reproducible. This tab sends the **same approved financial evidence** "
            "to a real language model for one bounded Analysis Agent job.\n\n"
            "The model may summarize the discrepancy and propose a bounded "
            "recommendation. It **cannot** choose its identity, expand its role, "
            "change trust, quarantine another agent, recover the mission, or approve "
            "publication. Those decisions remain in application code."
        )
        live_status = gr.Markdown(value=live_inference_status())
        live_button = gr.Button("Ask the live AI analyst", variant="primary")
        live_message = gr.Markdown()
        live_story = gr.Markdown()
        with gr.Accordion(
            "Engineering artifact — validated WorkProduct",
            open=False,
        ):
            live_product = gr.JSON(label="Validated WorkProduct")
        live_button.click(
            fn=live_callback,
            outputs=[live_message, live_story, live_product],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
