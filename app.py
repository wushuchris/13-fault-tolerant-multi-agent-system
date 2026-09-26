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


with gr.Blocks(title="Fault-Tolerant Multi-Agent System") as demo:
    gr.Markdown(
        "# Fault-Tolerant Multi-Agent System\n"
        "### Can an AI team keep working safely when one of its members fails or gives a bad answer?\n\n"
        "This demo deliberately breaks members of a six-agent financial-services AI team and "
        "shows how the surrounding software detects the problem, limits the bad "
        "agent's authority, recovers when possible, and asks a human when it cannot "
        "recover safely.\n\n"
        "**Engineering principle:** Agents contribute work. The application decides "
        "whom to trust and whether the result is safe to publish."
    )

    with gr.Tab("Story Demo"):
        gr.Markdown(
            "## The business case\n"
            f"{BUSINESS_CASE}\n\n"
            "**The AI team:** two agents confirm the facts, two independently analyze "
            "the valuation impact, and two independently verify the recommendation. "
            "Choose something to break and watch what the system does."
        )
        scenario = gr.Dropdown(
            choices=scenario_choices(),
            value="Healthy Mission",
            label="Choose what goes wrong",
        )
        run_button = gr.Button("Run the scenario", variant="primary")

        story = gr.Markdown()

        gr.Markdown("### At a glance")
        executive = gr.Dataframe(
            headers=["Question", "Answer"],
            interactive=False,
        )

        gr.Markdown("### What happened to the AI team")
        team = gr.Dataframe(
            headers=["AI teammate", "Plain-English job", "Status in this scenario"],
            interactive=False,
        )

        with gr.Accordion(
            "Engineering details — trust, recovery, metrics, audit, and consensus",
            open=False,
        ):
            gr.Markdown(
                "These are the implementation-level artifacts behind the story above."
            )
            gr.Markdown("#### Agent health and trust")
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

        run_button.click(
            fn=demo_callback,
            inputs=[scenario],
            outputs=[
                story,
                executive,
                team,
                agents,
                recovery,
                metrics,
                audit,
                consensus,
            ],
        )
        demo.load(
            fn=demo_callback,
            inputs=[scenario],
            outputs=[
                story,
                executive,
                team,
                agents,
                recovery,
                metrics,
                audit,
                consensus,
            ],
        )

    with gr.Tab("Reliability Evaluation"):
        gr.Markdown(
            "## Does the system behave safely across many failure cases?\n"
            "This tab runs the formal reliability suite. The summary is written for "
            "a general audience; the scenario table and baseline object preserve the "
            "engineering evidence."
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

    with gr.Tab("Live AI Analyst"):
        gr.Markdown(
            "## Now use a real language model\n"
            "The Story Demo is deterministic so failures are reproducible. This tab "
            "calls a real hosted language model for the **Analysis Agent** job. "
            "The model can draft an analysis of the pricing discrepancy, but it cannot choose its identity, "
            "change trust, recover the mission, or approve publication."
        )
        live_status = gr.Markdown(value=live_inference_status())
        live_button = gr.Button("Ask the live AI analyst", variant="primary")
        live_message = gr.Markdown()
        live_story = gr.Markdown()
        with gr.Accordion("Engineering artifact — validated WorkProduct", open=False):
            live_product = gr.JSON(label="Validated WorkProduct")
        live_button.click(
            fn=live_callback,
            outputs=[live_message, live_story, live_product],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
