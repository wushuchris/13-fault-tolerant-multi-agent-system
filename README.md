# Fault-Tolerant Multi-Agent System

Agent 13 in the **30 Agents for AI Engineers** portfolio.

## Purpose

Build a multi-agent system that can continue operating when one or more agents fail, disappear, contradict each other, or provide misleading information.

The engineering focus is organizational resilience rather than simply adding more agents. The system detects unreliable behavior, adjusts trust through application-owned policy, requests independent corroboration, substitutes failed capability, and escalates when safe recovery is no longer possible.

## Core Pattern

> **Detect → Distrust → Corroborate → Substitute → Recover → Escalate**

A second design principle carries forward from the earlier multi-agent projects:

> **Agents contribute work. The reliability protocol decides whom to trust and how to recover.**

## System

The MVP models a synthetic operations-intelligence team with redundant capabilities:

- Evidence Agent A / B
- Analysis Agent A / B
- Verification Agent A / B

The system includes:

- deterministic failure injection,
- independent health and trust state,
- redundant task assignment,
- corroboration under disagreement,
- bounded retry and substitution,
- trust-aware consensus,
- quarantine,
- human escalation,
- append-only audit events,
- recovery metrics,
- a 16-scenario evaluation harness,
- a centralized baseline comparison,
- and an optional bounded LLM specialist adapter.

## Control Boundary

LLMs may produce bounded specialist content, but application code retains authority over:

- agent identity,
- capability authorization,
- task identity,
- work-product identity,
- approved evidence,
- health state,
- trust state,
- task assignment validity,
- timeout policy,
- recovery policy,
- consensus thresholds,
- quarantine,
- publication,
- and human escalation.

The model is treated as an untrusted proposer behind schema validation and application-owned authority checks.

## Current Status

**Deterministic reliability core, evaluation harness, Gradio demo, and bounded LLM adapter implemented.**

The automated test suite covers healthy operation, fault injection, trust transitions, bounded recovery, consensus, audit metrics, role containment, centralized baseline comparison, runtime secret gates, and UI construction.

See [EVALUATION.md](EVALUATION.md) for the formal evaluation method and human-review rubric.

## Development

Python 3.11 is the target runtime.

Install dependencies:

~~~bash
python -m pip install -r requirements.txt
~~~

Run tests:

~~~bash
python -m pytest -q
~~~

Run the local Gradio app:

~~~bash
python app.py
~~~

## Repository Structure

~~~text
.
├── app.py
├── src/
│   └── fault_tolerant_agents/
├── tests/
├── scripts/
│   └── build_space_bundle.py
├── .github/
│   └── workflows/
│       ├── tests.yml
│       └── deploy-hf.yml
├── EVALUATION.md
├── .env.example
├── requirements.txt
└── README.md
~~~

## Hugging Face Deployment

GitHub is the source of truth.

The deployment workflow is gated behind the successful **Tests** workflow and checks out the exact tested commit SHA before deployment.

The workflow builds an explicit public allowlist bundle containing only the runtime files required by the Space. It does not mirror the complete GitHub repository.

GitHub deployment configuration:

- Repository variable **HF_SPACE_ID** — target Space in username/space-name form.
- Repository secret **HF_DEPLOY_TOKEN** — write-capable deployment token scoped to the target Space.

Hugging Face Space runtime configuration:

- Space secret **HF_TOKEN** — runtime inference credential.
- Space variable **MODEL_ID** — model identifier used by the optional Live Specialist tab.

**HF_DEPLOY_TOKEN** and **HF_TOKEN** serve different purposes and must remain separate.

No secret values belong in source control.
