# Fault-Tolerant Multi-Agent System

Agent 13 in the **30 Agents for AI Engineers** portfolio.

## Purpose

Build a multi-agent system that can continue operating when one or more agents fail, disappear, contradict each other, or provide misleading information.

The engineering focus is organizational resilience rather than simply adding more agents. The system will detect unreliable behavior, adjust trust through application-owned policy, request independent corroboration, substitute failed capability, and escalate when safe recovery is no longer possible.

## Core Pattern

> **Detect → Distrust → Corroborate → Substitute → Recover → Escalate**

A second design principle carries forward from the earlier multi-agent projects:

> **Agents contribute work. The reliability protocol decides whom to trust and how to recover.**

## Planned MVP

The first version will model a synthetic operations-intelligence team with redundant capabilities:

- Evidence Agent A / B
- Analysis Agent A / B
- Verification Agent A / B

The system will demonstrate:

- deterministic failure injection,
- health and trust state,
- redundant task assignment,
- corroboration under disagreement,
- recovery and reassignment,
- trust-aware consensus,
- human escalation,
- append-only audit events,
- and recovery metrics.

## Control Boundary

LLMs may eventually produce bounded substantive work products.

Application code will retain authority over:

- agent identity,
- capability authorization,
- health state,
- trust state,
- task assignment validity,
- timeout policy,
- recovery policy,
- consensus thresholds,
- quarantine,
- publication,
- and human escalation.

## Current Status

**Scaffolding only.**

The architecture has been defined, but Agent 13 schemas and fault-tolerance logic have not yet been implemented. Development is intentionally incremental and test-first.

## Development

Python 3.11 is the initial target.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -q
```

## Repository Structure

```text
.
├── src/
│   └── fault_tolerant_agents/
├── tests/
├── .github/
│   └── workflows/
├── .env.example
├── requirements.txt
└── README.md
```

## Deployment

GitHub is the source of truth. A Hugging Face Space and automatic deployment workflow will be added only after the deterministic core and automated evaluation are working.

Deployment and runtime credentials will remain separate:

- `HF_DEPLOY_TOKEN` — GitHub Actions deployment secret
- `HF_TOKEN` — Hugging Face Space runtime inference secret

No secret values belong in source control.
