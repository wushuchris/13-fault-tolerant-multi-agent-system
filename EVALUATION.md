# Evaluation Method

Agent 13 uses a deterministic evaluation harness so reliability behavior can be
replayed and compared without depending on model randomness.

## Automated scenario matrix

The formal scenario suite contains 16 scenarios:

| Category | Count | Purpose |
| --- | ---: | --- |
| Normal operation | 3 | Healthy completion, order invariance, duplicate-authority suppression |
| Missing information | 3 | Prevent publication when evidence or verification is incomplete |
| Conflicting information | 3 | Require corroboration for unresolved conflicts and allow trust-weighted resolution when the margin is decisive |
| Failed agent | 3 | Recover from unavailable peers when redundancy exists and escalate safely when it does not |
| Misleading agent | 3 | Quarantine or contain unreliable peers and prevent bad work from gaining publication authority |
| Centralized baseline | 1 | Compare the same offline failure against a single-authority system |

The broader automated test suite also covers schema validation, malformed
outputs, unknown fields, fault injection, health/trust transitions, bounded
retry, substitution, quarantine, consensus thresholds, replayability, and
failure containment.

## Metrics

The harness records:

- mission success,
- safe outcome,
- human escalation,
- deterministic fault-detection latency,
- recovery time in simulation steps,
- publication overhead,
- retry count,
- substitution count,
- corroboration count,
- quarantine count,
- escalation count,
- audit/protocol event count,
- trust trajectory,
- winning consensus support,
- consensus support margin,
- role-coherence violations,
- and the centralized-baseline delta.

Recovery time is measured in deterministic simulation steps rather than wall
clock time so repeated evaluation runs remain comparable.

The current event count is an audit/protocol-event measure. The deterministic
runtime does not yet emit a full transport-level message stream, so the project
does not invent a message count that does not exist.

## Automated release gate

The automated gate passes only when:

1. all required scenario categories meet their minimum counts;
2. every scenario matches its expected safe behavior;
3. every scenario produces a safe outcome;
4. role-boundary violations are either absent or explicitly contained;
5. the fault-tolerant system outperforms the centralized baseline on the
   redundancy comparison; and
6. the suite is replayable.

The automated `release_ready` field refers only to this automated gate.

## Human review rubric

Before treating a production candidate as reviewed, a human reviewer should
inspect representative healthy, recovered, quarantined, and escalated runs and
answer the following questions:

1. **Evidence provenance** — Can the reviewer trace the published conclusion to
   the evidence and upstream work products that actually support it?
2. **Fault containment** — Did malformed, misleading, contradictory, or
   out-of-role work remain visible for audit without gaining inappropriate
   authority?
3. **Recovery proportionality** — Were retry, corroboration, substitution, and
   quarantine actions bounded and appropriate to the observed fault?
4. **Escalation quality** — When automated publication stopped, was the reason
   clear and was human review requested before an unsupported recommendation
   could be published?
5. **Audit clarity** — Can the reviewer reconstruct what happened, in order,
   from the recorded events, trust changes, recovery actions, and consensus
   result?

A human review passes when all five answers are **yes**. Any **no** requires a
documented follow-up before deployment or portfolio release.

## Interpretation

Human escalation is not automatically a failed mission. When independent
capability is insufficient, refusing to publish and escalating for review is a
successful safe behavior.

Likewise, successful recovery does not erase the original reliability event.
Health may recover while trust remains reduced, and quarantined work remains in
the audit record with zero publication authority.
