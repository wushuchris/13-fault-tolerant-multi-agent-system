"""Typed contracts for the Agent 13 fault-tolerant multi-agent system.

The models in this module define data boundaries only. They intentionally do
not implement trust updates, recovery behavior, task allocation, or consensus
policy. Those application-owned control rules will be built in later steps.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    ),
]

ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]

LongText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=5000),
]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class AgentRole(str, Enum):
    EVIDENCE = "evidence"
    ANALYSIS = "analysis"
    VERIFICATION = "verification"


class Capability(str, Enum):
    EVIDENCE_REVIEW = "evidence_review"
    IMPACT_ANALYSIS = "impact_analysis"
    CLAIM_VERIFICATION = "claim_verification"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class TrustTier(str, Enum):
    TRUSTED = "trusted"
    WATCH = "watch"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"


class FaultMode(str, Enum):
    NONE = "none"
    OFFLINE = "offline"
    TIMEOUT = "timeout"
    MALFORMED_OUTPUT = "malformed_output"
    UNSUPPORTED_OUTPUT = "unsupported_output"
    CONTRADICTORY_OUTPUT = "contradictory_output"
    MISLEADING_OUTPUT = "misleading_output"
    ROLE_VIOLATION = "role_violation"


class TaskStatus(str, Enum):
    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MissionStatus(str, Enum):
    ACTIVE = "active"
    RECOVERING = "recovering"
    HUMAN_REVIEW = "human_review"
    COMPLETED = "completed"
    FAILED_SAFE = "failed_safe"


class MissionRecommendation(str, Enum):
    CONTINUE = "continue"
    MONITOR = "monitor"
    MITIGATE = "mitigate"
    PAUSE = "pause"


class MessageType(str, Enum):
    TASK_ANNOUNCEMENT = "task_announcement"
    TASK_BID = "task_bid"
    TASK_ASSIGNMENT = "task_assignment"
    WORK_PRODUCT = "work_product"
    CORROBORATION_REQUEST = "corroboration_request"
    VERIFICATION_RESULT = "verification_result"
    RECOVERY_NOTICE = "recovery_notice"
    HUMAN_ESCALATION = "human_escalation"


class RecoveryAction(str, Enum):
    CORROBORATE = "corroborate"
    RETRY = "retry"
    REASSIGN = "reassign"
    SUBSTITUTE = "substitute"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"


class ConsensusStatus(str, Enum):
    CONSENSUS_REACHED = "consensus_reached"
    CORROBORATION_REQUIRED = "corroboration_required"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INSUFFICIENT_TRUST = "insufficient_trust"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


_ROLE_CAPABILITIES: dict[AgentRole, frozenset[Capability]] = {
    AgentRole.EVIDENCE: frozenset({Capability.EVIDENCE_REVIEW}),
    AgentRole.ANALYSIS: frozenset({Capability.IMPACT_ANALYSIS}),
    AgentRole.VERIFICATION: frozenset({Capability.CLAIM_VERIFICATION}),
}


class RoleContract(StrictModel):
    role: AgentRole
    allowed_capabilities: frozenset[Capability] = Field(min_length=1)
    description: ShortText

    @model_validator(mode="after")
    def capabilities_match_role(self) -> "RoleContract":
        allowed = _ROLE_CAPABILITIES[self.role]
        if not self.allowed_capabilities.issubset(allowed):
            raise ValueError("role contract includes capability outside the role boundary")
        return self


class AgentDescriptor(StrictModel):
    agent_id: Identifier
    role: AgentRole
    capabilities: frozenset[Capability] = Field(min_length=1)
    display_name: ShortText

    @model_validator(mode="after")
    def capabilities_match_role(self) -> "AgentDescriptor":
        allowed = _ROLE_CAPABILITIES[self.role]
        if not self.capabilities.issubset(allowed):
            raise ValueError("agent claims capability outside its assigned role")
        return self


class MissionRequest(StrictModel):
    mission_id: Identifier
    title: ShortText
    objective: LongText
    allowed_recommendations: frozenset[MissionRecommendation] = Field(
        default_factory=lambda: frozenset(MissionRecommendation),
        min_length=1,
    )


class TaskSpec(StrictModel):
    task_id: Identifier
    mission_id: Identifier
    required_capability: Capability
    description: LongText
    evidence_ids: tuple[Identifier, ...] = ()
    depends_on_task_ids: tuple[Identifier, ...] = ()
    redundancy_required: int = Field(default=1, ge=1, le=2)


class TaskAssignment(StrictModel):
    assignment_id: Identifier
    task_id: Identifier
    agent_id: Identifier
    capability: Capability
    assignment_slot: int = Field(ge=1, le=2)


class TaskBid(StrictModel):
    bid_id: Identifier
    task_id: Identifier
    agent_id: Identifier
    capability: Capability
    confidence: float = Field(ge=0.0, le=1.0)
    normalized_cost: float = Field(ge=0.0, le=1.0)
    available: bool = True


class MessageEnvelope(StrictModel):
    message_id: Identifier
    correlation_id: Identifier
    sender_agent_id: Identifier
    recipient_agent_ids: tuple[Identifier, ...] = Field(min_length=1)
    message_type: MessageType
    sequence: int = Field(ge=0)
    task_id: Identifier | None = None
    payload_ref: Identifier | None = None
    body: ShortText | None = None

    @model_validator(mode="after")
    def sender_is_not_recipient(self) -> "MessageEnvelope":
        if self.sender_agent_id in self.recipient_agent_ids:
            raise ValueError("sender cannot also be a recipient")
        return self


class EvidenceItem(StrictModel):
    evidence_id: Identifier
    source_id: Identifier
    source_label: ShortText
    content: LongText
    observed_at_step: int = Field(ge=0)


class WorkProduct(StrictModel):
    work_product_id: Identifier
    task_id: Identifier
    producer_agent_id: Identifier
    capability: Capability
    summary: LongText
    conclusion: LongText
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1)
    input_work_product_ids: tuple[Identifier, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    created_at_step: int = Field(ge=0)


class AgentHealthState(StrictModel):
    agent_id: Identifier
    status: HealthStatus
    consecutive_failures: int = Field(default=0, ge=0)
    last_event_step: int = Field(ge=0)


class AgentTrustState(StrictModel):
    agent_id: Identifier
    tier: TrustTier
    score: float = Field(ge=0.0, le=1.0)
    validated_work_count: int = Field(default=0, ge=0)
    failed_verification_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)
    role_violation_count: int = Field(default=0, ge=0)


class FaultInjectionPlan(StrictModel):
    injection_id: Identifier
    target_agent_id: Identifier
    task_id: Identifier
    mode: FaultMode
    trigger_step: int = Field(ge=1)

    @model_validator(mode="after")
    def plan_requires_actual_fault(self) -> "FaultInjectionPlan":
        if self.mode is FaultMode.NONE:
            raise ValueError("fault injection plan cannot use NONE")
        return self


class FaultEvent(StrictModel):
    fault_event_id: Identifier
    target_agent_id: Identifier
    mode: FaultMode
    task_id: Identifier | None = None
    injected: bool = False
    observed_at_step: int = Field(ge=0)
    detail: ShortText

    @model_validator(mode="after")
    def event_requires_actual_fault(self) -> "FaultEvent":
        if self.mode is FaultMode.NONE:
            raise ValueError("fault event cannot use NONE")
        return self


class FaultObservation(StrictModel):
    injection_id: Identifier
    assignment_id: Identifier
    fault_event: FaultEvent
    raw_payload: dict[str, object] | None = None
    work_product: WorkProduct | None = None
    validation_error: ShortText | None = None

    @model_validator(mode="after")
    def observation_has_single_output_form(self) -> "FaultObservation":
        output_forms = sum(
            value is not None
            for value in (
                self.raw_payload,
                self.work_product,
                self.validation_error,
            )
        )
        if self.fault_event.mode in {FaultMode.OFFLINE, FaultMode.TIMEOUT}:
            if output_forms:
                raise ValueError("no-response faults cannot contain an output artifact")
            return self

        if self.fault_event.mode is FaultMode.MALFORMED_OUTPUT:
            if self.raw_payload is None or self.validation_error is None:
                raise ValueError(
                    "malformed output requires raw payload and validation error"
                )
            if self.work_product is not None:
                raise ValueError("malformed output cannot contain a valid work product")
            return self

        if self.work_product is None:
            raise ValueError("semantic fault modes require a structured work product")
        if self.raw_payload is not None or self.validation_error is not None:
            raise ValueError(
                "structured semantic faults cannot include schema-error artifacts"
            )
        return self


class ReliabilityAssessment(StrictModel):
    assessment_id: Identifier
    fault_event_id: Identifier
    agent_id: Identifier
    policy_version: ShortText
    health_before: AgentHealthState
    health_after: AgentHealthState
    trust_before: AgentTrustState
    trust_after: AgentTrustState
    health_reason: ShortText
    trust_reason: ShortText


class RecoveryEvent(StrictModel):
    recovery_event_id: Identifier
    trigger_fault_event_id: Identifier
    action: RecoveryAction
    task_id: Identifier | None = None
    affected_agent_ids: tuple[Identifier, ...] = Field(min_length=1)
    occurred_at_step: int = Field(ge=0)
    detail: ShortText


class ConsensusResult(StrictModel):
    consensus_id: Identifier
    mission_id: Identifier
    status: ConsensusStatus
    recommendation: MissionRecommendation | None = None
    supporting_work_product_ids: tuple[Identifier, ...] = ()
    dissenting_work_product_ids: tuple[Identifier, ...] = ()
    human_review_required: bool = False
    rationale: LongText

    @model_validator(mode="after")
    def consensus_state_is_consistent(self) -> "ConsensusResult":
        if self.status is ConsensusStatus.CONSENSUS_REACHED:
            if self.recommendation is None:
                raise ValueError("reached consensus requires a recommendation")
            if self.human_review_required:
                raise ValueError("reached consensus cannot require human review")
        elif self.recommendation is not None:
            raise ValueError("non-final consensus states cannot publish a recommendation")

        if self.status is ConsensusStatus.HUMAN_REVIEW_REQUIRED:
            if not self.human_review_required:
                raise ValueError("human-review status must set human_review_required")
        elif self.human_review_required:
            raise ValueError("human_review_required is only valid for human-review status")

        return self


class MissionState(StrictModel):
    mission_id: Identifier
    status: MissionStatus
    task_statuses: dict[Identifier, TaskStatus] = Field(default_factory=dict)
    quarantined_agent_ids: frozenset[Identifier] = frozenset()
    current_recommendation: MissionRecommendation | None = None
    current_step: int = Field(default=0, ge=0)


class AuditEvent(StrictModel):
    audit_event_id: Identifier
    mission_id: Identifier
    sequence: int = Field(ge=0)
    event_type: ShortText
    actor_id: Identifier | None = None
    subject_id: Identifier | None = None
    detail: LongText
