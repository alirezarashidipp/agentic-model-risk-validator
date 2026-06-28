from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RiskRating(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FinalDecision(str, Enum):
    APPROVED = "approved"
    APPROVED_WITH_CONDITIONS = "approved_with_conditions"
    REJECTED = "rejected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ValidationStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_name: str
    possible: bool
    rationale: str
    required_inputs: list[str] = Field(default_factory=list)


class EvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area: str
    missing_evidence: str
    impact: str


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    severity: RiskRating
    category: str
    evidence: str
    recommendation: str


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str = "PlannerAgent"
    summary: str
    possible_steps: list[ValidationStep] = Field(default_factory=list)
    missing_evidence: list[EvidenceGap] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class AgentAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    summary: str
    risk_rating: RiskRating
    findings: list[Finding] = Field(default_factory=list)
    challenge_questions: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class DocumentationReviewOutput(AgentAssessment):
    agent_name: str = "DocumentationReviewAgent"


class DataQualityReviewOutput(AgentAssessment):
    agent_name: str = "DataQualityAgent"


class MethodologyReviewOutput(AgentAssessment):
    agent_name: str = "MethodologyAgent"


class PerformanceReviewOutput(AgentAssessment):
    agent_name: str = "PerformanceAgent"


class ExplainabilityReviewOutput(AgentAssessment):
    agent_name: str = "ExplainabilityAgent"


class FairnessReviewOutput(AgentAssessment):
    agent_name: str = "FairnessAgent"


class GovernanceReviewOutput(AgentAssessment):
    agent_name: str = "GovernanceAgent"


class ReportWriterOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str = "ReportWriterAgent"
    executive_summary: str
    model_overview: str
    files_reviewed: str
    validation_scope: str
    data_quality_review: str
    methodology_review: str
    performance_review: str
    explainability_review: str
    bias_and_fairness_review: str
    governance_review: str
    key_findings: list[Finding] = Field(default_factory=list)
    challenge_questions: list[str] = Field(default_factory=list)
    risk_rating: RiskRating
    recommendations: list[str] = Field(default_factory=list)
    final_validation_decision: FinalDecision

