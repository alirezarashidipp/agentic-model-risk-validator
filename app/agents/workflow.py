from __future__ import annotations

from app.agents.fallbacks import (
    data_quality_fallback,
    documentation_fallback,
    explainability_fallback,
    fairness_fallback,
    governance_fallback,
    methodology_fallback,
    performance_fallback,
    planner_fallback,
    report_writer_fallback,
)
from app.agents.runner import AgentRunner, TimelineLogger, instructions_for
from app.schemas.agent_outputs import (
    DataQualityReviewOutput,
    DocumentationReviewOutput,
    ExplainabilityReviewOutput,
    FairnessReviewOutput,
    GovernanceReviewOutput,
    MethodologyReviewOutput,
    PerformanceReviewOutput,
    PlannerOutput,
    ReportWriterOutput,
)


def run_validation_agents(context: dict, timeline_path) -> dict:
    timeline = TimelineLogger(timeline_path)
    runner = AgentRunner(timeline)

    planner = runner.run(
        agent_name="PlannerAgent",
        instructions=instructions_for(
            "PlannerAgent",
            "Inspect available inputs, decide which validation steps are possible, and identify missing evidence.",
        ),
        context=context,
        output_model=PlannerOutput,
        fallback=planner_fallback,
    )

    agent_context = {**context, "plan": planner.model_dump(mode="json")}
    documentation = runner.run(
        agent_name="DocumentationReviewAgent",
        instructions=instructions_for(
            "DocumentationReviewAgent",
            "Review model documentation for purpose, assumptions, limitations, data, methodology, monitoring, governance, and approvals.",
        ),
        context=agent_context,
        output_model=DocumentationReviewOutput,
        fallback=documentation_fallback,
    )
    data_quality = runner.run(
        agent_name="DataQualityAgent",
        instructions=instructions_for(
            "DataQualityAgent",
            "Interpret deterministic data profiling, leakage, outlier, duplicate, missingness, and imbalance checks.",
        ),
        context=agent_context,
        output_model=DataQualityReviewOutput,
        fallback=data_quality_fallback,
    )
    methodology = runner.run(
        agent_name="MethodologyAgent",
        instructions=instructions_for(
            "MethodologyAgent",
            "Review model choice, baseline, feature engineering, hyperparameters, split design, and assumptions.",
        ),
        context=agent_context,
        output_model=MethodologyReviewOutput,
        fallback=methodology_fallback,
    )
    performance = runner.run(
        agent_name="PerformanceAgent",
        instructions=instructions_for(
            "PerformanceAgent",
            "Interpret deterministic classification or regression metrics and overfitting checks.",
        ),
        context=agent_context,
        output_model=PerformanceReviewOutput,
        fallback=performance_fallback,
    )
    explainability = runner.run(
        agent_name="ExplainabilityAgent",
        instructions=instructions_for(
            "ExplainabilityAgent",
            "Interpret feature importance and SHAP artifacts and identify suspicious model drivers.",
        ),
        context=agent_context,
        output_model=ExplainabilityReviewOutput,
        fallback=explainability_fallback,
    )
    fairness = runner.run(
        agent_name="FairnessAgent",
        instructions=instructions_for(
            "FairnessAgent",
            "Assess whether fairness analysis is possible and interpret deterministic group metrics.",
        ),
        context=agent_context,
        output_model=FairnessReviewOutput,
        fallback=fairness_fallback,
    )
    governance = runner.run(
        agent_name="GovernanceAgent",
        instructions=instructions_for(
            "GovernanceAgent",
            "Review ownership, versioning, approval, monitoring, retraining, lineage, and human oversight evidence.",
        ),
        context=agent_context,
        output_model=GovernanceReviewOutput,
        fallback=governance_fallback,
    )

    assessments = [
        documentation,
        data_quality,
        methodology,
        performance,
        explainability,
        fairness,
        governance,
    ]
    report_context = {**context, "agent_assessments": assessments}
    report_writer = runner.run(
        agent_name="ReportWriterAgent",
        instructions=instructions_for(
            "ReportWriterAgent",
            "Combine all agent findings into the final model validation report using the supplied policy decision.",
        ),
        context=report_context,
        output_model=ReportWriterOutput,
        fallback=report_writer_fallback,
    )

    return {
        "planner": planner,
        "documentation": documentation,
        "data_quality": data_quality,
        "methodology": methodology,
        "performance": performance,
        "explainability": explainability,
        "fairness": fairness,
        "governance": governance,
        "report_writer": report_writer,
    }

