from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    CREATED = "created"
    UPLOADED = "uploaded"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProblemType(str, Enum):
    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    target_column: str | None = None
    problem_type: str | None = None
    risk_rating: str | None = None
    final_decision: str | None = None
    error: str | None = None
    report_path: str | None = None


class UploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    uploaded_files: list[str] = Field(default_factory=list)
    message: str


class ValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str | None = None
    target_column: str | None = None
    problem_type: ProblemType = ProblemType.AUTO


class ValidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    risk_rating: str
    final_decision: str
    report_path: str
    report_markdown: str
    timeline_path: str
    findings: list[dict]


class StoredFileSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Path | None = None
    model: Path | None = None
    training_code: Path | None = None
    metrics_file: Path | None = None
    documentation: Path | None = None

