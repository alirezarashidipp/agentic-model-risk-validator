from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from app.schemas.jobs import JobStatus, UploadResponse, ValidationRequest, ValidationResponse
from app.services.job_service import create_job, job_dir, mark_uploaded, report_dir, save_upload
from app.services.validation_service import run_validation
from app.storage.db import get_job


router = APIRouter()


@router.get("/")
def root() -> dict:
    return {
        "service": "Agentic Model Risk Validator API",
        "status": "ok",
        "docs_url": "/docs",
        "health_url": "/health",
        "frontend_url": "http://127.0.0.1:8501",
    }


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    dataset: UploadFile | None = File(default=None),
    model: UploadFile | None = File(default=None),
    training_code: UploadFile | None = File(default=None),
    metrics_file: UploadFile | None = File(default=None),
    documentation: UploadFile | None = File(default=None),
) -> UploadResponse:
    job_id = create_job()
    uploaded_files: list[str] = []
    for field_name, upload in {
        "dataset": dataset,
        "model": model,
        "training_code": training_code,
        "metrics_file": metrics_file,
        "documentation": documentation,
    }.items():
        if upload is None:
            continue
        path = await save_upload(job_id, upload, field_name)
        uploaded_files.append(path.name)
    mark_uploaded(job_id)
    message = "Files uploaded." if uploaded_files else "Job created. Validation will use generated sample data unless files are uploaded."
    return UploadResponse(
        job_id=job_id,
        status=JobStatus.UPLOADED,
        uploaded_files=uploaded_files,
        message=message,
    )


@router.post("/validate", response_model=ValidationResponse)
def validate(request: ValidationRequest) -> ValidationResponse:
    try:
        return run_validation(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    timeline_path = job_dir(job_id) / "agent_timeline.json"
    timeline = []
    if timeline_path.exists():
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    return {"job": job.model_dump(mode="json"), "agent_timeline": timeline}


@router.get("/reports/{job_id}", response_class=PlainTextResponse)
def read_report(job_id: str) -> PlainTextResponse:
    path = report_dir(job_id) / "validation_report.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return PlainTextResponse(path.read_text(encoding="utf-8"))


@router.get("/reports/{job_id}/download")
def download_report(job_id: str, format: str = "md") -> FileResponse:
    reports = report_dir(job_id)
    preferred = reports / ("validation_report.pdf" if format.lower() == "pdf" else "validation_report.md")
    fallback = reports / "validation_report.md"
    path = preferred if preferred.exists() else fallback
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    media_type = "application/pdf" if path.suffix == ".pdf" else "text/markdown"
    return FileResponse(path, media_type=media_type, filename=Path(path).name)
