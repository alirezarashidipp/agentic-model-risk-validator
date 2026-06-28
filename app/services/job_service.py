from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import settings
from app.schemas.jobs import JobStatus, StoredFileSet
from app.storage.db import create_job_record, update_job_record


FILE_FIELD_NAMES = {
    "dataset": {"csv", "xlsx", "xls", "json", "parquet"},
    "model": {"pkl", "pickle", "joblib"},
    "training_code": {"py", "ipynb", "txt", "md"},
    "metrics_file": {"json", "csv", "txt", "md"},
    "documentation": {"txt", "md", "pdf", "docx"},
}


def create_job() -> str:
    job_id = str(uuid4())
    for folder in ("jobs", "uploads", "reports"):
        (settings.storage_dir / folder / job_id).mkdir(parents=True, exist_ok=True)
    create_job_record(job_id)
    return job_id


def job_dir(job_id: str) -> Path:
    path = settings.storage_dir / "jobs" / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def upload_dir(job_id: str) -> Path:
    path = settings.storage_dir / "uploads" / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_dir(job_id: str) -> Path:
    path = settings.storage_dir / "reports" / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str, fallback: str) -> str:
    clean = Path(filename or fallback).name.replace(" ", "_")
    return clean or fallback


async def save_upload(job_id: str, upload: UploadFile, field_name: str) -> Path:
    suffix = Path(upload.filename or "").suffix
    destination = upload_dir(job_id) / safe_filename(
        upload.filename or f"{field_name}{suffix}", f"{field_name}{suffix}"
    )
    data = await upload.read()
    destination.write_bytes(data)
    return destination


def discover_uploaded_files(job_id: str) -> StoredFileSet:
    uploads = upload_dir(job_id)
    found: dict[str, Path | None] = {
        "dataset": None,
        "model": None,
        "training_code": None,
        "metrics_file": None,
        "documentation": None,
    }
    for path in uploads.iterdir():
        if not path.is_file():
            continue
        ext = path.suffix.lower().lstrip(".")
        for field, extensions in FILE_FIELD_NAMES.items():
            if found[field] is None and ext in extensions:
                found[field] = path
                break
    return StoredFileSet(**found)


def mark_uploaded(job_id: str) -> None:
    update_job_record(job_id, status=JobStatus.UPLOADED)

