from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable
import uuid


@dataclass
class JobRecord:
    job_id: str
    client_id: str
    status: str
    created_at: str
    updated_at: str
    progress: float = 0.0
    step: str = "queued"
    message: str = ""
    work_dir: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def create_job(self, client_id: str, work_dir: Path) -> JobRecord:
        job_id = uuid.uuid4().hex
        now = self._now()
        record = JobRecord(
            job_id=job_id,
            client_id=client_id,
            status="queued",
            created_at=now,
            updated_at=now,
            work_dir=str(work_dir),
        )
        with self._lock:
            self._jobs[job_id] = record
        return record

    def submit(self, record: JobRecord, func: Callable[..., dict[str, Any]], *args: Any) -> None:
        self._executor.submit(self._run, record.job_id, func, *args)

    def _run(self, job_id: str, func: Callable[..., dict[str, Any]], *args: Any) -> None:
        self.update(job_id, status="running", step="starting", progress=0.01)
        try:
            result = func(job_id, self.update, *args)
            self.update(job_id, status="completed", step="completed", progress=1.0, result=result)
        except Exception as exc:
            self.update(job_id, status="failed", step="failed", error=str(exc), message=str(exc))

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = self._now()

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)