import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.app.services.flux_image_service import (
    FluxImageService,
    service as flux_image_service,
)


IMAGE_JOB_STATUS_QUEUED = "queued"
IMAGE_JOB_STATUS_RUNNING = "running"
IMAGE_JOB_STATUS_COMPLETED = "completed"
IMAGE_JOB_STATUS_FAILED = "failed"
IMAGE_JOB_STATUS_CANCELLED = "cancelled"


@dataclass
class ImageJobRecord:
    job_id: str
    status: str
    progress_percent: int
    progress_message: str
    created_at: str
    params: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    image_urls: list[str] = field(default_factory=list)


class ImageJobManager:
    def __init__(self, generator: FluxImageService) -> None:
        self.generator = generator
        self.jobs_dir = self.generator.outputs_root / "_image_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, ImageJobRecord] = {}
        self._load_existing_jobs()

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="toox3d-image-job-worker",
            daemon=True,
        )
        self._worker.start()

    def submit_job(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        num_images: int,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> ImageJobRecord:
        job_id = self._build_job_id()
        record = ImageJobRecord(
            job_id=job_id,
            status=IMAGE_JOB_STATUS_QUEUED,
            progress_percent=0,
            progress_message="Queued",
            created_at=self._now(),
            params={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_images": max(1, min(4, num_images)),
                "width": width,
                "height": height,
                "seed": seed,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )

        with self._lock:
            self._jobs[job_id] = record
            self._persist_job(record)

        self._queue.put(job_id)
        return record

    def get_job(self, job_id: str) -> ImageJobRecord:
        with self._lock:
            record = self._jobs.get(job_id)

        if record is None:
            raise FileNotFoundError("Image job not found.")
        return record

    def list_jobs(self, limit: int = 20) -> list[ImageJobRecord]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )
        return jobs[:limit]

    def delete_job(self, job_id: str, *, delete_outputs: bool = True) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise FileNotFoundError("Image job not found.")
            if record.status == IMAGE_JOB_STATUS_RUNNING:
                raise ValueError("Running image jobs cannot be deleted.")
            del self._jobs[job_id]

        (self.jobs_dir / f"{job_id}.json").unlink(missing_ok=True)
        if delete_outputs:
            output_dir = self.generator.outputs_root / job_id
            self._remove_tree(output_dir)

    def serialize_job(self, record: ImageJobRecord) -> dict[str, Any]:
        payload = asdict(record)
        payload["status_url"] = f"/v2/images/jobs/{record.job_id}"
        payload["download_ready"] = record.status == IMAGE_JOB_STATUS_COMPLETED
        return payload

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        self._update_job(
            job_id,
            status=IMAGE_JOB_STATUS_RUNNING,
            progress_percent=1,
            progress_message="Starting image job",
            started_at=self._now(),
        )
        record = self.get_job(job_id)

        try:
            result = self.generator.generate(
                job_id=job_id,
                prompt=record.params["prompt"],
                negative_prompt=record.params.get("negative_prompt"),
                num_images=record.params["num_images"],
                width=record.params["width"],
                height=record.params["height"],
                seed=record.params["seed"],
                num_inference_steps=record.params["num_inference_steps"],
                guidance_scale=record.params["guidance_scale"],
                progress_callback=lambda percent, message: self._update_job(
                    job_id,
                    progress_percent=percent,
                    progress_message=message,
                ),
            )
        except Exception as exc:
            self._update_job(
                job_id,
                status=IMAGE_JOB_STATUS_FAILED,
                progress_message="Failed",
                error=str(exc),
                completed_at=self._now(),
            )
            return

        self._update_job(
            job_id,
            status=IMAGE_JOB_STATUS_COMPLETED,
            progress_percent=100,
            progress_message="Completed",
            completed_at=self._now(),
            result=result,
            image_urls=[item["url"] for item in result.get("images", [])],
        )

    def _update_job(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._jobs[job_id]
            for key, value in changes.items():
                setattr(record, key, value)
            self._persist_job(record)

    def _persist_job(self, record: ImageJobRecord) -> None:
        target_path = self.jobs_dir / f"{record.job_id}.json"
        with open(target_path, "w", encoding="utf-8") as handle:
            json.dump(asdict(record), handle, ensure_ascii=False, indent=2)

    def _load_existing_jobs(self) -> None:
        for job_path in sorted(self.jobs_dir.glob("*.json")):
            try:
                with open(job_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                record = ImageJobRecord(**payload)
            except Exception:
                continue

            if record.status in {IMAGE_JOB_STATUS_QUEUED, IMAGE_JOB_STATUS_RUNNING}:
                record.status = IMAGE_JOB_STATUS_FAILED
                record.progress_message = "Interrupted by server restart"
                record.error = "The server restarted before this image job completed."
                record.completed_at = self._now()

            self._jobs[record.job_id] = record
            self._persist_job(record)

    def _build_job_id(self) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        return f"img_{timestamp}_{suffix}"

    def _remove_tree(self, path: Path) -> None:
        if not path.exists():
            return

        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        if path.is_dir():
            path.rmdir()

    def _now(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")


manager = ImageJobManager(flux_image_service)
