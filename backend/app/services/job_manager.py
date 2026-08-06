import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.app.services.hunyuan_v1 import (
    CudaUnavailableError,
    HunyuanV1Service,
    service as hunyuan_v1_service,
)
from backend.app.services.spar3d_v1 import service as spar3d_v1_service
from backend.app.services.trellis_v1 import service as trellis_v1_service


JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLED = "cancelled"


@dataclass
class JobRecord:
    job_id: str
    filename: str
    status: str
    progress_percent: int
    progress_message: str
    created_at: str
    params: dict[str, Any]
    bundle_urls: dict[str, str] = field(default_factory=dict)
    artifact_urls: dict[str, str] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class HunyuanJobManager:
    def __init__(self, generator: HunyuanV1Service) -> None:
        self.generator = generator
        self.jobs_dir = self.generator.outputs_root / "_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.generators = {
            "hunyuan": hunyuan_v1_service,
            "trellis": trellis_v1_service,
            "spar3d": spar3d_v1_service,
        }

        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, JobRecord] = {}
        self._current_job_id: str | None = None
        self._load_existing_jobs()

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="toox3d-job-worker",
            daemon=True,
        )
        self._worker.start()

    def submit_job(
        self,
        *,
        filename: str,
        content: bytes,
        engine: str,
        preset: str,
        print_profile: str,
        enable_postprocess: bool,
        octree_resolution: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        remove_background: bool,
    ) -> JobRecord:
        job_id = self._build_job_id(filename)
        payload_path = self.jobs_dir / f"{job_id}.input"
        with open(payload_path, "wb") as handle:
            handle.write(content)

        record = JobRecord(
            job_id=job_id,
            filename=filename,
            status=JOB_STATUS_QUEUED,
            progress_percent=0,
            progress_message="Queued",
            created_at=self._now(),
            params={
                "engine": engine,
                "preset": preset,
                "print_profile": print_profile,
                "enable_postprocess": enable_postprocess,
                "octree_resolution": octree_resolution,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "remove_background": remove_background,
            },
            bundle_urls={
                "bundle": f"/downloads/{job_id}/bundle",
                "all": f"/downloads/{job_id}/all",
            },
        )

        with self._lock:
            self._jobs[job_id] = record
            self._persist_job(record)

        self._queue.put(job_id)
        return record

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs.get(job_id)

        if record is None:
            raise FileNotFoundError("Job not found.")

        return record

    def list_jobs(self, limit: int = 20) -> list[JobRecord]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise FileNotFoundError("Job not found.")
            if record.status != JOB_STATUS_QUEUED:
                raise ValueError("Only queued jobs can be cancelled.")

            record.status = JOB_STATUS_CANCELLED
            record.progress_message = "Cancelled"
            record.error = "Cancelled by user."
            record.completed_at = self._now()
            self._persist_job(record)

        payload_path = self.jobs_dir / f"{job_id}.input"
        payload_path.unlink(missing_ok=True)
        return record

    def delete_job(self, job_id: str, *, delete_outputs: bool = True) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise FileNotFoundError("Job not found.")
            if record.status == JOB_STATUS_RUNNING:
                raise ValueError("Running jobs cannot be deleted.")
            del self._jobs[job_id]

        (self.jobs_dir / f"{job_id}.json").unlink(missing_ok=True)
        (self.jobs_dir / f"{job_id}.input").unlink(missing_ok=True)

        if delete_outputs:
            output_dir = self.generator.outputs_root / job_id
            self._remove_tree(output_dir)

    def cleanup_jobs(
        self,
        *,
        older_than_hours: int | None = None,
        statuses: set[str] | None = None,
    ) -> dict[str, Any]:
        now_ts = time.time()
        deleted: list[str] = []

        with self._lock:
            candidates = list(self._jobs.values())

        for record in candidates:
            if statuses and record.status not in statuses:
                continue

            if older_than_hours is not None:
                reference = record.completed_at or record.created_at
                record_ts = self._parse_timestamp(reference)
                if record_ts is None:
                    continue
                age_hours = (now_ts - record_ts) / 3600
                if age_hours < older_than_hours:
                    continue

            try:
                self.delete_job(record.job_id, delete_outputs=True)
                deleted.append(record.job_id)
            except ValueError:
                continue

        return {"deleted_job_ids": deleted, "deleted_count": len(deleted)}

    def serialize_job(self, record: JobRecord) -> dict[str, Any]:
        payload = asdict(record)
        payload["status_url"] = f"/v2/jobs/{record.job_id}"
        payload["download_ready"] = record.status == JOB_STATUS_COMPLETED
        payload["effective_params"] = dict(record.params)
        payload["timing"] = self._build_timing_payload(record)
        payload["result_summary"] = self._build_result_summary(record)
        return payload

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        record = self.get_job(job_id)
        if record.status == JOB_STATUS_CANCELLED:
            return
        payload_path = self.jobs_dir / f"{job_id}.input"
        if not payload_path.exists():
            self._mark_failed(job_id, "Input payload is missing.")
            return

        with self._lock:
            self._current_job_id = job_id

        self._update_job(
            job_id,
            status=JOB_STATUS_RUNNING,
            progress_percent=1,
            progress_message="Starting job",
            started_at=self._now(),
        )

        try:
            with open(payload_path, "rb") as handle:
                content = handle.read()

            generator = self._resolve_generator(record.params.get("engine"))
            result = generator.generate(
                filename=record.filename,
                content=content,
                job_id=job_id,
                **self._build_generation_kwargs(record, job_id),
            )
        except CudaUnavailableError as exc:
            self._mark_failed(job_id, str(exc))
            return
        except Exception as exc:
            self._mark_failed(job_id, str(exc))
            return
        finally:
            payload_path.unlink(missing_ok=True)
            with self._lock:
                if self._current_job_id == job_id:
                    self._current_job_id = None

        self._update_job(
            job_id,
            status=JOB_STATUS_COMPLETED,
            progress_percent=100,
            progress_message="Completed",
            completed_at=self._now(),
            result=result,
            artifact_urls=result.get("download_urls", {}),
        )

    def _mark_failed(self, job_id: str, error: str) -> None:
        self._update_job(
            job_id,
            status=JOB_STATUS_FAILED,
            progress_message="Failed",
            error=error,
            completed_at=self._now(),
        )

    def _update_job(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._jobs[job_id]
            for key, value in changes.items():
                setattr(record, key, value)
            self._persist_job(record)

    def _persist_job(self, record: JobRecord) -> None:
        job_path = self.jobs_dir / f"{record.job_id}.json"
        with open(job_path, "w", encoding="utf-8") as handle:
            json.dump(asdict(record), handle, ensure_ascii=False, indent=2)

    def _load_existing_jobs(self) -> None:
        for job_path in sorted(self.jobs_dir.glob("*.json")):
            try:
                with open(job_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                record = JobRecord(**payload)
            except Exception:
                continue

            if record.status in {JOB_STATUS_QUEUED, JOB_STATUS_RUNNING}:
                record.status = JOB_STATUS_FAILED
                record.progress_message = "Interrupted by server restart"
                record.error = "The server restarted before this job completed."
                record.completed_at = self._now()

            self._jobs[record.job_id] = record
            self._persist_job(record)

    def _build_job_id(self, filename: str) -> str:
        base_name = Path(filename).stem or "job"
        safe_base = self.generator._sanitize_filename(base_name)[:40] or "job"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        return f"{safe_base}_{timestamp}_{suffix}"

    def _now(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def _build_timing_payload(self, record: JobRecord) -> dict[str, Any]:
        average_seconds = self._average_completed_job_seconds(
            preset=record.params.get("preset")
        )
        payload: dict[str, Any] = {
            "average_completed_job_seconds": average_seconds,
            "timing_basis_preset": record.params.get("preset"),
        }

        if record.status == JOB_STATUS_QUEUED:
            queue_position = self._queue_position(record.job_id)
            running_remaining = self._current_running_remaining_seconds()
            wait_seconds = max(0.0, running_remaining)
            if queue_position > 0:
                wait_seconds += max(0, queue_position - 1) * average_seconds

            payload.update(
                {
                    "queue_position": queue_position,
                    "estimated_wait_seconds": round(wait_seconds, 1),
                    "estimated_total_seconds": round(
                        wait_seconds + average_seconds,
                        1,
                    ),
                }
            )
        elif record.status == JOB_STATUS_RUNNING:
            payload["estimated_remaining_seconds"] = round(
                self._estimate_running_remaining_seconds(record),
                1,
            )
        elif record.status == JOB_STATUS_COMPLETED and record.result is not None:
            payload["actual_processing_seconds"] = record.result.get(
                "processing_seconds"
            )
        elif record.status in {JOB_STATUS_FAILED, JOB_STATUS_CANCELLED}:
            payload["estimated_remaining_seconds"] = 0.0

        return payload

    def _build_result_summary(self, record: JobRecord) -> dict[str, Any] | None:
        if record.result is None:
            return None

        result = record.result
        bundle_size_mb = self._file_size_mb(
            self.generator.outputs_root / record.job_id / f"{record.job_id}_bundle.zip"
        )
        all_bundle_size_mb = self._file_size_mb(
            self.generator.outputs_root / record.job_id / f"{record.job_id}_all.zip"
        )
        stl_export = result.get("exports", {}).get("stl", {})
        raw_stl_export = result.get("exports", {}).get("raw_stl", {})
        obj_export = result.get("exports", {}).get("obj", {})
        glb_export = result.get("exports", {}).get("glb", {})
        print_mesh = result.get("print_mesh", {})
        raw_mesh = result.get("raw_mesh", {})
        return {
            "engine": record.params.get("engine", "hunyuan"),
            "vertices": result.get("vertices"),
            "faces": result.get("faces"),
            "watertight": result.get("watertight"),
            "processing_seconds": result.get("processing_seconds"),
            "preset": record.params.get("preset"),
            "print_profile": record.params.get("print_profile"),
            "enable_postprocess": record.params.get("enable_postprocess"),
            "octree_resolution": record.params.get("octree_resolution"),
            "num_inference_steps": record.params.get("num_inference_steps"),
            "guidance_scale": record.params.get("guidance_scale"),
            "stl_size_mb": stl_export.get("size_mb"),
            "raw_stl_size_mb": raw_stl_export.get("size_mb"),
            "obj_size_mb": obj_export.get("size_mb"),
            "glb_size_mb": glb_export.get("size_mb"),
            "raw_faces": raw_mesh.get("faces"),
            "raw_vertices": raw_mesh.get("vertices"),
            "print_faces": print_mesh.get("faces"),
            "print_vertices": print_mesh.get("vertices"),
            "print_target_faces": print_mesh.get("target_faces"),
            "print_reduction_ratio": print_mesh.get("reduction_ratio"),
            "print_operations": print_mesh.get("operations"),
            "bundle_size_mb": bundle_size_mb,
            "all_bundle_size_mb": all_bundle_size_mb,
        }

    def _build_generation_kwargs(
        self,
        record: JobRecord,
        job_id: str,
    ) -> dict[str, Any]:
        return {
            "preset": record.params["preset"],
            "print_profile": record.params.get("print_profile", "balanced"),
            "enable_postprocess": record.params.get("enable_postprocess", True),
            "octree_resolution": record.params["octree_resolution"],
            "num_inference_steps": record.params["num_inference_steps"],
            "guidance_scale": record.params["guidance_scale"],
            "seed": record.params["seed"],
            "remove_background": record.params["remove_background"],
            "progress_callback": lambda percent, message: self._update_job(
                job_id,
                progress_percent=percent,
                progress_message=message,
            ),
        }

    def _resolve_generator(self, engine: str | None) -> HunyuanV1Service:
        normalized_engine = (engine or "hunyuan").strip().lower()
        generator = self.generators.get(normalized_engine)
        if generator is None:
            raise ValueError(f"Unsupported engine: {normalized_engine}")
        return generator

    def _average_completed_job_seconds(self, preset: str | None = None) -> float:
        completed_seconds: list[float] = []
        for record in self._jobs.values():
            if record.status != JOB_STATUS_COMPLETED or record.result is None:
                continue
            if preset is not None and record.params.get("preset") != preset:
                continue

            seconds = record.result.get("processing_seconds")
            if isinstance(seconds, (int, float)) and seconds > 0:
                completed_seconds.append(float(seconds))

        if not completed_seconds and preset is not None:
            return self._average_completed_job_seconds(preset=None)

        if not completed_seconds:
            return 120.0

        return round(sum(completed_seconds) / len(completed_seconds), 2)

    def _queue_position(self, job_id: str) -> int:
        queued_jobs = sorted(
            (
                record
                for record in self._jobs.values()
                if record.status == JOB_STATUS_QUEUED
            ),
            key=lambda item: item.created_at,
        )

        for index, record in enumerate(queued_jobs, start=1):
            if record.job_id == job_id:
                return index

        return 0

    def _current_running_remaining_seconds(self) -> float:
        with self._lock:
            current_job_id = self._current_job_id
            current_record = self._jobs.get(current_job_id) if current_job_id else None

        if current_record is None:
            return 0.0

        return self._estimate_running_remaining_seconds(current_record)

    def _estimate_running_remaining_seconds(self, record: JobRecord) -> float:
        average_seconds = self._average_completed_job_seconds()
        progress = max(1, min(record.progress_percent, 99))
        estimated_total = average_seconds
        completed_fraction = progress / 100.0
        estimated_elapsed = estimated_total * completed_fraction
        return max(0.0, estimated_total - estimated_elapsed)

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

    def _parse_timestamp(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            return time.mktime(time.strptime(value, "%Y-%m-%d %H:%M:%S"))
        except Exception:
            return None

    def _file_size_mb(self, path: Path) -> float | None:
        if not path.exists() or not path.is_file():
            return None
        return round(path.stat().st_size / 1024**2, 2)


manager = HunyuanJobManager(hunyuan_v1_service)
