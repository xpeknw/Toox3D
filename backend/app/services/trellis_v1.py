import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from backend.app.services.hunyuan_v1 import (
    CudaUnavailableError,
    HunyuanV1Service,
    MeshGenerationError,
)


class TrellisV1Service(HunyuanV1Service):
    MODEL_ID = "microsoft/TRELLIS-image-large"
    MODEL_SUBFOLDER = None

    def __init__(self) -> None:
        super().__init__()
        self.vendor_repo_dir = self.models_root / "trellis-repo"
        self.hf_cache_dir = self.models_root / "huggingface_trellis"
        self.runtime_venv_dir = self._resolve_path(
            os.environ.get("TOOX_TRELLIS_VENV", "./models/trellis-venv")
        )
        self.worker_script = self.project_root / "backend" / "app" / "services" / "trellis_worker.py"

    def generate(
        self,
        filename: str,
        content: bytes,
        *,
        job_id: str | None = None,
        preset: str | None = None,
        octree_resolution: int = 384,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.5,
        seed: int = 1234,
        remove_background: bool = True,
        print_profile: str = "balanced",
        enable_postprocess: bool = True,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        import trimesh

        started_at = self._now_ts()
        self._report_progress(progress_callback, 2, "Preparing output directory")
        output_dir = self._create_output_dir(filename, job_id=job_id)
        safe_name = self._sanitize_filename(filename)
        input_suffix = Path(safe_name).suffix.lower() or ".png"
        input_path = output_dir / f"input{input_suffix}"

        with open(input_path, "wb") as handle:
            handle.write(content)

        image, processed_image_path, remove_background_applied = self._prepare_image(
            content=content,
            output_dir=output_dir,
            remove_background=remove_background,
            progress_callback=progress_callback,
        )
        del image

        self._report_progress(progress_callback, 20, "Checking TRELLIS runtime")
        self._ensure_pipeline()

        self._report_progress(progress_callback, 45, "Generating TRELLIS raw mesh")
        worker_result = self._run_worker_generate(
            image_path=processed_image_path,
            output_dir=output_dir,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        raw_stl_path = Path(worker_result["raw_stl_path"])
        if not raw_stl_path.exists():
            raise MeshGenerationError("TRELLIS worker finished without raw STL output.")

        self._report_progress(progress_callback, 80, "Loading and cleaning raw mesh")
        mesh = trimesh.load_mesh(str(raw_stl_path), file_type="stl")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [geometry for geometry in mesh.geometry.values()]
            )

        self._cleanup_mesh(mesh)
        raw_vertices = int(len(mesh.vertices))
        raw_faces = int(len(mesh.faces))
        raw_watertight = bool(mesh.is_watertight)

        self._report_progress(
            progress_callback,
            86,
            (
                "Optimizing mesh for 3D printing"
                if enable_postprocess
                else "Skipping print post-process"
            ),
        )
        print_mesh, print_profile_metadata = self._build_print_ready_mesh(
            mesh,
            print_profile=print_profile,
            enable_postprocess=enable_postprocess,
        )

        self._report_progress(progress_callback, 92, "Exporting mesh artifacts")
        exports = self._export_meshes(
            raw_mesh=mesh,
            print_mesh=print_mesh,
            base_name=Path(safe_name).stem,
            output_dir=output_dir,
        )

        metadata = {
            "job_id": output_dir.name,
            "engine": "trellis",
            "source_image_name": filename,
            "source_image_path": str(input_path),
            "processed_image": str(processed_image_path),
            "model_id": self.MODEL_ID,
            "model_subfolder": self.MODEL_SUBFOLDER,
            "preset": preset,
            "octree_resolution": octree_resolution,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "remove_background_requested": remove_background,
            "remove_background_applied": remove_background_applied,
            "enable_postprocess": enable_postprocess,
            "print_profile": print_profile_metadata["profile"],
            "vertices": int(len(print_mesh.vertices)),
            "faces": int(len(print_mesh.faces)),
            "watertight": bool(print_mesh.is_watertight),
            "raw_mesh": {
                "vertices": raw_vertices,
                "faces": raw_faces,
                "watertight": raw_watertight,
            },
            "print_mesh": print_profile_metadata,
            "processing_seconds": round(self._now_ts() - started_at, 2),
            "created_at": self._now(),
            "exports": exports,
            "download_urls": self._build_download_urls(output_dir, exports),
            "notes": {
                "octree_resolution": "ignored_by_trellis",
                "sampler_steps": worker_result.get("sampler_steps"),
                "sparse_cfg_strength": worker_result.get("sparse_cfg_strength"),
                "slat_cfg_strength": worker_result.get("slat_cfg_strength"),
                "runtime": "isolated_subprocess",
            },
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        metadata["metadata_path"] = str(metadata_path)
        metadata["metadata_url"] = f"/artifacts/{output_dir.name}/metadata.json"

        self._report_progress(progress_callback, 100, "Generation completed")
        del print_mesh
        del mesh
        self._cleanup_gpu()
        return metadata

    def _ensure_pipeline(self):
        python_bin = self._trellis_python()
        if not python_bin.exists():
            raise RuntimeError(
                "TRELLIS runtime is missing. Run bootstrap with --install-trellis."
            )
        if not self.vendor_repo_dir.exists():
            raise RuntimeError(
                "TRELLIS repository is missing. Run bootstrap with --install-trellis."
            )
        if not self.worker_script.exists():
            raise RuntimeError("TRELLIS worker script is missing from the project.")

        self._ensure_torch()
        torch = self._torch
        assert torch is not None
        if not torch.cuda.is_available():
            raise CudaUnavailableError("PyTorch did not detect CUDA.")

        return True

    def preload_runtime(self) -> None:
        self._ensure_pipeline()
        command = [
            str(self._trellis_python()),
            str(self.worker_script),
            "preload",
            "--repo-dir",
            str(self.vendor_repo_dir),
            "--hf-home",
            str(self.hf_cache_dir),
            "--model-id",
            self.MODEL_ID,
        ]
        self._run_worker_command(command)

    def _run_worker_generate(
        self,
        *,
        image_path: Path,
        output_dir: Path,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> dict[str, Any]:
        raw_stl_path = output_dir / "RAW" / "trellis_raw.stl"
        result_json_path = output_dir / "trellis_worker_result.json"
        command = [
            str(self._trellis_python()),
            str(self.worker_script),
            "generate",
            "--repo-dir",
            str(self.vendor_repo_dir),
            "--hf-home",
            str(self.hf_cache_dir),
            "--model-id",
            self.MODEL_ID,
            "--image-path",
            str(image_path),
            "--raw-stl-path",
            str(raw_stl_path),
            "--result-json-path",
            str(result_json_path),
            "--seed",
            str(seed),
            "--steps",
            str(max(1, int(num_inference_steps))),
            "--guidance-scale",
            str(float(guidance_scale)),
        ]
        self._run_worker_command(command)

        with open(result_json_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _run_worker_command(self, command: list[str]) -> None:
        env = dict(os.environ)
        env["HF_HOME"] = str(self.hf_cache_dir)
        process = subprocess.run(
            command,
            cwd=str(self.project_root),
            env=env,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            tail = (process.stdout or "") + "\n" + (process.stderr or "")
            raise RuntimeError(tail.strip() or "TRELLIS worker failed.")

    def _trellis_python(self) -> Path:
        return self.runtime_venv_dir / "bin" / "python"

    def _now_ts(self) -> float:
        import time

        return time.time()


service = TrellisV1Service()
