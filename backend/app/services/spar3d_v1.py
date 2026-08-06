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


class Spar3DV1Service(HunyuanV1Service):
    MODEL_ID = "stabilityai/stable-point-aware-3d"
    MODEL_SUBFOLDER = None

    def __init__(self) -> None:
        super().__init__()
        self.vendor_repo_dir = self.models_root / "spar3d-repo"
        self.hf_cache_dir = self.models_root / "huggingface_spar3d"
        self.runtime_venv_dir = self._resolve_path(
            os.environ.get("TOOX_SPAR3D_VENV", "./models/spar3d-venv")
        )
        self.worker_script = (
            self.project_root / "backend" / "app" / "services" / "spar3d_worker.py"
        )

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
        del octree_resolution, num_inference_steps, guidance_scale, seed
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

        self._report_progress(progress_callback, 20, "Checking SPAR3D runtime")
        self._ensure_runtime()

        self._report_progress(progress_callback, 45, "Generating SPAR3D GLB")
        worker_result = self._run_worker_generate(
            image_path=processed_image_path,
            output_dir=output_dir,
        )

        glb_path = Path(worker_result["glb_path"])
        if not glb_path.exists():
            raise MeshGenerationError("SPAR3D finished without GLB output.")

        self._report_progress(progress_callback, 80, "Loading SPAR3D mesh")
        glb_scene = trimesh.load(str(glb_path), file_type="glb")
        if isinstance(glb_scene, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [geometry for geometry in glb_scene.geometry.values()]
            )
        else:
            mesh = glb_scene

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

        glb_export = exports.get("glb")
        if glb_export is not None:
            glb_export["path"] = str(glb_path)
            glb_export["size_mb"] = round(glb_path.stat().st_size / 1024**2, 2)
            glb_export["ok"] = True

        metadata = {
            "job_id": output_dir.name,
            "engine": "spar3d",
            "source_image_name": filename,
            "source_image_path": str(input_path),
            "processed_image": str(processed_image_path),
            "model_id": self.MODEL_ID,
            "model_subfolder": self.MODEL_SUBFOLDER,
            "preset": preset,
            "seed": 1234,
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
                "runtime": "isolated_subprocess",
                "gated_model": True,
                "low_vram_mode": self._low_vram_mode_enabled(),
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

    def preload_runtime(self) -> None:
        self._ensure_runtime()
        command = [
            str(self._spar3d_python()),
            str(self.worker_script),
            "preload",
            "--repo-dir",
            str(self.vendor_repo_dir),
            "--hf-home",
            str(self.hf_cache_dir),
            "--hf-token",
            self._hf_token(),
        ]
        self._run_worker_command(command)

    def _ensure_runtime(self) -> None:
        if not self._spar3d_python().exists():
            raise RuntimeError(
                "SPAR3D runtime is missing. Run bootstrap with --install-spar3d."
            )
        if not self.vendor_repo_dir.exists():
            raise RuntimeError(
                "SPAR3D repository is missing. Run bootstrap with --install-spar3d."
            )
        if not self.worker_script.exists():
            raise RuntimeError("SPAR3D worker script is missing from the project.")
        if not self._hf_token():
            raise RuntimeError(
                "SPAR3D requires HF_TOKEN or TOOX_HF_TOKEN because the Stability model is gated."
            )

        self._ensure_torch()
        torch = self._torch
        assert torch is not None
        if not torch.cuda.is_available():
            raise CudaUnavailableError("PyTorch did not detect CUDA.")

    def _run_worker_generate(
        self,
        *,
        image_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        command = [
            str(self._spar3d_python()),
            str(self.worker_script),
            "generate",
            "--repo-dir",
            str(self.vendor_repo_dir),
            "--hf-home",
            str(self.hf_cache_dir),
            "--hf-token",
            self._hf_token(),
            "--image-path",
            str(image_path),
            "--output-dir",
            str(output_dir / "spar3d_output"),
        ]
        if self._low_vram_mode_enabled():
            command.append("--low-vram-mode")

        self._run_worker_command(command)
        result_path = output_dir / "spar3d_output" / "spar3d_worker_result.json"
        with open(result_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _run_worker_command(self, command: list[str]) -> None:
        env = dict(os.environ)
        env["HF_HOME"] = str(self.hf_cache_dir)
        env["HF_TOKEN"] = self._hf_token()
        process = subprocess.run(
            command,
            cwd=str(self.project_root),
            env=env,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            tail = (process.stdout or "") + "\n" + (process.stderr or "")
            raise RuntimeError(tail.strip() or "SPAR3D worker failed.")

    def _hf_token(self) -> str:
        return (
            os.environ.get("TOOX_HF_TOKEN", "").strip()
            or os.environ.get("HF_TOKEN", "").strip()
        )

    def _low_vram_mode_enabled(self) -> bool:
        return os.environ.get("TOOX_SPAR3D_LOW_VRAM", "0").strip() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _spar3d_python(self) -> Path:
        return self.runtime_venv_dir / "bin" / "python"

    def _now_ts(self) -> float:
        import time

        return time.time()

    def _now(self) -> str:
        import time

        return time.strftime("%Y-%m-%d %H:%M:%S")


service = Spar3DV1Service()
