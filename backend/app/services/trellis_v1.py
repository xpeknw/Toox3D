import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from backend.app.services.gpu_runtime import GPU_GENERATION_LOCK
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

        self._report_progress(progress_callback, 20, "Loading TRELLIS pipeline")
        pipeline = self._ensure_pipeline()

        sparse_cfg_strength = max(1.0, float(guidance_scale))
        slat_cfg_strength = max(1.0, round(float(guidance_scale) * 0.5, 2))
        sampler_steps = max(1, int(num_inference_steps))

        self._report_progress(progress_callback, 45, "Generating 3D mesh")
        self._report_progress(
            progress_callback,
            62,
            "TRELLIS is sampling sparse structure and mesh geometry",
        )

        try:
            with GPU_GENERATION_LOCK:
                outputs = pipeline.run(
                    image,
                    seed=int(seed),
                    formats=["mesh", "gaussian"],
                    sparse_structure_sampler_params={
                        "steps": sampler_steps,
                        "cfg_strength": sparse_cfg_strength,
                    },
                    slat_sampler_params={
                        "steps": sampler_steps,
                        "cfg_strength": slat_cfg_strength,
                    },
                )
        except ValueError as exc:
            raise MeshGenerationError(
                "TRELLIS could not extract a valid mesh from this image."
            ) from exc

        mesh_items = outputs.get("mesh") if isinstance(outputs, dict) else None
        if not mesh_items:
            raise MeshGenerationError(
                "The TRELLIS pipeline returned no mesh for this image."
            )

        mesh = self._convert_trellis_mesh(mesh_items[0])
        self._report_progress(progress_callback, 80, "Cleaning raw mesh")
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
                "sampler_steps": sampler_steps,
                "sparse_cfg_strength": sparse_cfg_strength,
                "slat_cfg_strength": slat_cfg_strength,
            },
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            import json

            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        metadata["metadata_path"] = str(metadata_path)
        metadata["metadata_url"] = f"/artifacts/{output_dir.name}/metadata.json"

        self._report_progress(progress_callback, 100, "Generation completed")
        del print_mesh
        del mesh
        del image
        self._cleanup_gpu()
        return metadata

    def _ensure_pipeline(self):
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            self._ensure_torch()
            torch = self._torch
            assert torch is not None
            if not torch.cuda.is_available():
                raise CudaUnavailableError("PyTorch did not detect CUDA.")

            self._ensure_repo()
            self._ensure_repo_in_syspath()
            self._ensure_runtime_site_packages()
            os.environ["HF_HOME"] = str(self.hf_cache_dir)
            os.environ.setdefault("SPCONV_ALGO", "native")

            try:
                from trellis.pipelines import TrellisImageTo3DPipeline
            except Exception as exc:
                raise RuntimeError(
                    "TRELLIS is not installed in this environment yet. "
                    "Install its Python dependencies on the server before using this engine."
                ) from exc

            self._pipeline = TrellisImageTo3DPipeline.from_pretrained(self.MODEL_ID)
            self._pipeline.cuda()
            return self._pipeline

    def _ensure_repo(self) -> None:
        repo_ok = (
            self.vendor_repo_dir.is_dir()
            and (self.vendor_repo_dir / "trellis").is_dir()
            and (self.vendor_repo_dir / "setup.sh").is_file()
        )
        if repo_ok:
            return

        if self.vendor_repo_dir.exists():
            raise RuntimeError(
                "The TRELLIS repository directory exists but is incomplete."
            )

        self.vendor_repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--recurse-submodules",
                "https://github.com/microsoft/TRELLIS.git",
                str(self.vendor_repo_dir),
            ],
            check=True,
            cwd=str(self.vendor_repo_dir.parent),
        )

    def _ensure_repo_in_syspath(self) -> None:
        repo_path = str(self.vendor_repo_dir)
        if repo_path not in os.sys.path:
            os.sys.path.insert(0, repo_path)

    def _ensure_runtime_site_packages(self) -> None:
        lib_dir = self.runtime_venv_dir / "lib"
        if not lib_dir.is_dir():
            return

        for site_packages in sorted(lib_dir.glob("python*/site-packages")):
            site_packages_str = str(site_packages)
            if site_packages_str not in os.sys.path:
                os.sys.path.insert(0, site_packages_str)

    def _convert_trellis_mesh(self, mesh: Any) -> Any:
        import numpy as np
        import trimesh

        vertices = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)

        if vertices is None or faces is None:
            raise MeshGenerationError("TRELLIS returned an unsupported mesh object.")

        if hasattr(vertices, "detach"):
            vertices = vertices.detach().cpu().numpy()
        elif hasattr(vertices, "cpu"):
            vertices = vertices.cpu().numpy()

        if hasattr(faces, "detach"):
            faces = faces.detach().cpu().numpy()
        elif hasattr(faces, "cpu"):
            faces = faces.cpu().numpy()

        return trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=float),
            faces=np.asarray(faces, dtype=int),
            process=False,
        )

    def _now_ts(self) -> float:
        import time

        return time.time()


service = TrellisV1Service()
