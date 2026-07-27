import gc
import io
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any


class CudaUnavailableError(RuntimeError):
    pass


class HunyuanV1Service:
    MODEL_ID = "tencent/Hunyuan3D-2mini"
    MODEL_SUBFOLDER = "hunyuan3d-dit-v2-mini-turbo"

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        self.models_root = self._resolve_path(
            os.environ.get("TOOX_MODELS_DIR", "./models")
        )
        self.outputs_root = self._resolve_path(
            os.environ.get("TOOX_OUTPUTS_DIR", "./outputs")
        )
        self.vendor_repo_dir = self.models_root / "hunyuan3d-2-repo"
        self.hf_cache_dir = self.models_root / "huggingface"

        self._lock = threading.Lock()
        self._pipeline = None
        self._background_remover = None
        self._torch = None
        self._image_module = None

    def generate(
        self,
        filename: str,
        content: bytes,
        *,
        octree_resolution: int = 384,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.5,
        seed: int = 1234,
        remove_background: bool = True,
    ) -> dict[str, Any]:
        started_at = time.time()
        output_dir = self._create_output_dir(filename)
        safe_name = self._sanitize_filename(filename)
        input_suffix = Path(safe_name).suffix.lower() or ".png"
        input_path = output_dir / f"input{input_suffix}"

        with open(input_path, "wb") as handle:
            handle.write(content)

        (
            image,
            processed_image_path,
            remove_background_applied,
        ) = self._prepare_image(
            content=content,
            output_dir=output_dir,
            remove_background=remove_background,
        )

        pipeline = self._ensure_pipeline()
        torch = self._torch
        assert torch is not None

        generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.inference_mode():
            result = pipeline(
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                octree_resolution=octree_resolution,
            )

        mesh = result[0]
        if mesh is None:
            raise RuntimeError("The Hunyuan pipeline returned no mesh.")

        self._cleanup_mesh(mesh)
        exports = self._export_mesh(mesh, Path(safe_name).stem, output_dir)

        metadata = {
            "job_id": output_dir.name,
            "source_image_name": filename,
            "source_image_path": str(input_path),
            "processed_image": str(processed_image_path),
            "model_id": self.MODEL_ID,
            "model_subfolder": self.MODEL_SUBFOLDER,
            "octree_resolution": octree_resolution,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "remove_background_requested": remove_background,
            "remove_background_applied": remove_background_applied,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "processing_seconds": round(time.time() - started_at, 2),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "exports": exports,
            "download_urls": self._build_download_urls(output_dir, exports),
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        metadata["metadata_path"] = str(metadata_path)
        metadata["metadata_url"] = (
            f"/artifacts/{output_dir.name}/metadata.json"
        )

        del mesh
        del image
        self._cleanup_gpu()
        return metadata

    def resolve_artifact_path(self, job_id: str, artifact_path: str) -> Path:
        job_dir = self.outputs_root / job_id
        target_path = (job_dir / artifact_path).resolve()

        if not str(target_path).startswith(str(job_dir.resolve())):
            raise ValueError("Artifact path escapes the job directory.")

        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError("Artifact not found.")

        return target_path

    def build_download_bundle(
        self,
        generation_result: dict[str, Any],
        *,
        include_all: bool,
    ) -> Path:
        job_id = generation_result["job_id"]
        output_dir = self.outputs_root / job_id
        bundle_name = "all" if include_all else "bundle"
        zip_path = output_dir / f"{job_id}_{bundle_name}.zip"

        files_to_include: list[tuple[Path, Path]] = []

        metadata_path = Path(generation_result["metadata_path"])
        files_to_include.append((metadata_path, Path("metadata.json")))

        stl_export = generation_result["exports"].get("stl")
        if stl_export and stl_export.get("ok"):
            stl_path = Path(stl_export["path"])
            files_to_include.append((stl_path, Path("STL") / stl_path.name))

        if include_all:
            for export_type in ("obj", "glb"):
                export_data = generation_result["exports"].get(export_type)
                if not export_data or not export_data.get("ok"):
                    continue

                export_path = Path(export_data["path"])
                files_to_include.append(
                    (
                        export_path,
                        Path(export_type.upper()) / export_path.name,
                    )
                )

            processed_image_path = Path(generation_result["processed_image"])
            files_to_include.append(
                (
                    processed_image_path,
                    Path("processed_image") / processed_image_path.name,
                )
            )

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
            for source_path, archive_path in files_to_include:
                if source_path.exists() and source_path.is_file():
                    zip_handle.write(source_path, archive_path.as_posix())

        return zip_path

    def build_download_bundle_for_job(
        self,
        job_id: str,
        *,
        include_all: bool,
    ) -> Path:
        job_dir = (self.outputs_root / job_id).resolve()
        if not job_dir.exists() or not job_dir.is_dir():
            raise FileNotFoundError("Job output directory not found.")

        if not str(job_dir).startswith(str(self.outputs_root.resolve())):
            raise ValueError("Job path escapes outputs directory.")

        metadata_path = job_dir / "metadata.json"
        if not metadata_path.exists() or not metadata_path.is_file():
            raise FileNotFoundError("Job metadata was not found.")

        with open(metadata_path, "r", encoding="utf-8") as handle:
            generation_result = json.load(handle)

        generation_result["job_id"] = job_id
        generation_result["metadata_path"] = str(metadata_path)
        generation_result.setdefault(
            "processed_image",
            str(job_dir / "processed_image" / "imagen_procesada.png"),
        )
        return self.build_download_bundle(
            generation_result,
            include_all=include_all,
        )

    def _resolve_path(self, configured_path: str) -> Path:
        raw_path = Path(configured_path)
        if raw_path.is_absolute():
            return raw_path
        return (self.project_root / raw_path).resolve()

    def _sanitize_filename(self, name: str) -> str:
        return "".join(
            char if char.isalnum() or char in "._-" else "_"
            for char in Path(name).name
        )

    def _create_output_dir(self, filename: str) -> Path:
        base_name = Path(self._sanitize_filename(filename)).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        output_dir = self.outputs_root / f"{base_name}_{timestamp}_{unique_id}"

        (output_dir / "STL").mkdir(parents=True, exist_ok=False)
        (output_dir / "OBJ").mkdir(parents=True, exist_ok=True)
        (output_dir / "GLB").mkdir(parents=True, exist_ok=True)
        (output_dir / "processed_image").mkdir(parents=True, exist_ok=True)
        return output_dir

    def _prepare_image(
        self,
        *,
        content: bytes,
        output_dir: Path,
        remove_background: bool,
    ) -> tuple[Any, Path, bool]:
        image_module = self._ensure_image_module()
        image = image_module.open(io.BytesIO(content))
        useful_alpha = self._detect_useful_alpha(image)
        image = image.convert("RGBA")

        remove_background_applied = False
        if remove_background and not useful_alpha:
            image = self._ensure_background_remover()(image)
            remove_background_applied = True
            self._cleanup_gpu()

        processed_image_path = (
            output_dir / "processed_image" / "imagen_procesada.png"
        )
        image.save(processed_image_path)
        return image, processed_image_path, remove_background_applied

    def _detect_useful_alpha(self, image: Any) -> bool:
        if image.mode not in ("RGBA", "LA"):
            return False

        alpha = image.getchannel("A")
        min_alpha, max_alpha = alpha.getextrema()
        return min_alpha < 250 and max_alpha > 0

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            torch = self._ensure_torch()
            self._ensure_repo()
            self._configure_environment()
            self._ensure_repo_in_syspath()
            self._assert_cuda_ready()

            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.MODEL_ID,
                subfolder=self.MODEL_SUBFOLDER,
                torch_dtype=torch.float16,
                device="cuda",
            )

        return self._pipeline

    def _assert_cuda_ready(self) -> None:
        torch = self._ensure_torch()

        try:
            if not torch.cuda.is_available():
                raise CudaUnavailableError(self._build_cuda_diagnostics())

            device_count = torch.cuda.device_count()
            if device_count < 1:
                raise CudaUnavailableError(self._build_cuda_diagnostics())

            torch.cuda.get_device_name(0)
        except CudaUnavailableError:
            raise
        except Exception as exc:
            diagnostics = self._build_cuda_diagnostics()
            raise CudaUnavailableError(
                f"CUDA is not usable on this server. {diagnostics}"
            ) from exc

    def _build_cuda_diagnostics(self) -> str:
        details: list[str] = [
            "PyTorch cannot use CUDA on this host."
        ]

        nvidia_smi_output = self._read_nvidia_smi_output()
        if nvidia_smi_output:
            details.append(f"nvidia-smi: {nvidia_smi_output}")

        details.append(
            "This usually means the VM GPU, NVIDIA driver, CUDA runtime, "
            "or container runtime are incompatible."
        )
        details.append(
            "The log 'forward compatibility was attempted on non supported HW' "
            "commonly points to a driver/CUDA stack mismatch on the server."
        )
        return " ".join(details)

    def _read_nvidia_smi_output(self) -> str:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,cuda_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return ""

        return " | ".join(
            line.strip() for line in result.stdout.splitlines() if line.strip()
        )

    def _ensure_repo(self) -> None:
        repo_ok = (
            (self.vendor_repo_dir / "setup.py").is_file()
            and (self.vendor_repo_dir / "requirements.txt").is_file()
            and (self.vendor_repo_dir / "hy3dgen").is_dir()
        )

        if repo_ok:
            return

        if self.vendor_repo_dir.exists():
            raise RuntimeError(
                "The Hunyuan repository directory exists but is incomplete."
            )

        self.vendor_repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git",
                str(self.vendor_repo_dir),
            ],
            check=True,
        )

    def _configure_environment(self) -> None:
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(self.hf_cache_dir)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.hf_cache_dir / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(
            self.hf_cache_dir / "transformers"
        )
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True,max_split_size_mb:128"
        )

    def _ensure_repo_in_syspath(self) -> None:
        repo_path = str(self.vendor_repo_dir)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    def _ensure_torch(self):
        if self._torch is not None:
            return self._torch

        import torch

        self._torch = torch
        return self._torch

    def _ensure_image_module(self):
        if self._image_module is not None:
            return self._image_module

        from PIL import Image

        self._image_module = Image
        return self._image_module

    def _ensure_background_remover(self):
        if self._background_remover is not None:
            return self._background_remover

        self._ensure_repo()
        self._ensure_repo_in_syspath()
        from hy3dgen.rembg import BackgroundRemover

        self._background_remover = BackgroundRemover()
        return self._background_remover

    def _cleanup_gpu(self) -> None:
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
            self._torch.cuda.ipc_collect()

    def _cleanup_mesh(self, mesh: Any) -> None:
        for action in (
            "remove_unreferenced_vertices",
            "merge_vertices",
            "remove_infinite_values",
        ):
            try:
                getattr(mesh, action)()
            except Exception:
                pass

    def _export_mesh(
        self,
        mesh: Any,
        base_name: str,
        output_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        export_plan = {
            "stl": output_dir / "STL" / f"{base_name}_Hunyuan3D.stl",
            "obj": output_dir / "OBJ" / f"{base_name}_Hunyuan3D.obj",
            "glb": output_dir / "GLB" / f"{base_name}_Hunyuan3D.glb",
        }
        export_results: dict[str, dict[str, Any]] = {}

        for export_type, path in export_plan.items():
            try:
                mesh.export(str(path), file_type=export_type)
                export_results[export_type] = {
                    "ok": True,
                    "path": str(path),
                    "size_mb": round(path.stat().st_size / 1024**2, 2),
                }
            except Exception as exc:
                export_results[export_type] = {
                    "ok": False,
                    "path": str(path),
                    "error": str(exc),
                }

        if not any(item["ok"] for item in export_results.values()):
            raise RuntimeError("All mesh export formats failed.")

        return export_results

    def _build_download_urls(
        self,
        output_dir: Path,
        exports: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        urls: dict[str, str] = {}
        for export_type, payload in exports.items():
            if not payload.get("ok"):
                continue

            file_path = Path(payload["path"])
            relative_path = file_path.relative_to(output_dir)
            urls[export_type] = (
                f"/artifacts/{output_dir.name}/{relative_path.as_posix()}"
            )

        processed_relative = Path("processed_image") / "imagen_procesada.png"
        urls["processed_image"] = (
            f"/artifacts/{output_dir.name}/{processed_relative.as_posix()}"
        )
        return urls


service = HunyuanV1Service()
