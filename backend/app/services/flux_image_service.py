import gc
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from backend.app.services.gpu_runtime import GPU_GENERATION_LOCK
from backend.app.services.hunyuan_v1 import CudaUnavailableError


class FluxImageService:
    MODEL_ID = os.environ.get(
        "TOOX_IMAGE_MODEL_ID",
        "black-forest-labs/FLUX.1-schnell",
    )

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        self.models_root = self._resolve_path(
            os.environ.get("TOOX_MODELS_DIR", "./models")
        )
        self.outputs_root = self._resolve_path(
            os.environ.get("TOOX_OUTPUTS_DIR", "./outputs")
        )
        self.hf_cache_dir = self.models_root / "huggingface"
        self._pipeline = None
        self._torch = None

    def generate(
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
        job_id: str | None = None,
        progress_callback=None,
    ) -> dict[str, Any]:
        started_at = time.time()
        self._report(progress_callback, 4, "Preparing image job")
        output_dir = self._create_output_dir(job_id=job_id)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        self._report(progress_callback, 18, "Loading image model")
        pipeline = self._ensure_pipeline()
        torch = self._ensure_torch()

        safe_width = self._normalize_dimension(width)
        safe_height = self._normalize_dimension(height)
        safe_num_images = max(1, min(4, num_images))
        safe_steps = max(1, num_inference_steps)

        generator_device = "cpu"
        generator = torch.Generator(generator_device).manual_seed(seed)

        self._report(progress_callback, 44, "Generating images")
        with GPU_GENERATION_LOCK:
            result = pipeline(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=safe_steps,
                width=safe_width,
                height=safe_height,
                num_images_per_prompt=safe_num_images,
                generator=generator,
                max_sequence_length=256,
            )

        saved_images: list[dict[str, Any]] = []
        self._report(progress_callback, 82, "Saving generated images")
        for index, image in enumerate(result.images):
            file_name = f"generated_{index:02d}.png"
            target_path = images_dir / file_name
            image.save(target_path)
            saved_images.append(
                {
                    "index": index,
                    "filename": file_name,
                    "path": str(target_path),
                    "url": f"/artifacts/{output_dir.name}/images/{file_name}",
                    "size_mb": round(target_path.stat().st_size / 1024**2, 2),
                    "width": image.width,
                    "height": image.height,
                }
            )

        metadata = {
            "job_id": output_dir.name,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_id": self.MODEL_ID,
            "num_images": len(saved_images),
            "width": safe_width,
            "height": safe_height,
            "seed": seed,
            "num_inference_steps": safe_steps,
            "guidance_scale": guidance_scale,
            "images": saved_images,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_seconds": round(time.time() - started_at, 2),
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        metadata["metadata_path"] = str(metadata_path)
        metadata["metadata_url"] = f"/artifacts/{output_dir.name}/metadata.json"
        self._report(progress_callback, 100, "Image generation completed")
        self._cleanup_gpu()
        return metadata

    def load_generated_image_bytes(
        self,
        *,
        job_id: str,
        image_index: int,
    ) -> tuple[str, bytes]:
        metadata_path = self.outputs_root / job_id / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError("Image job metadata not found.")

        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        images = metadata.get("images", [])
        if image_index < 0 or image_index >= len(images):
            raise FileNotFoundError("Requested generated image index does not exist.")

        selected = images[image_index]
        image_path = Path(selected["path"])
        if not image_path.exists():
            raise FileNotFoundError("Generated image file not found.")

        return selected["filename"], image_path.read_bytes()

    def _resolve_path(self, configured_path: str) -> Path:
        raw_path = Path(configured_path)
        if raw_path.is_absolute():
            return raw_path
        return (self.project_root / raw_path).resolve()

    def _normalize_dimension(self, value: int) -> int:
        clamped = max(256, min(1536, value))
        return int(round(clamped / 8) * 8)

    def _create_output_dir(self, *, job_id: str | None = None) -> Path:
        if job_id is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            suffix = uuid.uuid4().hex[:8]
            directory_name = f"img_{timestamp}_{suffix}"
        else:
            directory_name = "".join(
                char if char.isalnum() or char in "._-" else "_"
                for char in job_id
            )

        output_dir = self.outputs_root / directory_name
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir

    def _report(self, callback, percent: int, message: str) -> None:
        if callback is not None:
            callback(percent, message)

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        from diffusers import FluxPipeline

        torch = self._ensure_torch()
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(self.hf_cache_dir)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.hf_cache_dir / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(
            self.hf_cache_dir / "transformers"
        )

        if not torch.cuda.is_available():
            raise CudaUnavailableError("PyTorch did not detect CUDA for image generation.")

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        pipeline = FluxPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=dtype,
        )
        pipeline.enable_model_cpu_offload()
        self._pipeline = pipeline
        return self._pipeline

    def _ensure_torch(self):
        if self._torch is not None:
            return self._torch

        import torch

        self._torch = torch
        return self._torch

    def _cleanup_gpu(self) -> None:
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
            self._torch.cuda.ipc_collect()


service = FluxImageService()
