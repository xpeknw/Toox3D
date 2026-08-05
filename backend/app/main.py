import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.services.job_manager import manager as job_manager
from backend.app.services.hunyuan_v1 import (
    CudaUnavailableError,
    MeshGenerationError,
    service as hunyuan_v1_service,
)


app = FastAPI(title="Toox 3D")

cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "TOOX_CORS_ORIGINS",
        "http://127.0.0.1:4200,http://localhost:4200",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GENERATION_PRESETS = {
    "v1-stable": {
        "octree_resolution": 384,
        "num_inference_steps": 30,
        "guidance_scale": 5.5,
        "description": "Stable Colab-equivalent baseline.",
    },
    "high": {
        "octree_resolution": 512,
        "num_inference_steps": 30,
        "guidance_scale": 5.5,
        "description": "Higher geometry detail with similar prompting behavior.",
    },
    "max": {
        "octree_resolution": 512,
        "num_inference_steps": 40,
        "guidance_scale": 5.5,
        "description": "Highest default quality profile currently recommended.",
    },
}

PRINT_PROFILES = {
    "safe": {
        "description": "Conservative decimation. Keeps more geometry for safer print fidelity.",
    },
    "balanced": {
        "description": "Recommended default. Strong polygon reduction with solid printable detail.",
    },
    "aggressive": {
        "description": "Maximum simplification for lighter STL files and faster slicing.",
    },
}

ENGINES = {
    "hunyuan": {
        "description": "Tencent Hunyuan3D mini turbo. Stable default path.",
    },
    "trellis": {
        "description": "Microsoft TRELLIS image-to-3D pipeline. Optional heavier engine.",
    },
}


def resolve_generation_params(
    *,
    engine: str,
    preset: str,
    octree_resolution: int | None,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    seed: int,
    remove_background: bool,
    print_profile: str,
    enable_postprocess: bool,
) -> dict:
    normalized_preset = preset.strip().lower()
    normalized_engine = engine.strip().lower()
    if normalized_engine not in ENGINES:
        raise HTTPException(
            status_code=400,
            detail=("engine must be one of: " + ", ".join(sorted(ENGINES))),
        )

    if normalized_preset not in GENERATION_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=(
                "preset must be one of: "
                + ", ".join(sorted(GENERATION_PRESETS))
            ),
        )

    preset_values = GENERATION_PRESETS[normalized_preset]

    if octree_resolution is not None and octree_resolution <= 0:
        octree_resolution = None

    if num_inference_steps is not None and num_inference_steps <= 0:
        num_inference_steps = None

    if guidance_scale is not None and guidance_scale <= 0:
        guidance_scale = None

    normalized_print_profile = print_profile.strip().lower()
    if normalized_print_profile not in PRINT_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=(
                "print_profile must be one of: "
                + ", ".join(sorted(PRINT_PROFILES))
            ),
        )

    return {
        "engine": normalized_engine,
        "preset": normalized_preset,
        "print_profile": normalized_print_profile,
        "octree_resolution": (
            octree_resolution
            if octree_resolution is not None
            else preset_values["octree_resolution"]
        ),
        "num_inference_steps": (
            num_inference_steps
            if num_inference_steps is not None
            else preset_values["num_inference_steps"]
        ),
        "guidance_scale": (
            guidance_scale
            if guidance_scale is not None
            else preset_values["guidance_scale"]
        ),
        "seed": seed,
        "remove_background": remove_background,
        "enable_postprocess": enable_postprocess,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v2/presets")
def list_v2_presets() -> dict[str, dict]:
    return {
        "engines": ENGINES,
        "presets": GENERATION_PRESETS,
        "print_profiles": PRINT_PROFILES,
    }


@app.post("/v2/jobs")
async def create_v2_job(
    image: UploadFile = File(...),
    engine: str = Form("hunyuan"),
    preset: str = Form("v1-stable"),
    octree_resolution: int | None = Form(None),
    num_inference_steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    seed: int = Form(1234),
    remove_background: bool = Form(True),
    print_profile: str = Form("balanced"),
    enable_postprocess: bool = Form(True),
) -> dict:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image filename is required.")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    resolved = resolve_generation_params(
        engine=engine,
        preset=preset,
        octree_resolution=octree_resolution,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        remove_background=remove_background,
        print_profile=print_profile,
        enable_postprocess=enable_postprocess,
    )

    record = job_manager.submit_job(
        filename=image.filename,
        content=content,
        **resolved,
    )
    return job_manager.serialize_job(record)


@app.get("/v2/jobs")
def list_v2_jobs(limit: int = 20) -> dict[str, list[dict]]:
    jobs = [
        job_manager.serialize_job(record)
        for record in job_manager.list_jobs(limit=limit)
    ]
    return {"jobs": jobs}


@app.get("/v2/jobs/{job_id}")
def get_v2_job(job_id: str) -> dict:
    try:
        record = job_manager.get_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return job_manager.serialize_job(record)


@app.post("/v2/jobs/{job_id}/cancel")
def cancel_v2_job(job_id: str) -> dict:
    try:
        record = job_manager.cancel_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return job_manager.serialize_job(record)


@app.delete("/v2/jobs/{job_id}")
def delete_v2_job(job_id: str) -> dict[str, str]:
    try:
        job_manager.delete_job(job_id, delete_outputs=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"status": "deleted", "job_id": job_id}


@app.post("/v2/jobs/cleanup")
def cleanup_v2_jobs(
    older_than_hours: int = Form(24),
    statuses: str = Form("completed,failed,cancelled"),
) -> dict[str, object]:
    normalized_statuses = {
        item.strip().lower()
        for item in statuses.split(",")
        if item.strip()
    }
    valid_statuses = {
        "completed",
        "failed",
        "cancelled",
        "queued",
    }
    invalid_statuses = sorted(normalized_statuses - valid_statuses)
    if invalid_statuses:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid statuses for cleanup: "
                + ", ".join(invalid_statuses)
            ),
        )

    return job_manager.cleanup_jobs(
        older_than_hours=older_than_hours,
        statuses=normalized_statuses,
    )


@app.post("/generate-v1", response_model=None)
async def generate_v1(
    image: UploadFile = File(...),
    preset: str = Form("v1-stable"),
    octree_resolution: int | None = Form(None),
    num_inference_steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    seed: int = Form(1234),
    remove_background: bool = Form(True),
    print_profile: str = Form("balanced"),
    enable_postprocess: bool = Form(True),
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image filename is required.")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    resolved = resolve_generation_params(
        engine="hunyuan",
        preset=preset,
        octree_resolution=octree_resolution,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        remove_background=remove_background,
        print_profile=print_profile,
        enable_postprocess=enable_postprocess,
    )

    try:
        result = hunyuan_v1_service.generate(
            filename=image.filename,
            content=content,
            **resolved,
        )
        job_id = result["job_id"]
        result["bundle_urls"] = {
            "bundle": f"/downloads/{job_id}/bundle",
            "all": f"/downloads/{job_id}/all",
        }
        return result
    except Exception as exc:
        if isinstance(exc, CudaUnavailableError):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(exc, MeshGenerationError):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/downloads/{job_id}/{bundle_kind}")
def download_bundle(job_id: str, bundle_kind: str) -> FileResponse:
    normalized_kind = bundle_kind.strip().lower()
    if normalized_kind not in {"bundle", "all"}:
        raise HTTPException(
            status_code=400,
            detail="bundle_kind must be one of: bundle, all.",
        )

    try:
        bundle_path = hunyuan_v1_service.build_download_bundle_for_job(
            job_id=job_id,
            include_all=normalized_kind == "all",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(
        path=bundle_path,
        filename=bundle_path.name,
        media_type="application/zip",
    )


@app.get("/artifacts/{job_id}/{artifact_path:path}")
def download_artifact(job_id: str, artifact_path: str) -> FileResponse:
    try:
        target_path = hunyuan_v1_service.resolve_artifact_path(
            job_id=job_id,
            artifact_path=artifact_path,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    media_type = None
    suffix = Path(target_path).suffix.lower()
    if suffix == ".json":
        media_type = "application/json"
    elif suffix == ".png":
        media_type = "image/png"
    elif suffix == ".stl":
        media_type = "model/stl"
    elif suffix == ".obj":
        media_type = "text/plain"
    elif suffix == ".glb":
        media_type = "model/gltf-binary"

    return FileResponse(
        path=target_path,
        filename=target_path.name,
        media_type=media_type,
    )
