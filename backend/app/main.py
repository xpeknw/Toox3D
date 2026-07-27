from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.app.services.job_manager import manager as job_manager
from backend.app.services.hunyuan_v1 import (
    CudaUnavailableError,
    MeshGenerationError,
    service as hunyuan_v1_service,
)


app = FastAPI(title="Toox 3D")

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


def resolve_generation_params(
    *,
    preset: str,
    octree_resolution: int | None,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    seed: int,
    remove_background: bool,
) -> dict:
    normalized_preset = preset.strip().lower()
    if normalized_preset not in GENERATION_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=(
                "preset must be one of: "
                + ", ".join(sorted(GENERATION_PRESETS))
            ),
        )

    preset_values = GENERATION_PRESETS[normalized_preset]
    return {
        "preset": normalized_preset,
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
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v2/presets")
def list_v2_presets() -> dict[str, dict]:
    return {"presets": GENERATION_PRESETS}


@app.post("/v2/jobs")
async def create_v2_job(
    image: UploadFile = File(...),
    preset: str = Form("v1-stable"),
    octree_resolution: int | None = Form(None),
    num_inference_steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    seed: int = Form(1234),
    remove_background: bool = Form(True),
) -> dict:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image filename is required.")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    resolved = resolve_generation_params(
        preset=preset,
        octree_resolution=octree_resolution,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        remove_background=remove_background,
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


@app.post("/generate-v1", response_model=None)
async def generate_v1(
    image: UploadFile = File(...),
    preset: str = Form("v1-stable"),
    octree_resolution: int | None = Form(None),
    num_inference_steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    seed: int = Form(1234),
    remove_background: bool = Form(True),
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image filename is required.")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    resolved = resolve_generation_params(
        preset=preset,
        octree_resolution=octree_resolution,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        remove_background=remove_background,
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
