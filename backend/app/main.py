from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.app.services.hunyuan_v1 import (
    CudaUnavailableError,
    service as hunyuan_v1_service,
)


app = FastAPI(title="Toox 3D")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate-v1")
async def generate_v1(
    image: UploadFile = File(...),
    octree_resolution: int = Form(384),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(5.5),
    seed: int = Form(1234),
    remove_background: bool = Form(True),
) -> dict:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image filename is required.")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    try:
        return hunyuan_v1_service.generate(
            filename=image.filename,
            content=content,
            octree_resolution=octree_resolution,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            remove_background=remove_background,
        )
    except Exception as exc:
        if isinstance(exc, CudaUnavailableError):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
