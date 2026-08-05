import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRELLIS isolated worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preload_parser = subparsers.add_parser("preload")
    preload_parser.add_argument("--repo-dir", required=True)
    preload_parser.add_argument("--hf-home", required=True)
    preload_parser.add_argument("--model-id", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--repo-dir", required=True)
    generate_parser.add_argument("--hf-home", required=True)
    generate_parser.add_argument("--model-id", required=True)
    generate_parser.add_argument("--image-path", required=True)
    generate_parser.add_argument("--raw-stl-path", required=True)
    generate_parser.add_argument("--result-json-path", required=True)
    generate_parser.add_argument("--seed", type=int, required=True)
    generate_parser.add_argument("--steps", type=int, required=True)
    generate_parser.add_argument("--guidance-scale", type=float, required=True)
    return parser


def ensure_runtime(repo_dir: str, hf_home: str):
    import sys

    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("SPCONV_ALGO", "native")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def preload_pipeline(repo_dir: str, hf_home: str, model_id: str) -> None:
    ensure_runtime(repo_dir, hf_home)
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()
    print("[toox3d] TRELLIS pipeline loaded and cached.")
    del pipeline


def generate_mesh(
    repo_dir: str,
    hf_home: str,
    model_id: str,
    image_path: str,
    raw_stl_path: str,
    result_json_path: str,
    seed: int,
    steps: int,
    guidance_scale: float,
) -> None:
    ensure_runtime(repo_dir, hf_home)

    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    import numpy as np
    import trimesh

    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()

    image = Image.open(image_path).convert("RGBA")
    sparse_cfg_strength = max(1.0, float(guidance_scale))
    slat_cfg_strength = max(1.0, round(float(guidance_scale) * 0.5, 2))
    sampler_steps = max(1, int(steps))

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

    mesh_items = outputs.get("mesh") if isinstance(outputs, dict) else None
    if not mesh_items:
        raise RuntimeError("The TRELLIS pipeline returned no mesh for this image.")

    mesh = mesh_items[0]
    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    if vertices is None or faces is None:
        raise RuntimeError("TRELLIS returned an unsupported mesh object.")

    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    elif hasattr(vertices, "cpu"):
        vertices = vertices.cpu().numpy()
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()
    elif hasattr(faces, "cpu"):
        faces = faces.cpu().numpy()

    trimesh_mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=int),
        process=False,
    )

    raw_stl = Path(raw_stl_path)
    raw_stl.parent.mkdir(parents=True, exist_ok=True)
    trimesh_mesh.export(str(raw_stl), file_type="stl")

    result_payload = {
        "raw_stl_path": str(raw_stl),
        "vertices": int(len(trimesh_mesh.vertices)),
        "faces": int(len(trimesh_mesh.faces)),
        "watertight": bool(trimesh_mesh.is_watertight),
        "sampler_steps": sampler_steps,
        "sparse_cfg_strength": sparse_cfg_strength,
        "slat_cfg_strength": slat_cfg_strength,
    }
    result_path = Path(result_json_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "preload":
        preload_pipeline(args.repo_dir, args.hf_home, args.model_id)
        return

    if args.command == "generate":
        generate_mesh(
            repo_dir=args.repo_dir,
            hf_home=args.hf_home,
            model_id=args.model_id,
            image_path=args.image_path,
            raw_stl_path=args.raw_stl_path,
            result_json_path=args.result_json_path,
            seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
