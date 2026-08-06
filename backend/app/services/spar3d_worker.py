import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


MODEL_ID = "stabilityai/stable-point-aware-3d"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPAR3D isolated worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preload_parser = subparsers.add_parser("preload")
    preload_parser.add_argument("--repo-dir", required=True)
    preload_parser.add_argument("--hf-home", required=True)
    preload_parser.add_argument("--hf-token", required=False, default="")

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--repo-dir", required=True)
    generate_parser.add_argument("--hf-home", required=True)
    generate_parser.add_argument("--hf-token", required=False, default="")
    generate_parser.add_argument("--image-path", required=True)
    generate_parser.add_argument("--output-dir", required=True)
    generate_parser.add_argument("--low-vram-mode", action="store_true")
    return parser


def ensure_runtime(repo_dir: str, hf_home: str, hf_token: str):
    os.environ["HF_HOME"] = hf_home
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    repo_path = Path(repo_dir)
    if not repo_path.exists():
        raise RuntimeError(f"SPAR3D repo directory does not exist: {repo_dir}")

    patch_transparent_background()
    patch_spar3d_run_py(repo_path)


def patch_transparent_background() -> None:
    import site

    patched_code = """from transparent_background.Remover import Remover

try:
    from transparent_background.gui import gui
except Exception:
    gui = None
"""

    for base in site.getsitepackages():
        init_file = Path(base) / "transparent_background" / "__init__.py"
        if init_file.exists():
            init_file.write_text(patched_code, encoding="utf-8")


def patch_spar3d_run_py(repo_path: Path) -> None:
    run_py = repo_path / "run.py"
    if not run_py.exists():
        return

    source = run_py.read_text(encoding="utf-8")
    marker = "# toox3d compatibility patch"
    if marker in source:
        return

    original = "args = parser.parse_args()"
    replacement = """args = parser.parse_args()
# toox3d compatibility patch
if not hasattr(args, "reduction_count_type"):
    args.reduction_count_type = "keep"
if not hasattr(args, "reduction_count"):
    args.reduction_count = -1
"""

    if original in source:
        source = source.replace(original, replacement, 1)
        run_py.write_text(source, encoding="utf-8")


def preload_model(repo_dir: str, hf_home: str, hf_token: str) -> None:
    ensure_runtime(repo_dir, hf_home, hf_token)
    if not hf_token:
        raise RuntimeError(
            "SPAR3D requires HF_TOKEN because the Stability model is gated."
        )

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_ID, cache_dir=hf_home, token=hf_token)
    print("[toox3d] SPAR3D model snapshot downloaded.")


def generate_glb(
    repo_dir: str,
    hf_home: str,
    hf_token: str,
    image_path: str,
    output_dir: str,
    low_vram_mode: bool,
) -> None:
    ensure_runtime(repo_dir, hf_home, hf_token)
    if not hf_token:
        raise RuntimeError(
            "SPAR3D requires HF_TOKEN because the Stability model is gated."
        )

    command = [
        sys.executable,
        "run.py",
        image_path,
        "--output-dir",
        output_dir,
        "--device",
        "cuda",
    ]
    if low_vram_mode:
        command.append("--low-vram-mode")

    process = subprocess.run(
        command,
        cwd=repo_dir,
        env=dict(os.environ),
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        combined = (process.stdout or "") + "\n" + (process.stderr or "")
        raise RuntimeError(combined.strip() or "SPAR3D run.py failed.")

    output_path = Path(output_dir)
    glb_candidates = sorted(
        output_path.glob("*.glb"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not glb_candidates:
        raise RuntimeError("SPAR3D completed without producing a GLB file.")

    result_payload = {
        "glb_path": str(glb_candidates[0]),
        "stdout_tail": (process.stdout or "")[-4000:],
    }
    result_path = output_path / "spar3d_worker_result.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preload":
        preload_model(args.repo_dir, args.hf_home, args.hf_token)
        return
    if args.command == "generate":
        generate_glb(
            repo_dir=args.repo_dir,
            hf_home=args.hf_home,
            hf_token=args.hf_token,
            image_path=args.image_path,
            output_dir=args.output_dir,
            low_vram_mode=args.low_vram_mode,
        )
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
