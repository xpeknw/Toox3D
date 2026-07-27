"""
# ============================================================
# HUNYUAN3D STUDIO — IMAGEN A STL / OBJ / GLB
# GOOGLE COLAB — UNA SOLA CELDA
#
# Objetivo:
# - Mantener la estabilidad práctica de la V1.
# - Cargar el modelo una sola vez por sesión.
# - Permitir varias generaciones sin rerun de celda.
# - Persistir repo, caché y resultados en Google Drive.
#
# Recomendado:
# Entorno de ejecución -> Cambiar tipo -> GPU T4
# ============================================================
"""

import gc
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ============================================================
# CONFIGURACION
# ============================================================

DRIVE_DIR = Path("/content/drive/MyDrive/IA_Modelos/Hunyuan3D-2")
REPO_DIR = DRIVE_DIR / "repo"
HF_CACHE_DIR = DRIVE_DIR / "modelos_huggingface"
RESULTS_DIR = DRIVE_DIR / "resultados"

RUNTIME_DIR = Path("/content/hunyuan3d_runtime")
INPUT_DIR = RUNTIME_DIR / "input"

# Modelo recomendado por la V1 estable.
MODEL_ID = "tencent/Hunyuan3D-2mini"
MODEL_SUBFOLDER = "hunyuan3d-dit-v2-mini-turbo"

# Calidad objetivo definida en AGENTS.md.
OCTREE_RESOLUTION = 384
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 5.5

SEED = 1234
REMOVE_BACKGROUND = True
AUTO_DOWNLOAD = False

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
STATE_KEY = "__HUNYUAN3D_STUDIO_STATE__"

SYSTEM_PACKAGES = [
    "libgl1",
    "libglib2.0-0",
    "libegl1",
    "ninja-build",
]

PYTHON_PACKAGES = [
    "pip",
    "setuptools",
    "wheel",
    "huggingface_hub==0.34.4",
    "diffusers>=0.30.0,<0.36",
    "transformers>=4.44.0,<5",
    "accelerate>=0.33.0",
    "safetensors",
    "einops",
    "omegaconf",
    "opencv-python-headless",
    "numpy<2.1",
    "tqdm",
    "trimesh>=4.4.0",
    "pymeshlab",
    "pygltflib",
    "xatlas",
    "ninja",
    "pybind11",
    "rembg",
    "onnxruntime",
    "ipywidgets",
]


# ============================================================
# ESTADO DE SESION
# ============================================================


@dataclass
class SessionState:
    bootstrapped: bool = False
    ui_rendered: bool = False
    drive_mounted: bool = False
    repo_ready: bool = False
    dependencies_ready: bool = False
    pipeline_ready: bool = False
    busy: bool = False
    lock: Any = field(default=None, repr=False)
    pipeline: Any = None
    background_remover: Any = None
    torch: Any = None
    Image: Any = None
    widgets: Any = None
    display: Any = None
    clear_output: Any = None
    files: Any = None
    output_widget: Any = None
    preview_widget: Any = None
    downloads_box: Any = None
    metadata_box: Any = None
    status_html: Any = None
    file_upload: Any = None
    generate_button: Any = None
    image_bytes: Optional[bytes] = None
    image_name: Optional[str] = None
    last_result: Optional[dict[str, Any]] = None


if STATE_KEY not in globals():
    globals()[STATE_KEY] = SessionState(lock=threading.Lock())

STATE = globals()[STATE_KEY]


# ============================================================
# UTILIDADES
# ============================================================


def log_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_command(command, cwd=None, env=None) -> None:
    if isinstance(command, (list, tuple)):
        display_command = " ".join(str(part) for part in command)
        shell = False
    else:
        display_command = command
        shell = True

    print(f"\n▶ {display_command}\n", flush=True)

    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)

    process.wait()

    if process.returncode != 0:
        tail = "".join(output_lines[-80:])
        raise RuntimeError(
            f"\nFallo el comando con codigo {process.returncode}:\n"
            f"{display_command}\n\n"
            f"ULTIMA SALIDA DEL PROCESO:\n{tail}"
        )


def is_valid_file(path: Path, min_bytes: int = 1) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def sanitize_filename(name: str) -> str:
    return "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in Path(name).name
    )


def cleanup_gpu(preserve_pipeline: bool = True) -> None:
    del preserve_pipeline
    gc.collect()
    if STATE.torch is not None and STATE.torch.cuda.is_available():
        STATE.torch.cuda.empty_cache()
        STATE.torch.cuda.ipc_collect()


def update_status(message: str, tone: str = "info") -> None:
    if STATE.status_html is None:
        return

    colors = {
        "info": "#0f4c81",
        "success": "#1b5e20",
        "warning": "#8d6e00",
        "error": "#b71c1c",
    }
    color = colors.get(tone, colors["info"])
    STATE.status_html.value = (
        "<div style='padding:10px 12px;border-radius:10px;"
        f"background:#f5f7fa;color:{color};font-weight:600'>"
        f"{message}</div>"
    )


def ensure_colab_environment():
    if "google.colab" not in sys.modules:
        try:
            from google.colab import drive, files  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Este notebook esta pensado para Google Colab."
            ) from exc
    else:
        from google.colab import drive, files  # type: ignore

    STATE.files = files
    return drive


def mount_drive_once() -> None:
    if STATE.drive_mounted:
        print("\n✅ Google Drive ya estaba montado en esta sesion.")
        return

    log_section("MONTANDO GOOGLE DRIVE")
    drive = ensure_colab_environment()
    drive.mount("/content/drive", force_remount=False)
    STATE.drive_mounted = True


def prepare_directories() -> None:
    log_section("PREPARANDO DIRECTORIOS")

    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n✅ Directorio persistente:\n{DRIVE_DIR}")
    print(f"\n✅ Cache HuggingFace:\n{HF_CACHE_DIR}")
    print(f"\n✅ Resultados:\n{RESULTS_DIR}")


def assert_gpu_ready() -> None:
    log_section("COMPROBANDO GPU")
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        print(f"\n✅ GPU detectada: {output}")
    except Exception as exc:
        raise RuntimeError(
            "No se detecto una GPU NVIDIA.\n\n"
            "Activa:\n"
            "Entorno de ejecucion -> Cambiar tipo de entorno -> T4 GPU"
        ) from exc


def ensure_repo() -> None:
    if STATE.repo_ready:
        print("\n✅ Repositorio ya verificado en esta sesion.")
        return

    log_section("COMPROBANDO REPOSITORIO HUNYUAN3D-2")

    repo_ok = (
        is_valid_file(REPO_DIR / "setup.py", 100)
        and is_valid_file(REPO_DIR / "requirements.txt", 100)
        and (REPO_DIR / "hy3dgen").is_dir()
    )

    if repo_ok:
        print(f"\n✅ Repositorio encontrado:\n{REPO_DIR}")
        print("✅ Se omite la descarga.")
        STATE.repo_ready = True
        return

    print("\n⬇ El repositorio falta o esta incompleto.")

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)

    run_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git",
            str(REPO_DIR),
        ]
    )

    if not (REPO_DIR / "hy3dgen").is_dir():
        raise RuntimeError("El repositorio se descargo, pero falta hy3dgen.")

    print("\n✅ Repositorio guardado en Drive.")
    STATE.repo_ready = True


def ensure_system_packages() -> None:
    log_section("INSTALANDO DEPENDENCIAS DEL SISTEMA")
    run_command("apt-get update -qq")
    run_command(["apt-get", "install", "-y", "-qq", *SYSTEM_PACKAGES])


def ensure_python_packages() -> None:
    log_section("INSTALANDO DEPENDENCIAS DE PYTHON")

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--force-reinstall",
            "huggingface_hub==0.34.4",
        ]
    )

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            *[
                package
                for package in PYTHON_PACKAGES
                if package != "huggingface_hub==0.34.4"
            ],
        ]
    )

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--editable",
            str(REPO_DIR),
            "--no-deps",
        ]
    )

    # Colab a veces queda con una instalacion mezclada de PIL/Pillow
    # tras upgrades sucesivos. Forzamos una reinstalacion limpia sin cache
    # y verificamos que los modulos internos queden alineados.
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            "Pillow",
        ]
    )

    run_command(
        [
            sys.executable,
            "-c",
            (
                "from PIL import Image, ImageText; "
                "from PIL._typing import _Ink; "
                "print(Image.__version__)"
            ),
        ]
    )


def configure_environment() -> None:
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,max_split_size_mb:128"
    )

    pythonpath = os.environ.get("PYTHONPATH", "")
    repo_path = str(REPO_DIR)
    if repo_path not in pythonpath.split(":"):
        os.environ["PYTHONPATH"] = f"{repo_path}:{pythonpath}" if pythonpath else repo_path

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def import_runtime_modules() -> None:
    if STATE.torch is not None and STATE.widgets is not None:
        return

    # Si PIL ya fue importado antes de reinstalar Pillow, Colab puede
    # conservar submodulos viejos en memoria y mezclarlos con archivos
    # nuevos en disco. Eso produce errores como:
    # ImportError: cannot import name '_Ink' from 'PIL._typing'
    pil_modules = [
        module_name
        for module_name in list(sys.modules)
        if module_name == "PIL" or module_name.startswith("PIL.")
    ]
    for module_name in pil_modules:
        del sys.modules[module_name]

    import torch
    import ipywidgets as widgets
    from IPython.display import clear_output, display
    from PIL import Image

    STATE.torch = torch
    STATE.widgets = widgets
    STATE.display = display
    STATE.clear_output = clear_output
    STATE.Image = Image


def ensure_dependencies() -> None:
    if STATE.dependencies_ready:
        print("\n✅ Dependencias ya preparadas en esta sesion.")
        return

    ensure_system_packages()
    ensure_python_packages()
    configure_environment()
    import_runtime_modules()

    STATE.dependencies_ready = True


def ensure_background_remover():
    if STATE.background_remover is not None:
        return STATE.background_remover

    from hy3dgen.rembg import BackgroundRemover

    STATE.background_remover = BackgroundRemover()
    return STATE.background_remover


def ensure_pipeline_loaded() -> None:
    if STATE.pipeline_ready and STATE.pipeline is not None:
        print("\n✅ El modelo ya estaba cargado en memoria.")
        return

    log_section("CARGANDO HUNYUAN3D-2")

    if not STATE.torch.cuda.is_available():
        raise RuntimeError("PyTorch no detecto CUDA dentro de la sesion.")

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    print(f"\nGPU: {STATE.torch.cuda.get_device_name(0)}")
    print(f"Modelo: {MODEL_ID}")
    print(f"Subfolder: {MODEL_SUBFOLDER}")
    print(f"Octree resolution por defecto: {OCTREE_RESOLUTION}")
    print(f"Inference steps por defecto: {NUM_INFERENCE_STEPS}")
    print(f"Guidance scale por defecto: {GUIDANCE_SCALE}")

    STATE.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        MODEL_ID,
        subfolder=MODEL_SUBFOLDER,
        torch_dtype=STATE.torch.float16,
        device="cuda",
    )

    STATE.pipeline_ready = True
    print("\n✅ Modelo cargado y retenido en memoria para toda la sesion.")


def ensure_bootstrap() -> None:
    if STATE.bootstrapped:
        print("\n✅ Bootstrap ya completado en esta sesion.")
        return

    mount_drive_once()
    prepare_directories()
    assert_gpu_ready()
    ensure_repo()
    ensure_dependencies()
    ensure_pipeline_loaded()

    STATE.bootstrapped = True


def unpack_file_upload_value(value) -> tuple[str, bytes]:
    if not value:
        raise RuntimeError("No seleccionaste ninguna imagen.")

    if isinstance(value, dict):
        first_key = next(iter(value))
        uploaded = value[first_key]
        if isinstance(uploaded, dict):
            content = uploaded.get("content")
            name = uploaded.get("name", first_key)
        else:
            content = getattr(uploaded, "content", None)
            name = getattr(uploaded, "name", first_key)
    else:
        uploaded = value[0]
        if isinstance(uploaded, dict):
            content = uploaded.get("content")
            name = uploaded.get("name", "imagen")
        else:
            content = getattr(uploaded, "content", None)
            name = getattr(uploaded, "name", "imagen")

    if content is None:
        raise RuntimeError("No fue posible leer el contenido del archivo subido.")

    if isinstance(content, memoryview):
        content = content.tobytes()
    elif isinstance(content, bytearray):
        content = bytes(content)

    return str(name), content


def create_result_directory(safe_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = RESULTS_DIR / f"{Path(safe_name).stem}_{timestamp}"
    suffix = 1

    while base_dir.exists():
        suffix += 1
        base_dir = RESULTS_DIR / f"{Path(safe_name).stem}_{timestamp}_{suffix:02d}"

    (base_dir / "STL").mkdir(parents=True, exist_ok=False)
    (base_dir / "OBJ").mkdir(parents=True, exist_ok=True)
    (base_dir / "GLB").mkdir(parents=True, exist_ok=True)
    (base_dir / "processed_image").mkdir(parents=True, exist_ok=True)
    return base_dir


def detect_useful_alpha(image) -> bool:
    if image.mode not in ("RGBA", "LA"):
        return False

    alpha = image.getchannel("A")
    min_alpha, max_alpha = alpha.getextrema()
    return min_alpha < 250 and max_alpha > 0


def prepare_input_image(image_name: str, image_bytes: bytes, output_dir: Path):
    safe_name = sanitize_filename(image_name)
    extension = Path(safe_name).suffix.lower()

    if extension not in VALID_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Formato no permitido: {extension}\n"
            "Usa PNG, JPG, JPEG, WEBP o BMP."
        )

    input_path = INPUT_DIR / safe_name
    with open(input_path, "wb") as handle:
        handle.write(image_bytes)

    image = STATE.Image.open(io.BytesIO(image_bytes))
    useful_alpha = detect_useful_alpha(image)
    image = image.convert("RGBA")

    remove_background_applied = False
    if REMOVE_BACKGROUND and not useful_alpha:
        print("\nEliminando fondo automaticamente...")
        remover = ensure_background_remover()
        image = remover(image)
        remove_background_applied = True
        cleanup_gpu()
    elif useful_alpha:
        print("\nLa imagen ya contiene transparencia util.")
        print("Se conserva el fondo transparente original.")
    else:
        print("\nEliminacion de fondo desactivada.")

    processed_image_path = (
        output_dir / "processed_image" / f"{Path(safe_name).stem}_processed.png"
    )
    image.save(processed_image_path)
    print(f"\n✅ Imagen procesada:\n{processed_image_path}")

    return safe_name, input_path, image, processed_image_path, remove_background_applied


def cleanup_mesh(mesh) -> None:
    for action in (
        "remove_unreferenced_vertices",
        "merge_vertices",
        "remove_infinite_values",
    ):
        try:
            getattr(mesh, action)()
        except Exception:
            pass


def export_mesh(mesh, base_name: str, output_dir: Path) -> dict[str, dict[str, Any]]:
    export_plan = {
        "stl": output_dir / "STL" / f"{base_name}_Hunyuan3D.stl",
        "obj": output_dir / "OBJ" / f"{base_name}_Hunyuan3D.obj",
        "glb": output_dir / "GLB" / f"{base_name}_Hunyuan3D.glb",
    }
    results: dict[str, dict[str, Any]] = {}

    print("\nExportando resultados...")
    for export_type, path in export_plan.items():
        try:
            mesh.export(str(path), file_type=export_type)
            results[export_type] = {
                "ok": True,
                "path": str(path),
                "size_mb": round(path.stat().st_size / 1024**2, 2),
            }
            print(f"✅ {export_type.upper()}: {path}")
        except Exception as exc:
            results[export_type] = {
                "ok": False,
                "path": str(path),
                "error": str(exc),
            }
            print(f"⚠ {export_type.upper()} fallo: {exc}")

    if not any(item["ok"] for item in results.values()):
        raise RuntimeError("Todas las exportaciones fallaron.")

    return results


def build_metadata(
    image_name: str,
    input_path: Path,
    processed_image_path: Path,
    output_dir: Path,
    exports: dict[str, dict[str, Any]],
    mesh,
    remove_background_applied: bool,
    started_at: float,
) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "source_image_name": image_name,
        "source_image_path": str(input_path),
        "processed_image": str(processed_image_path),
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "model_id": MODEL_ID,
        "model_subfolder": MODEL_SUBFOLDER,
        "octree_resolution": OCTREE_RESOLUTION,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED,
        "remove_background_requested": REMOVE_BACKGROUND,
        "remove_background_applied": remove_background_applied,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "exports": exports,
        "processing_seconds": round(time.time() - started_at, 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_name": STATE.torch.cuda.get_device_name(0),
    }

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata


def process_image(image_name: str, image_bytes: bytes) -> dict[str, Any]:
    started_at = time.time()
    safe_name = sanitize_filename(image_name)
    output_dir = create_result_directory(safe_name)

    print(f"\n📥 Archivo recibido: {image_name}")
    print(f"📁 Carpeta de salida: {output_dir}")

    (
        safe_name,
        input_path,
        image,
        processed_image_path,
        remove_background_applied,
    ) = prepare_input_image(image_name, image_bytes, output_dir)

    generator = STATE.torch.Generator(device="cuda").manual_seed(SEED)

    print("\nGenerando geometria...")
    with STATE.torch.inference_mode():
        result = STATE.pipeline(
            image=image,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            octree_resolution=OCTREE_RESOLUTION,
        )

    mesh = result[0]
    if mesh is None:
        raise RuntimeError("La pipeline termino sin devolver una malla.")

    print("\n✅ Geometria generada.")
    cleanup_mesh(mesh)
    exports = export_mesh(mesh, Path(safe_name).stem, output_dir)
    metadata = build_metadata(
        image_name=image_name,
        input_path=input_path,
        processed_image_path=processed_image_path,
        output_dir=output_dir,
        exports=exports,
        mesh=mesh,
        remove_background_applied=remove_background_applied,
        started_at=started_at,
    )

    del mesh
    del image
    cleanup_gpu()
    return metadata


def render_preview(image_bytes: bytes) -> None:
    if STATE.preview_widget is None:
        return

    preview_image = STATE.Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    preview_buffer = io.BytesIO()
    preview_image.save(preview_buffer, format="PNG")

    with STATE.preview_widget:
        STATE.clear_output(wait=True)
        STATE.display(
            STATE.widgets.HTML(
                "<div style='font-weight:700;margin:4px 0 10px 0'>Vista previa</div>"
            )
        )
        STATE.display(
            STATE.widgets.Image(
                value=preview_buffer.getvalue(),
                format="png",
                width=320,
            )
        )


def render_downloads(metadata: dict[str, Any]) -> None:
    buttons = []

    def build_download_button(label: str, path: str):
        button = STATE.widgets.Button(
            description=label,
            button_style="success",
            layout=STATE.widgets.Layout(width="180px"),
        )

        def _download(_):
            STATE.files.download(path)

        button.on_click(_download)
        return button

    for export_type in ("stl", "obj", "glb"):
        export_data = metadata["exports"].get(export_type)
        if export_data and export_data.get("ok"):
            buttons.append(
                build_download_button(
                    f"Descargar {export_type.upper()}",
                    export_data["path"],
                )
            )

    processed_button = build_download_button(
        "Descargar imagen",
        metadata["processed_image"],
    )
    metadata_button = build_download_button(
        "Descargar metadata",
        metadata["metadata_path"],
    )
    buttons.extend([processed_button, metadata_button])

    STATE.downloads_box.children = tuple(buttons)


def render_metadata(metadata: dict[str, Any]) -> None:
    exports = []
    for export_type in ("stl", "obj", "glb"):
        export_data = metadata["exports"][export_type]
        if export_data.get("ok"):
            exports.append(
                f"<li><b>{export_type.upper()}</b>: OK - {export_data['size_mb']} MB</li>"
            )
        else:
            exports.append(
                f"<li><b>{export_type.upper()}</b>: fallo - {export_data['error']}</li>"
            )

    html = (
        "<div style='padding:14px;border:1px solid #d9dee7;border-radius:12px;"
        "background:#ffffff;margin-top:8px'>"
        "<div style='font-size:16px;font-weight:700;margin-bottom:8px'>Resultado</div>"
        f"<div><b>Vertices:</b> {metadata['vertices']:,}</div>"
        f"<div><b>Caras:</b> {metadata['faces']:,}</div>"
        f"<div><b>Malla cerrada:</b> {metadata['watertight']}</div>"
        f"<div><b>Tiempo:</b> {metadata['processing_seconds']} s</div>"
        f"<div><b>Carpeta:</b> {metadata['output_dir']}</div>"
        "<div style='margin-top:8px'><b>Exportaciones</b></div>"
        f"<ul>{''.join(exports)}</ul>"
        "</div>"
    )
    STATE.metadata_box.children = (STATE.widgets.HTML(value=html),)


def reset_generation_state() -> None:
    STATE.downloads_box.children = ()
    STATE.metadata_box.children = ()
    STATE.last_result = None


def handle_file_change(change) -> None:
    if change.get("name") != "value":
        return

    try:
        image_name, image_bytes = unpack_file_upload_value(change["new"])
    except Exception as exc:
        update_status(f"Error leyendo el archivo: {exc}", "error")
        return

    STATE.image_name = image_name
    STATE.image_bytes = image_bytes
    reset_generation_state()
    render_preview(image_bytes)
    update_status(
        "Imagen lista. El modelo ya esta cargado; puedes generar cuando quieras.",
        "success",
    )


def handle_generate_click(_button) -> None:
    if STATE.busy:
        update_status("Ya hay una generacion en curso.", "warning")
        return

    if not STATE.image_bytes or not STATE.image_name:
        update_status("Primero sube una imagen valida.", "warning")
        return

    with STATE.lock:
        STATE.busy = True

    update_status("Generando malla 3D...", "info")
    STATE.generate_button.disabled = True

    try:
        with STATE.output_widget:
            STATE.clear_output(wait=True)
            print("\n" + "=" * 72)
            print("GENERANDO MODELO 3D")
            print("=" * 72)
            metadata = process_image(STATE.image_name, STATE.image_bytes)
            STATE.last_result = metadata

        render_downloads(metadata)
        render_metadata(metadata)
        update_status(
            "Generacion terminada. Puedes descargar y volver a generar sin rerun.",
            "success",
        )

        if AUTO_DOWNLOAD:
            for export_type in ("stl", "obj", "glb"):
                export_data = metadata["exports"].get(export_type)
                if export_data and export_data.get("ok"):
                    STATE.files.download(export_data["path"])
    except Exception as exc:
        with STATE.output_widget:
            print("\n" + "=" * 72)
            print("ERROR")
            print("=" * 72)
            traceback.print_exc()
            if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
                print(
                    "\n⚠ Si el error fue CUDA out of memory, prueba bajar "
                    "OCTREE_RESOLUTION de 384 a 256."
                )
        cleanup_gpu()
        update_status(f"Fallo la generacion: {exc}", "error")
    finally:
        STATE.busy = False
        STATE.generate_button.disabled = False


def build_ui() -> None:
    widgets = STATE.widgets

    title = widgets.HTML(
        """
        <div style="padding:18px 20px;border-radius:18px;background:linear-gradient(135deg,#11253d,#1f4f73);color:white;margin-bottom:12px">
          <div style="font-size:24px;font-weight:800">Hunyuan3D Studio</div>
          <div style="margin-top:6px;font-size:14px;opacity:0.92">
            Sube una imagen, genera STL/OBJ/GLB y repite sin volver a correr la celda.
          </div>
        </div>
        """
    )

    guidance = widgets.HTML(
        """
        <div style="padding:12px 14px;border-radius:12px;background:#fff8e1;border:1px solid #f0d88a;margin-bottom:10px">
          <b>Para mejor geometria:</b> un solo objeto, vista de tres cuartos, fondo simple o transparente,
          buena iluminacion y que el objeto ocupe 70-85 % de la imagen.
        </div>
        """
    )

    upload = widgets.FileUpload(
        accept=".png,.jpg,.jpeg,.webp,.bmp",
        multiple=False,
        description="Subir imagen",
        layout=widgets.Layout(width="220px"),
    )
    upload.observe(handle_file_change, names="value")

    generate_button = widgets.Button(
        description="Generar 3D",
        button_style="primary",
        layout=widgets.Layout(width="220px", height="42px"),
    )
    generate_button.on_click(handle_generate_click)

    preview_widget = widgets.Output(
        layout=widgets.Layout(border="1px solid #d9dee7", padding="12px")
    )
    output_widget = widgets.Output(
        layout=widgets.Layout(border="1px solid #d9dee7", padding="12px")
    )
    downloads_box = widgets.HBox(
        [],
        layout=widgets.Layout(flex_flow="row wrap", gap="8px", margin="8px 0"),
    )
    metadata_box = widgets.VBox([])
    status_html = widgets.HTML()

    controls = widgets.HBox(
        [upload, generate_button],
        layout=widgets.Layout(gap="10px", flex_flow="row wrap"),
    )

    layout = widgets.VBox(
        [
            title,
            guidance,
            status_html,
            controls,
            preview_widget,
            widgets.HTML("<div style='font-weight:700;margin-top:10px'>Descargas</div>"),
            downloads_box,
            metadata_box,
            widgets.HTML("<div style='font-weight:700;margin-top:10px'>Consola</div>"),
            output_widget,
        ]
    )

    STATE.file_upload = upload
    STATE.generate_button = generate_button
    STATE.preview_widget = preview_widget
    STATE.output_widget = output_widget
    STATE.downloads_box = downloads_box
    STATE.metadata_box = metadata_box
    STATE.status_html = status_html

    STATE.display(layout)
    update_status(
        "Notebook listo. El modelo esta cargado una sola vez y queda esperando imagen.",
        "success",
    )


def launch_app() -> None:
    ensure_bootstrap()
    build_ui()
    STATE.ui_rendered = True
    print("\n✅ Interfaz renderizada. Si vuelves a correr la celda, el modelo se reutiliza.")


launch_app()
