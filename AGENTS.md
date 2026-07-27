
# AGENTS.md

# Hunyuan3D Studio

> Engineering guide for AI coding agents (Codex, ChatGPT, Claude, Gemini, etc.)

---

# Mission

Transform the current Hunyuan3D Google Colab notebook into a production-grade application for converting images into high-quality printable 3D models.

The notebook should eventually feel like a desktop application running inside Google Colab instead of a research notebook.

The current implementation is considered **stable** and must always remain reproducible.

---

# Core Principles

Priority order:

1. Stability
2. Mesh quality
3. Correctness
4. Maintainability
5. User experience
6. Performance

Never sacrifice mesh quality for speed unless explicitly requested.

---

# Current State

The notebook already:

- Mounts Google Drive.
- Reuses HuggingFace cache.
- Downloads the repository only once.
- Generates STL / OBJ / GLB.
- Uses Google Colab with NVIDIA T4.
- Produces correct results.

Treat the current implementation as the reference implementation.

---

# Target State

Desired workflow:

Notebook starts

↓

Dependencies checked

↓

Model loaded ONE TIME

↓

Wait for image

↓

Generate mesh

↓

Export

↓

Wait for another image

↓

Repeat forever

No second model load should occur while the Colab session is alive.

---

# Quality

Always prefer maximum quality.

Target configuration:

```python
OCTREE_RESOLUTION = 384
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 5.5
```

Do not introduce Draft/Fast/Low quality modes unless explicitly requested.

---

# Hardware Target

- Google Colab
- NVIDIA T4 GPU

Stay within Colab limitations whenever possible.

---

# Google Drive

Persist:

- HuggingFace cache
- Repository
- Outputs

Never redownload files already available.

Never overwrite previous outputs.

---

# Outputs

Each generation should have its own folder.

Example:

resultados/

    part_name_timestamp/

        STL

        OBJ

        GLB

        processed_image

        metadata.json

---

# Exports

Always export:

- STL
- OBJ
- GLB

If one export fails, try to preserve the remaining ones whenever possible.

---

# UI Goals

Target UI:

- Upload image
- Preview image
- Generate button
- Progress information
- Download links
- Generate another image

No need to rerun the notebook cell.

---

# Code Style

Prefer:

- Small functions
- Clear names
- Single responsibility
- Minimal globals
- Readable code

Avoid giant monolithic functions.

Avoid duplicated logic.

---

# Refactoring Rules

Before changing architecture:

1. Understand current behavior.
2. Preserve functionality.
3. Improve readability.
4. Then optimize.

Never rewrite large blocks only because another implementation looks cleaner.

---

# Error History

These bugs already happened.

Do not reintroduce them.

## CUDA

Never do:

```python
pipeline = pipeline.to("cuda")
```

This overwrites the pipeline with None.

Correct usage:

```python
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    ...,
    device="cuda"
)
```

## Torch

Do not reinstall torch.

Use Colab's version.

## HuggingFace

Reuse cache stored in Google Drive.

---

# Future Features

Not required immediately.

Architecture should make them easy to implement.

- Batch generation
- ZIP download
- GLB preview
- REST API
- Automatic repair
- History
- Statistics
- VRAM usage
- Processing time

---

# Pull Request Checklist (Mental)

Before considering a change complete:

- [ ] Existing functionality preserved
- [ ] No duplicated code
- [ ] No unnecessary globals
- [ ] No quality regression
- [ ] Outputs still generated
- [ ] Drive cache preserved
- [ ] STL export validated
- [ ] GPU still used

---

# Testing Checklist

Minimum manual tests:

1.

Fresh Colab session

↓

Run notebook

↓

Generate STL

2.

Generate second STL

↓

Without reloading model

3.

Generate third STL

↓

Without rerunning notebook

4.

Confirm outputs exist.

5.

Confirm Drive cache reused.

6.

Confirm GPU active.

---

# Philosophy

Prefer engineering over cleverness.

Readable code beats smart code.

Stable code beats short code.

Correct code beats fast code.

---

# Freedom

The implementation may change completely.

The architecture may change completely.

The notebook structure may change completely.

However:

The user-visible functionality must never regress.

If a major redesign is proposed, explain why it is better.

---

# Long-Term Vision

Eventually this project should behave like:

Image

↓

Automatic preprocessing

↓

Background removal

↓

Hunyuan3D

↓

Mesh cleanup

↓

Export

↓

Ready for printing

without manual intervention.

---

# Notes for AI Agents

Assume the existing notebook is correct unless proven otherwise.

Do not delete code simply because it looks redundant.

Understand its purpose first.

When uncertain, preserve behavior.

Always explain architectural decisions.

---

# CURRENT STABLE IMPLEMENTATION (V1)

The following section contains the complete notebook that is currently known to work correctly.

It is intentionally embedded below so future AI agents always have the reference implementation available.

Do not modify this section directly.

Instead, build improvements on top of its behavior.

=================================================================

# ============================================================
# HUNYUAN3D-2 — IMAGEN A STL / OBJ / GLB
# GOOGLE COLAB — UNA SOLA CELDA
#
# MODELO Y CÓDIGO PERSISTENTES EN:
# /content/drive/MyDrive/IA_Modelos/Hunyuan3D-2
#
# Recomendado:
# Entorno de ejecución -> Cambiar tipo -> GPU T4
# ============================================================

import os
import sys
import shutil
import subprocess
import textwrap
import time
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

DRIVE_DIR = Path(
    "/content/drive/MyDrive/IA_Modelos/Hunyuan3D-2"
)

REPO_DIR = DRIVE_DIR / "repo"
HF_CACHE_DIR = DRIVE_DIR / "modelos_huggingface"
RESULTS_DIR = DRIVE_DIR / "resultados"

RUNTIME_DIR = Path("/content/hunyuan3d_runtime")
INPUT_DIR = RUNTIME_DIR / "input"

# Modelo completo de geometría:
#MODEL_ID = "tencent/Hunyuan3D-2"
#MODEL_SUBFOLDER = "hunyuan3d-dit-v2-0"
MODEL_ID = "tencent/Hunyuan3D-2mini"
MODEL_SUBFOLDER = "hunyuan3d-dit-v2-mini-turbo"
# Calidad:
#
# OCTREE_RESOLUTION:
# 256 = calidad normal
# 384 = más detalle, recomendado para T4
# 512 = máximo detalle, más lento y más VRAM
#
# Si 384 da CUDA out of memory, cambia a 256.
OCTREE_RESOLUTION = 256
NUM_INFERENCE_STEPS = 8
GUIDANCE_SCALE = 5.0

#OCTREE_RESOLUTION = 384

# 30 es el valor estándar del proyecto.
#NUM_INFERENCE_STEPS = 30

# Valor oficial habitual: 5.0–5.5.
#GUIDANCE_SCALE = 5.5

# Semilla reproducible.
SEED = 1234

# True:
# intenta quitar el fondo automáticamente.
#
# False:
# úsalo para PNG transparente ya preparado.
REMOVE_BACKGROUND = True

# Descargar automáticamente el STL al terminar.
AUTO_DOWNLOAD = True


# ============================================================
# FUNCIONES
# ============================================================

def ejecutar(comando, cwd=None, env=None):
    if isinstance(comando, (list, tuple)):
        mostrado = " ".join(str(parte) for parte in comando)
        shell = False
    else:
        mostrado = comando
        shell = True

    print(f"\n▶ {mostrado}\n", flush=True)

    proceso = subprocess.Popen(
        comando,
        cwd=str(cwd) if cwd else None,
        env=env,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    salida = []

    for linea in proceso.stdout:
        print(linea, end="", flush=True)
        salida.append(linea)

    proceso.wait()

    if proceso.returncode != 0:
        ultimas_lineas = "".join(salida[-80:])

        raise RuntimeError(
            f"\nFalló el comando con código "
            f"{proceso.returncode}:\n"
            f"{mostrado}\n\n"
            f"ÚLTIMA SALIDA DEL PROCESO:\n"
            f"{ultimas_lineas}"
        )

    return proceso


def archivo_valido(ruta, minimo_bytes=1):
    ruta = Path(ruta)

    return (
        ruta.exists()
        and ruta.is_file()
        and ruta.stat().st_size >= minimo_bytes
    )


# ============================================================
# 1. MONTAR GOOGLE DRIVE
# ============================================================

print("=" * 72)
print("MONTANDO GOOGLE DRIVE")
print("=" * 72)

from google.colab import drive, files

drive.mount("/content/drive")


# ============================================================
# 2. CREAR DIRECTORIOS
# ============================================================

DRIVE_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if RUNTIME_DIR.exists():
    shutil.rmtree(RUNTIME_DIR)

INPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n✅ Directorio persistente:\n{DRIVE_DIR}")
print(f"\n✅ Caché del modelo:\n{HF_CACHE_DIR}")
print(f"\n✅ Resultados:\n{RESULTS_DIR}")


# ============================================================
# 3. COMPROBAR GPU
# ============================================================

print("\n" + "=" * 72)
print("COMPROBANDO GPU")
print("=" * 72)

try:
    salida_gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader"
        ],
        text=True
    ).strip()

    print(f"\n✅ GPU detectada: {salida_gpu}")

except Exception:
    raise RuntimeError(
        "No se detectó una GPU NVIDIA.\n\n"
        "Activa:\n"
        "Entorno de ejecución → Cambiar tipo de entorno → T4 GPU"
    )


# ============================================================
# 4. CLONAR O REUTILIZAR HUNYUAN3D-2
# ============================================================

print("\n" + "=" * 72)
print("COMPROBANDO REPOSITORIO HUNYUAN3D-2")
print("=" * 72)

repo_correcto = (
    archivo_valido(REPO_DIR / "setup.py", 100)
    and archivo_valido(REPO_DIR / "requirements.txt", 100)
    and (REPO_DIR / "hy3dgen").is_dir()
)

if repo_correcto:
    print(f"\n✅ Repositorio encontrado:\n{REPO_DIR}")
    print("✅ Se omite la descarga.")

else:
    print("\n⬇ El repositorio falta o está incompleto.")

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)

    ejecutar([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git",
        str(REPO_DIR)
    ])

    if not (REPO_DIR / "hy3dgen").is_dir():
        raise RuntimeError(
            "El repositorio se descargó, pero falta hy3dgen."
        )

    print("\n✅ Repositorio guardado en Drive.")


# ============================================================
# 5. DEPENDENCIAS DEL SISTEMA
# ============================================================

print("\n" + "=" * 72)
print("INSTALANDO DEPENDENCIAS DEL SISTEMA")
print("=" * 72)

ejecutar(
    "apt-get update -qq && "
    "apt-get install -y -qq "
    "libgl1 libglib2.0-0 libegl1 ninja-build"
)


# ============================================================
# 6. DEPENDENCIAS DE PYTHON
# ============================================================

print("\n" + "=" * 72)
print("INSTALANDO DEPENDENCIAS DE PYTHON")
print("=" * 72)

# El código de inferencia se ejecutará en un proceso Python nuevo.
# Esto evita el problema de módulos viejos ya cargados en memoria.

ejecutar([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "--upgrade",
    "pip",
    "setuptools",
    "wheel"
])

# Reinstalación completa para evitar el Frankenstein anterior
# de huggingface_hub.
ejecutar([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "--upgrade",
    "--force-reinstall",
    "huggingface_hub==0.34.4"
])

# No reinstalamos torch ni torchvision porque Colab ya incluye
# versiones preparadas para su GPU.
DEPENDENCIAS = [
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
    "Pillow"
]

ejecutar(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade"
    ] + DEPENDENCIAS
)

# Instalar el repositorio sin volver a resolver todas sus
# dependencias.
ejecutar([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "--editable",
    str(REPO_DIR),
    "--no-deps"
])


# ============================================================
# 7. SELECCIONAR IMAGEN
# ============================================================

print("\n" + "=" * 72)
print("SELECCIONA UNA IMAGEN")
print("=" * 72)

print("""
Para obtener mejor geometría:

• Un solo objeto.
• Objeto completo, sin partes cortadas.
• Vista de tres cuartos.
• Fondo simple o transparente.
• Buena iluminación.
• Evita objetos transparentes.
• Evita sombras muy fuertes.
• Que el objeto ocupe aproximadamente 70–85 % de la imagen.
""")

subidos = files.upload()

if not subidos:
    raise RuntimeError("No seleccionaste ninguna imagen.")

nombre_original, contenido = next(iter(subidos.items()))

extension = Path(nombre_original).suffix.lower()

extensiones_validas = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp"
}

if extension not in extensiones_validas:
    raise ValueError(
        f"Formato no permitido: {extension}\n"
        "Usa PNG, JPG, JPEG, WEBP o BMP."
    )

nombre_seguro = "".join(
    caracter
    if caracter.isalnum() or caracter in "._-"
    else "_"
    for caracter in Path(nombre_original).name
)

INPUT_IMAGE = INPUT_DIR / nombre_seguro

with open(INPUT_IMAGE, "wb") as archivo:
    archivo.write(contenido)

print(f"\n✅ Imagen cargada:\n{INPUT_IMAGE}")


# ============================================================
# 8. CREAR SCRIPT DE INFERENCIA
# ============================================================

GENERATOR_SCRIPT = RUNTIME_DIR / "generar_hunyuan.py"

codigo_generador = r'''
import os
import gc
import sys
import json
import traceback
from pathlib import Path

import torch
from PIL import Image


input_image = Path(os.environ["HY_INPUT_IMAGE"])
output_dir = Path(os.environ["HY_OUTPUT_DIR"])

model_id = os.environ["HY_MODEL_ID"]
model_subfolder = os.environ["HY_MODEL_SUBFOLDER"]

octree_resolution = int(
    os.environ.get("HY_OCTREE_RESOLUTION", "384")
)

num_inference_steps = int(
    os.environ.get("HY_NUM_INFERENCE_STEPS", "30")
)

guidance_scale = float(
    os.environ.get("HY_GUIDANCE_SCALE", "5.5")
)

seed = int(
    os.environ.get("HY_SEED", "1234")
)

remove_background = (
    os.environ.get("HY_REMOVE_BACKGROUND", "1") == "1"
)

output_dir.mkdir(parents=True, exist_ok=True)


def limpiar_gpu():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


print("=" * 72)
print("INICIANDO HUNYUAN3D-2")
print("=" * 72)

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch no detectó CUDA dentro del proceso de inferencia."
    )

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Modelo: {model_id}")
print(f"Subfolder: {model_subfolder}")
print(f"Octree resolution: {octree_resolution}")
print(f"Inference steps: {num_inference_steps}")
print(f"Guidance scale: {guidance_scale}")
print(f"Seed: {seed}")

# Imports después de comprobar CUDA.
from hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline
)

from hy3dgen.rembg import BackgroundRemover


# ------------------------------------------------------------
# ABRIR Y PREPARAR IMAGEN
# ------------------------------------------------------------

image = Image.open(input_image)

tiene_alpha_util = False

if image.mode in ("RGBA", "LA"):
    alpha = image.getchannel("A")
    minimo_alpha, maximo_alpha = alpha.getextrema()

    tiene_alpha_util = (
        minimo_alpha < 250
        and maximo_alpha > 0
    )

image = image.convert("RGBA")

if remove_background and not tiene_alpha_util:
    print("\nEliminando fondo automáticamente...")

    rembg = BackgroundRemover()
    image = rembg(image)

    del rembg
    limpiar_gpu()

elif tiene_alpha_util:
    print("\nLa imagen ya contiene transparencia útil.")
    print("Se conserva el fondo transparente original.")

else:
    print("\nEliminación de fondo desactivada.")

processed_image = output_dir / "imagen_procesada.png"
image.save(processed_image)

print(f"Imagen procesada: {processed_image}")


# ------------------------------------------------------------
# CARGAR MODELO
# ------------------------------------------------------------

print("\nCargando Hunyuan3D-2...")

#pipeline = (
#    Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
#        model_id,
#        subfolder=model_subfolder,
#        torch_dtype=torch.float16
#    )
#)

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    model_id,
    subfolder=model_subfolder,
    torch_dtype=torch.float16,
    device="cuda"
)

#pipeline = pipeline.to("cuda")

print("Modelo cargado.")


# ------------------------------------------------------------
# GENERAR MALLA
# ------------------------------------------------------------

generator = torch.Generator(
    device="cuda"
).manual_seed(seed)

print("\nGenerando geometría...")

with torch.inference_mode():
    resultado = pipeline(
        image=image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution
    )

mesh = resultado[0]

if mesh is None:
    raise RuntimeError(
        "La pipeline terminó sin devolver una malla."
    )

print("\nGeometría generada.")


# ------------------------------------------------------------
# LIMPIEZA BÁSICA
# ------------------------------------------------------------

try:
    mesh.remove_unreferenced_vertices()
except Exception:
    pass

try:
    mesh.merge_vertices()
except Exception:
    pass

try:
    mesh.remove_infinite_values()
except Exception:
    pass


# ------------------------------------------------------------
# EXPORTAR
# ------------------------------------------------------------

nombre_base = input_image.stem

stl_path = output_dir / f"{nombre_base}_Hunyuan3D.stl"
obj_path = output_dir / f"{nombre_base}_Hunyuan3D.obj"
glb_path = output_dir / f"{nombre_base}_Hunyuan3D.glb"

print("\nExportando resultados...")

mesh.export(
    str(stl_path),
    file_type="stl"
)

mesh.export(
    str(obj_path),
    file_type="obj"
)

mesh.export(
    str(glb_path),
    file_type="glb"
)

stats = {
    "stl": str(stl_path),
    "obj": str(obj_path),
    "glb": str(glb_path),
    "vertices": int(len(mesh.vertices)),
    "faces": int(len(mesh.faces)),
    "watertight": bool(mesh.is_watertight),
    "stl_size_mb": round(
        stl_path.stat().st_size / 1024**2,
        2
    )
}

stats_path = output_dir / "resultado.json"

with open(stats_path, "w", encoding="utf-8") as archivo:
    json.dump(
        stats,
        archivo,
        ensure_ascii=False,
        indent=2
    )

print("\n" + "=" * 72)
print("RESULTADO")
print("=" * 72)

print(f"Vértices: {stats['vertices']:,}")
print(f"Caras: {stats['faces']:,}")
print(f"Malla cerrada: {stats['watertight']}")
print(f"Tamaño STL: {stats['stl_size_mb']} MB")

print(f"\nSTL: {stl_path}")
print(f"OBJ: {obj_path}")
print(f"GLB: {glb_path}")

# Liberar VRAM.
del pipeline
del mesh
limpiar_gpu()
'''

GENERATOR_SCRIPT.write_text(
    textwrap.dedent(codigo_generador),
    encoding="utf-8"
)

print(f"\n✅ Script temporal creado:\n{GENERATOR_SCRIPT}")


# ============================================================
# 9. CREAR CARPETA DE RESULTADO
# ============================================================

timestamp = time.strftime("%Y%m%d_%H%M%S")

nombre_base = Path(nombre_seguro).stem

OUTPUT_DIR = (
    RESULTS_DIR /
    f"{nombre_base}_{timestamp}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 10. EJECUTAR EN PROCESO PYTHON LIMPIO
# ============================================================

print("\n" + "=" * 72)
print("GENERANDO MODELO 3D")
print("=" * 72)

env = os.environ.copy()

# Caché persistente en Drive.
env["HF_HOME"] = str(HF_CACHE_DIR)
env["HUGGINGFACE_HUB_CACHE"] = str(
    HF_CACHE_DIR / "hub"
)
env["TRANSFORMERS_CACHE"] = str(
    HF_CACHE_DIR / "transformers"
)

# Evita usar hf_transfer.
env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Parámetros.
env["HY_INPUT_IMAGE"] = str(INPUT_IMAGE)
env["HY_OUTPUT_DIR"] = str(OUTPUT_DIR)
env["HY_MODEL_ID"] = MODEL_ID
env["HY_MODEL_SUBFOLDER"] = MODEL_SUBFOLDER

env["HY_OCTREE_RESOLUTION"] = str(
    OCTREE_RESOLUTION
)

env["HY_NUM_INFERENCE_STEPS"] = str(
    NUM_INFERENCE_STEPS
)

env["HY_GUIDANCE_SCALE"] = str(
    GUIDANCE_SCALE
)

env["HY_SEED"] = str(SEED)

env["HY_REMOVE_BACKGROUND"] = (
    "1" if REMOVE_BACKGROUND else "0"
)

# Optimización de memoria de CUDA.
env["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"
    "max_split_size_mb:128"
)

# Añadir repo al PYTHONPATH.
pythonpath_actual = env.get("PYTHONPATH", "")

env["PYTHONPATH"] = (
    f"{REPO_DIR}:"
    f"{pythonpath_actual}"
)

try:
    ejecutar(
        [
            sys.executable,
            str(GENERATOR_SCRIPT)
        ],
        cwd=REPO_DIR,
        env=env
    )

except RuntimeError as error:
    mensaje = str(error)

    if (
        "out of memory" in mensaje.lower()
        or "cuda" in mensaje.lower()
    ):
        print(
            "\n⚠ Si el error fue CUDA out of memory, "
            "cambia:\n\n"
            "OCTREE_RESOLUTION = 384\n\n"
            "por:\n\n"
            "OCTREE_RESOLUTION = 256"
        )

    raise


# ============================================================
# 11. LEER RESULTADO
# ============================================================

import json

RESULT_JSON = OUTPUT_DIR / "resultado.json"

if not RESULT_JSON.exists():
    raise RuntimeError(
        "La generación terminó, pero no apareció resultado.json."
    )

with open(
    RESULT_JSON,
    "r",
    encoding="utf-8"
) as archivo:
    datos_resultado = json.load(archivo)

STL_PATH = Path(datos_resultado["stl"])
OBJ_PATH = Path(datos_resultado["obj"])
GLB_PATH = Path(datos_resultado["glb"])

if not STL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el STL:\n{STL_PATH}"
    )


# ============================================================
# 12. RESUMEN
# ============================================================

print("\n" + "=" * 72)
print("🔥 HUNYUAN3D TERMINÓ")
print("=" * 72)

print(f"\n✅ Vértices: {datos_resultado['vertices']:,}")
print(f"✅ Caras: {datos_resultado['faces']:,}")
print(
    f"✅ Malla cerrada: "
    f"{datos_resultado['watertight']}"
)
print(
    f"✅ Tamaño STL: "
    f"{datos_resultado['stl_size_mb']} MB"
)

print(f"\n✅ STL:\n{STL_PATH}")
print(f"\n✅ OBJ:\n{OBJ_PATH}")
print(f"\n✅ GLB:\n{GLB_PATH}")

print(
    "\nLos pesos quedan cacheados en Drive. "
    "La siguiente ejecución debe reutilizarlos."
)


# ============================================================
# 13. DESCARGAR STL
# ============================================================

if AUTO_DOWNLOAD:
    print("\n⬇ Iniciando descarga del STL...")
    files.download(str(STL_PATH))

=================================================================
