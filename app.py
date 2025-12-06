import os
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    url_for,
    jsonify,
)
import torch
from huggingface_hub import snapshot_download
from diffusers import ZImagePipeline
import psutil
import pynvml


app = Flask(__name__)

# Add this:
import logging
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.WARNING)  # ou ERROR

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "static" / "generated"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Known models configuration
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Z-Image-Turbo": {
        "repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "local_dir": MODELS_DIR / "Z-Image-Turbo",
    },
    "Flux-Schnell": {
        "repo_id": "black-forest-labs/FLUX.1-schnell", 
        "local_dir": MODELS_DIR / "flux-schnell"
    },
    "SDXL-Turbo": {
        "repo_id": "stabilityai/sdxl-turbo", 
        "local_dir": MODELS_DIR / "sdxl-turbo"
    },
    "PixArt-XL-2-1024": {
        "repo_id": "PixArt-alpha/PixArt-XL-2-1024-MS", 
        "local_dir": MODELS_DIR / "pixart-xl"
    }
}

DEFAULT_MODEL_NAME = "Z-Image-Turbo"

# Global pipeline state (simple cache)
pipe: ZImagePipeline | None = None
current_model_name: str | None = None


# ---------------------------
# Model / pipeline handling
# ---------------------------

def ensure_model_downloaded(repo_id: str, model_dir: Path) -> Path:
    """
    Ensure the model is downloaded locally in model_dir without symlinks.
    If already present, just return the path.
    """
    model_index_path = model_dir / "model_index.json"

    if model_index_path.exists():
        print(f"Using existing local model at: {model_dir}")
        return model_dir

    print(f"Local model not found for repo {repo_id}. Downloading to: {model_dir}")

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Download finished. Model stored at: {model_dir}")
    return model_dir


def load_pipeline(model_name: str) -> ZImagePipeline:
    """
    Load (or reuse) a pipeline for the given model name.
    Uses global pipe/current_model_name to avoid reloading if not needed.
    """
    global pipe, current_model_name

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}")

    if current_model_name == model_name and pipe is not None:
        return pipe

    config = MODEL_CONFIGS[model_name]
    repo_id = config["repo_id"]
    model_dir = config["local_dir"]

    model_path = ensure_model_downloaded(repo_id, model_dir)

    # Free previous pipeline if any
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()

    print(f"Loading pipeline for model '{model_name}' from {model_path}")

    new_pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,    # réduit la VRAM
        local_files_only=True,
        low_cpu_mem_usage=False,       # comme app2.py - évite les conflits
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check your PyTorch + GPU setup.")

    # Charge directement sur GPU (comme app2.py)
    new_pipe.to("cuda")

    pipe = new_pipe
    current_model_name = model_name
    return pipe


def parse_repo_id_from_input(text: str) -> str:
    """
    Parse a Hugging Face repo_id from either:
    - Straight repo_id: 'author/model-name'
    - Or URL: 'https://huggingface.co/author/model-name[/...]'
    """
    text = text.strip()
    if "huggingface.co" not in text:
        return text

    text = text.rstrip("/")
    parts = text.split("/")
    if len(parts) < 2:
        raise ValueError(f"Could not parse repo_id from: {text}")
    repo_id = "/".join(parts[-2:])
    return repo_id


def register_custom_model(repo_or_url: str) -> str:
    """
    Register a new model from repo_id or HF URL.
    Returns the internal model name (display name).
    """
    if not repo_or_url.strip():
        return DEFAULT_MODEL_NAME

    repo_id = parse_repo_id_from_input(repo_or_url)
    model_name = repo_id.split("/")[-1]

    if model_name not in MODEL_CONFIGS:
        model_dir = MODELS_DIR / model_name
        MODEL_CONFIGS[model_name] = {
            "repo_id": repo_id,
            "local_dir": model_dir,
        }
        print(f"Registered new model '{model_name}' with repo_id '{repo_id}'.")

    return model_name


# ---------------------------
# System stats
# ---------------------------

def init_gpu_monitor():
    """Initialize NVML for GPU monitoring (NVIDIA only)."""
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        print(f"NVML init error: {e}")


def get_gpu_stats():
    """
    Return a list of GPU stats dictionaries using NVML.
    Works only with NVIDIA GPUs and NVML available.
    """
    try:
        init_gpu_monitor()
        device_count = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError:
        return []

    gpus = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        # Newer pynvml returns str, older versions return bytes
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            temp = None

        gpus.append(
            {
                "index": i,
                "name": name,
                "gpu_util_percent": util.gpu,
                "mem_used_MB": round(mem_info.used / (1024 ** 2), 1),
                "mem_total_MB": round(mem_info.total / (1024 ** 2), 1),
                "mem_util_percent": round(mem_info.used / mem_info.total * 100, 1)
                if mem_info.total > 0
                else 0.0,
                "temperature_C": temp,
            }
        )
    return gpus

def get_cpu_ram_stats():
    """
    Return CPU and RAM statistics.
    Temperatures may not be available on all platforms.
    """
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()

    stats = {
        "cpu_percent": cpu_percent,
        "ram_percent": ram.percent,
        "ram_used_GB": round(ram.used / (1024 ** 3), 2),
        "ram_total_GB": round(ram.total / (1024 ** 3), 2),
    }

    try:
        temps = psutil.sensors_temperatures()
        if temps:
            first_key = next(iter(temps.keys()))
            first_entry = temps[first_key][0]
            stats["cpu_temperature_C"] = first_entry.current
    except Exception:
        pass

    return stats


def get_system_stats():
    """
    Combine CPU/RAM and GPU stats into a single dictionary for JSON.
    """
    return {
        "cpu_ram": get_cpu_ram_stats(),
        "gpus": get_gpu_stats(),
    }


# ---------------------------
# Flask routes
# ---------------------------

@app.route("/", methods=["GET"])
def index():
    """
    Main page: show form + last generated images (none by default).
    """
    models = sorted(MODEL_CONFIGS.keys())
    return render_template(
        "index.html",
        models=models,
        selected_model=DEFAULT_MODEL_NAME,
        images=[],
        last_prompt="",
        last_steps=8,
        last_height=512,
        last_width=512,
        last_num_images=1,
        custom_model_repo="",
        error_message=None,
    )


def generate_images(
    prompt: str,
    steps: int,
    height: int,
    width: int,
    num_images: int,
    model_name: str,
) -> List[dict]:
    """
    Generate num_images images and save them to disk.
    Returns a list of dicts with metadata and paths.
    """
    pipeline = load_pipeline(model_name)

    images_info: List[dict] = []

    # Use a time-based seed if not controlled externally
    base_seed = int(time.time() * 1000) % 1_000_000_000

    for i in range(num_images):
        image_seed = base_seed + i
        generator = torch.Generator("cuda").manual_seed(image_seed)

        result = pipeline(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=0.0,
            generator=generator,
            height=int(height),
            width=int(width),
        )

        image = result.images[0]

        filename = f"{uuid.uuid4().hex}.png"
        filepath = IMAGES_DIR / filename
        image.save(filepath)

        images_info.append(
            {
                "filename": filename,
                "url": url_for("serve_generated_image", filename=filename),
                "download_url": url_for("download_image", filename=filename),
                "prompt": prompt,
                "model_name": model_name,
                "steps": steps,
                "height": height,
                "width": width,
            }
        )

    return images_info


@app.route("/generate", methods=["POST"])
def generate():
    """
    Handle form submission to generate images.
    """
    error_message = None

    try:
        model_name = request.form.get("model_name", DEFAULT_MODEL_NAME)
        custom_model_repo = request.form.get("custom_model_repo", "").strip()
        prompt = request.form.get("prompt", "").strip()

        steps = int(request.form.get("steps", "8"))
        height = int(request.form.get("height", "512"))
        width = int(request.form.get("width", "512"))
        num_images = int(request.form.get("num_images", "1"))

        if custom_model_repo:
            # If a custom repo is given, register and use it
            model_name = register_custom_model(custom_model_repo)

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        images_info = generate_images(
            prompt=prompt,
            steps=steps,
            height=height,
            width=width,
            num_images=num_images,
            model_name=model_name,
        )

        models = sorted(MODEL_CONFIGS.keys())
        return render_template(
            "index.html",
            models=models,
            selected_model=model_name,
            images=images_info,
            last_prompt=prompt,
            last_steps=steps,
            last_height=height,
            last_width=width,
            last_num_images=num_images,
            custom_model_repo="",
            error_message=None,
        )

    except Exception as e:
        error_message = str(e)
        print(f"Error during generation: {error_message}")

        models = sorted(MODEL_CONFIGS.keys())
        return render_template(
            "index.html",
            models=models,
            selected_model=DEFAULT_MODEL_NAME,
            images=[],
            last_prompt="",
            last_steps=8,
            last_height=512,
            last_width=512,
            last_num_images=1,
            custom_model_repo="",
            error_message=error_message,
        )


@app.route("/static/generated/<path:filename>")
def serve_generated_image(filename: str):
    """
    Serve generated images from disk.
    """
    return send_from_directory(IMAGES_DIR, filename)


@app.route("/download/<path:filename>")
def download_image(filename: str):
    """
    Force image download to disk.
    """
    return send_from_directory(IMAGES_DIR, filename, as_attachment=True)


@app.route("/api/system_stats", methods=["GET"])
def api_system_stats():
    """
    Return system stats as JSON for the frontend.
    """
    stats = get_system_stats()
    return jsonify(stats)


if __name__ == "__main__":
    # For development only; use a proper WSGI server in production
    app.run(host="127.0.0.1", port=7860, debug=True)
