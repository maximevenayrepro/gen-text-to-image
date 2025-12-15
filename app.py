import os
import uuid
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import (
    Flask,
    render_template,
    request,
    url_for,
    jsonify,
    session,
    Response,
)
import base64
from io import BytesIO
import torch
from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline, ZImagePipeline, StableDiffusion3Pipeline
import psutil
import pynvml


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session

# Add this:
import logging
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.WARNING)  # or ERROR

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "static" / "generated"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = BASE_DIR / "config.json"

# Known models configuration
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "SDXL-Turbo": {
        "repo_id": "stabilityai/sdxl-turbo",
        "local_dir": MODELS_DIR / "sdxl-turbo",
        "pipeline_type": "sdxl",
        "default_steps": 4,
        "default_height": 512,
        "default_width": 512,
    },
    "SDXL": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "local_dir": MODELS_DIR / "sdxl-base",
        "pipeline_type": "sdxl",
        "default_steps": 25,
        "default_height": 1024,
        "default_width": 1024,
    },
    "SD-Turbo": {
        "repo_id": "stabilityai/sd-turbo",
        "local_dir": MODELS_DIR / "sd-turbo",
        "pipeline_type": "sd",
        "default_steps": 4,
        "default_height": 512,
        "default_width": 512,
    },
    "Z-Image-Turbo": {
        "repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "local_dir": MODELS_DIR / "Z-Image-Turbo",
        "pipeline_type": "z_image",
        "default_steps": 8,
        "default_height": 512,
        "default_width": 512,
    },
    "stable-diffusion-3.5-large": {
        "repo_id": "stabilityai/stable-diffusion-3.5-large",
        "local_dir": MODELS_DIR / "stable-diffusion-3.5-large",
        "pipeline_type": "sd3",
        "default_steps": 28,
        "default_height": 1024,
        "default_width": 1024,
    }
}

DEFAULT_MODEL_NAME = "SDXL"

# Global pipeline state (simple cache)
pipe: DiffusionPipeline | ZImagePipeline | StableDiffusion3Pipeline | None = None
current_model_name: str | None = None

# In-memory cache for generated images (base64)
# Key: image_id, Value: dict with base64 and metadata
image_cache: Dict[str, Dict[str, Any]] = {}


# ---------------------------
# Token management
# ---------------------------

def load_hf_token() -> Optional[str]:
    """
    Load Hugging Face token from config.json file.
    Returns the token string or None if not found.
    """
    if not CONFIG_FILE.exists():
        return None
    
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            return config.get("hf_token")
    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"Error loading token from config.json: {e}")
        return None


def save_hf_token(token: str) -> bool:
    """
    Save Hugging Face token to config.json file.
    Returns True if successful, False otherwise.
    """
    try:
        config = {}
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, IOError):
                config = {}
        
        config["hf_token"] = token
        
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        # Set environment variable for Hugging Face Hub
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        
        return True
    except IOError as e:
        print(f"Error saving token to config.json: {e}")
        return False


def get_hf_token() -> Optional[str]:
    """
    Get Hugging Face token from environment variable or config file.
    Returns the token string or None if not found.
    """
    # First check environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    
    # Then check config file
    token = load_hf_token()
    if token:
        # Set environment variable for future use
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    
    return token


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
    
    # Get token for authentication
    token = get_hf_token()
    token_kwargs = {"token": token} if token else {}

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        **token_kwargs,
    )

    print(f"Download finished. Model stored at: {model_dir}")
    return model_dir


def get_default_steps(pipeline) -> int | None:
    """
    Extract default number of inference steps from pipeline configuration.
    Returns int or None if not found.
    """
    # Check if this is a turbo model (1-4 steps recommended)
    if hasattr(pipeline, "config") and hasattr(pipeline.config, "_name_or_path"):
        model_path = pipeline.config._name_or_path
        if "turbo" in str(model_path).lower() or "schnell" in str(model_path).lower():
            return 4
    
    # Method 1: Check scheduler config (most reliable)
    if hasattr(pipeline, "scheduler") and hasattr(pipeline.scheduler, "config"):
        scheduler_config = pipeline.scheduler.config
        # Some schedulers have num_train_timesteps which can indicate default steps
        if hasattr(scheduler_config, "num_train_timesteps"):
            num_train_timesteps = scheduler_config.num_train_timesteps
            # For turbo models, default is usually 1-4 steps
            # For regular models, default is usually 20-50 steps
            # We can infer from the number of timesteps
            if num_train_timesteps <= 100:
                # Likely a turbo model, default to 4 steps
                return 4
            elif num_train_timesteps <= 1000:
                # Regular model, default to 20-30 steps
                return 20
            else:
                # SDXL or similar, default to 30-50 steps
                return 50
    
    # Method 2: Check pipeline config
    if hasattr(pipeline, "config"):
        config = pipeline.config
        if hasattr(config, "num_inference_steps"):
            return config.num_inference_steps
        if hasattr(config, "scheduler") and isinstance(config.scheduler, dict):
            if "num_train_timesteps" in config.scheduler:
                num_train_timesteps = config.scheduler["num_train_timesteps"]
                if num_train_timesteps <= 100:
                    return 4
                elif num_train_timesteps <= 1000:
                    return 20
                else:
                    return 50
    
    # Method 3: Check scheduler directly
    if hasattr(pipeline, "scheduler"):
        scheduler = pipeline.scheduler
        if hasattr(scheduler, "num_train_timesteps"):
            num_train_timesteps = scheduler.num_train_timesteps
            if num_train_timesteps <= 100:
                return 4
            elif num_train_timesteps <= 1000:
                return 20
            else:
                return 50
    
    return None


def get_default_dimensions(pipeline) -> Dict[str, int] | None:
    """
    Extract default height and width from pipeline configuration.
    Returns dict with 'height' and 'width' or None if not found.
    """
    # Try multiple methods to find default dimensions
    height = None
    width = None
    
    # Method 1: Check VAE config (common for most models)
    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "config"):
        vae_config = pipeline.vae.config
        if hasattr(vae_config, "sample_size"):
            size = vae_config.sample_size
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                height, width = size[0], size[1]
            elif isinstance(size, int):
                height = width = size
        elif hasattr(vae_config, "latent_size"):
            size = vae_config.latent_size
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                height, width = size[0], size[1]
            elif isinstance(size, int):
                height = width = size
    
    # Method 2: Check UNet config
    if (height is None or width is None) and hasattr(pipeline, "unet") and hasattr(pipeline.unet, "config"):
        unet_config = pipeline.unet.config
        if hasattr(unet_config, "sample_size"):
            size = unet_config.sample_size
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                height, width = size[0], size[1]
            elif isinstance(size, int):
                height = width = size
    
    # Method 3: Check pipeline config directly
    if (height is None or width is None) and hasattr(pipeline, "config"):
        config = pipeline.config
        if hasattr(config, "sample_size"):
            size = config.sample_size
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                height, width = size[0], size[1]
            elif isinstance(size, int):
                height = width = size
    
    # Method 4: Check transformer config (for Z-Image models)
    if (height is None or width is None) and hasattr(pipeline, "transformer") and hasattr(pipeline.transformer, "config"):
        transformer_config = pipeline.transformer.config
        if hasattr(transformer_config, "sample_size"):
            size = transformer_config.sample_size
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                height, width = size[0], size[1]
            elif isinstance(size, int):
                height = width = size
    
    # Convert latent size to image size if needed (VAE typically has 8x downsampling)
    if height and width and hasattr(pipeline, "vae") and hasattr(pipeline.vae, "config"):
        vae_config = pipeline.vae.config
        if hasattr(vae_config, "scaling_factor") or hasattr(vae_config, "scale_factor"):
            # Most VAEs have 8x downsampling, so multiply by 8
            scale_factor = getattr(vae_config, "scaling_factor", getattr(vae_config, "scale_factor", 8))
            height = int(height * scale_factor)
            width = int(width * scale_factor)
    
    # Ensure dimensions are divisible by 8 (required by diffusion models)
    if height and width:
        # If not divisible by 8, multiply by 8
        if height % 8 != 0:
            height = height * 8
        if width % 8 != 0:
            width = width * 8
        return {"height": height, "width": width}
    return None


def load_pipeline(model_name: str) -> torch.nn.Module:
    """
    Load (or reuse) a pipeline for the given model.
    """
    global pipe, current_model_name

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}")

    # Reuse if already loaded
    if current_model_name == model_name and pipe is not None:
        return pipe

    config = MODEL_CONFIGS[model_name]
    repo_id = config["repo_id"]
    model_dir = config["local_dir"]

    model_path = ensure_model_downloaded(repo_id, model_dir)

    # Release the old pipeline
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()

    print(f"Loading pipeline for model '{model_name}' from {model_path}")

    try:
        # Get token for authentication (needed for gated models)
        token = get_hf_token()
        token_kwargs = {"token": token} if token else {}
        
        # Logic: use specific pipeline types based on model configuration
        pipeline_type = config.get("pipeline_type", "")
        if pipeline_type == "z_image":
            new_pipe = ZImagePipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                low_cpu_mem_usage=True,
                **token_kwargs,
            )
            new_pipe.to("cuda")
        elif pipeline_type == "sd3":
            new_pipe = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                low_cpu_mem_usage=True,
                **token_kwargs,
            )
            new_pipe.to("cuda")
        else:
            new_pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                low_cpu_mem_usage=True,
                device_map="cuda",
                **token_kwargs,
            )

        # Check for transformer component
        if hasattr(new_pipe, "transformer"):
            print("Transformer component is available.")
        else:
            print("Transformer component is missing!")

        # Log default dimensions and steps
        default_dims = get_default_dimensions(new_pipe)
        if default_dims:
            print(f"Default dimensions: {default_dims['width']}x{default_dims['height']}")
        default_steps = get_default_steps(new_pipe)
        if default_steps:
            print(f"Default steps: {default_steps}")

        pipe = new_pipe
        current_model_name = model_name
        return pipe

    except Exception as e:
        print(f"Error loading model pipeline for '{model_name}': {e}")
        raise e


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


def get_gpu_shared_memory():
    """
    Get GPU shared memory (system RAM used by GPU) on Windows using Python only.
    Uses ctypes to access Windows PDH API for performance counters.
    Returns dict with shared_mem_used_GB and shared_mem_total_GB.
    """
    try:
        import platform
        if platform.system() != "Windows":
            return None
        
        import ctypes
        from ctypes import wintypes
        
        # Use PDH (Performance Data Helper) API to get shared memory usage
        pdh = ctypes.windll.pdh
        PDH_FMT_DOUBLE = 0x00000200
        
        shared_used_bytes = 0.0
        
        # Try to enumerate GPU Engine instances and get shared usage
        try:
            # First, get the size needed for the buffer
            instance_buffer_size = ctypes.c_ulong(0)
            counter_buffer_size = ctypes.c_ulong(0)
            
            pdh.PdhEnumObjectItemsW(
                None, None, "GPU Engine",
                None, ctypes.byref(instance_buffer_size),
                None, ctypes.byref(counter_buffer_size),
                ctypes.wintypes.DWORD(0x00000002 | 0x00000001), 0
            )
            
            if instance_buffer_size.value > 0:
                # Allocate buffers
                instance_buffer = (ctypes.c_wchar * instance_buffer_size.value)()
                counter_buffer = (ctypes.c_wchar * counter_buffer_size.value)()
                instance_size = ctypes.c_ulong(instance_buffer_size.value)
                counter_size = ctypes.c_ulong(counter_buffer_size.value)
                
                if pdh.PdhEnumObjectItemsW(
                    None, None, "GPU Engine",
                    instance_buffer, ctypes.byref(instance_size),
                    counter_buffer, ctypes.byref(counter_size),
                    ctypes.wintypes.DWORD(0x00000002 | 0x00000001), 0
                ) == 0:
                    # Parse instances
                    instances_str = instance_buffer.value
                    if instances_str:
                        instances = [i for i in instances_str.split('\0') if i and 'engtype' in i]
                        
                        # Query each instance for shared usage
                        for instance in instances:
                            counter_path = f"\\GPU Engine({instance})\\Shared Usage"
                            try:
                                hQuery = wintypes.HANDLE()
                                if pdh.PdhOpenQueryW(None, 0, ctypes.byref(hQuery)) == 0:
                                    hCounter = wintypes.HANDLE()
                                    path_bytes = counter_path.encode('utf-16le')
                                    if pdh.PdhAddCounterW(hQuery, path_bytes, 0, ctypes.byref(hCounter)) == 0:
                                        # Collect data (PDH requires two samples)
                                        pdh.PdhCollectQueryData(hQuery)
                                        time.sleep(0.1)
                                        pdh.PdhCollectQueryData(hQuery)
                                        
                                        # Get formatted value
                                        fmtValue = ctypes.c_double()
                                        if pdh.PdhGetFormattedCounterValue(hCounter, PDH_FMT_DOUBLE, None, ctypes.byref(fmtValue)) == 0:
                                            if fmtValue.value > 0:
                                                shared_used_bytes += fmtValue.value
                                        
                                        pdh.PdhRemoveCounter(hCounter)
                                    pdh.PdhCloseQuery(hQuery)
                            except Exception:
                                continue
        except Exception as e:
            print(f"Error enumerating GPU counters: {e}")
        
        # Get total shared memory - estimate from system RAM
        # Windows typically allows up to 50% of system RAM for shared GPU memory
        # But we try to get actual value if possible
        ram = psutil.virtual_memory()
        # Default: half of system RAM (Windows standard)
        shared_total_bytes = ram.total // 2
        
        # Try to get more accurate value from registry (NVIDIA specific)
        try:
            import winreg
            try:
                # Try NVIDIA registry path
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000"
                )
                try:
                    # SharedSystemMemory might be stored here
                    value, _ = winreg.QueryValueEx(key, "SharedSystemMemory")
                    if value and value > 0:
                        shared_total_bytes = value
                except FileNotFoundError:
                    pass
                finally:
                    winreg.CloseKey(key)
            except Exception:
                pass
        except ImportError:
            pass
        
        if shared_total_bytes > 0:
            shared_used_gb = round(shared_used_bytes / (1024 ** 3), 2)
            shared_total_gb = round(shared_total_bytes / (1024 ** 3), 2)
            shared_percent = round(shared_used_gb / shared_total_gb * 100, 1) if shared_total_gb > 0 else 0.0
            
            return {
                "shared_mem_used_GB": shared_used_gb,
                "shared_mem_total_GB": shared_total_gb,
                "shared_mem_percent": shared_percent
            }
        
        return None
        
    except Exception as e:
        print(f"Error getting GPU shared memory: {e}")
        return None


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

    # Get shared memory info (same for all GPUs typically)
    shared_mem_info = get_gpu_shared_memory()

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

        gpu_dict = {
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
        
        # Add shared memory info if available
        if shared_mem_info:
            gpu_dict.update(shared_mem_info)
        
        gpus.append(gpu_dict)
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
# Helper functions for default dimensions
# ---------------------------

def get_model_default_dimensions(model_name: str) -> Dict[str, int]:
    """
    Get default dimensions for a model.
    First checks MODEL_CONFIGS, then tries to get from pipeline.
    Falls back to 512x512 if not found.
    """
    # First, check if model has default dimensions in config
    if model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_name]
        if "default_height" in config and "default_width" in config:
            return {
                "height": config["default_height"],
                "width": config["default_width"],
            }
    
    # Check if pipeline is already loaded for this model
    global pipe, current_model_name
    
    if current_model_name is not None and current_model_name == model_name and pipe is not None:
        try:
            dims = get_default_dimensions(pipe)
            if dims:
                return dims
        except Exception as e:
            print(f"Error getting dimensions from loaded pipeline: {e}")
    
    # Try to load pipeline to get dimensions (may be slow on first call)
    try:
        pipeline = load_pipeline(model_name)
        dims = get_default_dimensions(pipeline)
        if dims:
            return dims
    except Exception as e:
        print(f"Could not get default dimensions for {model_name}: {e}")
    
    # Fallback to 512x512
    return {"height": 512, "width": 512}


def get_model_default_steps(model_name: str) -> int:
    """
    Get default number of steps for a model.
    First checks MODEL_CONFIGS, then tries to get from pipeline.
    Falls back to 4 if not found.
    """
    # First, check if model has default steps in config
    if model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_name]
        if "default_steps" in config:
            return config["default_steps"]
    
    # Check if pipeline is already loaded for this model
    global pipe, current_model_name
    
    if current_model_name is not None and current_model_name == model_name and pipe is not None:
        try:
            steps = get_default_steps(pipe)
            if steps:
                return steps
        except Exception as e:
            print(f"Error getting steps from loaded pipeline: {e}")
    
    # Try to load pipeline to get steps (may be slow on first call)
    try:
        pipeline = load_pipeline(model_name)
        steps = get_default_steps(pipeline)
        if steps:
            return steps
    except Exception as e:
        print(f"Could not get default steps for {model_name}: {e}")
    
    # Fallback to 4
    return 4


# ---------------------------
# Flask routes
# ---------------------------

@app.route("/", methods=["GET"])
def index():
    """
    Main page: show form + last generated images (none by default).
    """
    models = sorted(MODEL_CONFIGS.keys())
    default_dims = get_model_default_dimensions(DEFAULT_MODEL_NAME)
    default_steps = get_model_default_steps(DEFAULT_MODEL_NAME)
    return render_template(
        "index.html",
        models=models,
        selected_model=DEFAULT_MODEL_NAME,
        images=[],
        last_prompt="",
        last_steps=default_steps,
        last_height=default_dims["height"],
        last_width=default_dims["width"],
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

    # Ensure dimensions are divisible by 8 (required by diffusion models)
    height = int(height)
    width = int(width)
    if height % 8 != 0:
        height = height * 8
    if width % 8 != 0:
        width = width * 8

    # Use a time-based seed if not controlled externally
    base_seed = int(time.time() * 1000) % 1_000_000_000

    for i in range(num_images):
        image_seed = base_seed + i
        generator = torch.Generator("cuda").manual_seed(image_seed)

        # Determine guidance scale based on model type
        # Turbo models use 0.0, SD3 models use 3.5, regular models use 7.5-9.0
        config = MODEL_CONFIGS.get(model_name, {})
        pipeline_type = config.get("pipeline_type", "")
        repo_id = config.get("repo_id", "").lower()
        is_turbo = "turbo" in model_name.lower() or "turbo" in repo_id
        
        if is_turbo:
            guidance_scale = 0.0
        elif pipeline_type == "sd3":
            guidance_scale = 3.5
        else:
            guidance_scale = 7.5
        
        result = pipeline(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
        )

        image = result.images[0]

        # Convert image to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/png;base64,{image_base64}"
        
        # Generate unique ID for this image
        image_id = uuid.uuid4().hex
        
        # Store base64 in in-memory cache (not in session to avoid cookie size issues)
        image_cache[image_id] = {
            "base64": image_base64,
            "prompt": prompt,
            "model_name": model_name,
            "steps": steps,
            "height": height,
            "width": width,
        }

        images_info.append(
            {
                "image_id": image_id,
                "data_url": image_data_url,
                "view_url": url_for("view_image", image_id=image_id),
                "download_url": url_for("download_image", image_id=image_id),
                "prompt": prompt,
                "model_name": model_name,
                "steps": steps,
                "height": height,
                "width": width,
            }
        )

    return images_info


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    API endpoint to generate images (returns JSON).
    """
    try:
        data = request.get_json()
        model_name = data.get("model_name", DEFAULT_MODEL_NAME)
        custom_model_repo = data.get("custom_model_repo", "").strip()
        prompt = data.get("prompt", "").strip()
        num_images = int(data.get("num_images", "1"))

        if custom_model_repo:
            model_name = register_custom_model(custom_model_repo)

        default_dims = get_model_default_dimensions(model_name)
        default_steps = get_model_default_steps(model_name)
        steps = int(data.get("steps", str(default_steps)))
        height = int(data.get("height", str(default_dims["height"])))
        width = int(data.get("width", str(default_dims["width"])))

        if not prompt:
            return jsonify({"error": "Prompt cannot be empty."}), 400

        images_info = generate_images(
            prompt=prompt,
            steps=steps,
            height=height,
            width=width,
            num_images=num_images,
            model_name=model_name,
        )

        return jsonify({
            "success": True,
            "images": images_info,
            "model_name": model_name,
            "prompt": prompt,
            "steps": steps,
            "height": height,
            "width": width,
            "num_images": num_images,
        })

    except Exception as e:
        error_message = str(e)
        print(f"Error during generation: {error_message}")
        return jsonify({"error": error_message}), 500


@app.route("/generate", methods=["POST"])
def generate():
    """
    Handle form submission to generate images (legacy support).
    """
    error_message = None

    try:
        model_name = request.form.get("model_name", DEFAULT_MODEL_NAME)
        custom_model_repo = request.form.get("custom_model_repo", "").strip()
        prompt = request.form.get("prompt", "").strip()

        num_images = int(request.form.get("num_images", "1"))

        if custom_model_repo:
            # If a custom repo is given, register and use it
            model_name = register_custom_model(custom_model_repo)

        # Get default values for the selected model
        default_dims = get_model_default_dimensions(model_name)
        default_steps = get_model_default_steps(model_name)
        steps = int(request.form.get("steps", str(default_steps)))
        height = int(request.form.get("height", str(default_dims["height"])))
        width = int(request.form.get("width", str(default_dims["width"])))

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
        default_dims = get_model_default_dimensions(DEFAULT_MODEL_NAME)
        default_steps = get_model_default_steps(DEFAULT_MODEL_NAME)
        return render_template(
            "index.html",
            models=models,
            selected_model=DEFAULT_MODEL_NAME,
            images=[],
            last_prompt="",
            last_steps=default_steps,
            last_height=default_dims["height"],
            last_width=default_dims["width"],
            last_num_images=1,
            custom_model_repo="",
            error_message=error_message,
        )


@app.route("/view/<image_id>")
def view_image(image_id: str):
    """
    Display image base64 in a new tab as HTML page.
    """
    if image_id not in image_cache:
        return "Image not found", 404
    
    image_data = image_cache[image_id]
    base64_data = image_data["base64"]
    image_data_url = f"data:image/png;base64,{base64_data}"
    
    # Return HTML page with the image
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Generated Image</title>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                background: #0f172a;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }}
            img {{
                max-width: 100%;
                max-height: 100vh;
                object-fit: contain;
                border-radius: 8px;
                box-shadow: 0 20px 45px rgba(15, 23, 42, 0.9);
            }}
        </style>
    </head>
    <body>
        <img src="{image_data_url}" alt="Generated image">
    </body>
    </html>
    """
    return html


@app.route("/download/<image_id>")
def download_image(image_id: str):
    """
    Convert base64 image to PNG and save to disk, then download it.
    """
    if image_id not in image_cache:
        return "Image not found", 404
    
    image_data = image_cache[image_id]
    base64_data = image_data["base64"]
    
    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_data)
    
    # Generate filename
    filename = f"{image_id}.png"
    filepath = IMAGES_DIR / filename
    
    # Save to disk
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    
    # Clean up cache after download (optional - keeps memory usage down)
    # Uncomment if you want to remove images from cache after download:
    # del image_cache[image_id]
    
    # Return as download
    return Response(
        image_bytes,
        mimetype="image/png",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@app.route("/api/system_stats", methods=["GET"])
def api_system_stats():
    """
    Return system stats as JSON for the frontend.
    """
    stats = get_system_stats()
    return jsonify(stats)


@app.route("/api/register_model", methods=["POST"])
def api_register_model():
    """
    Register a new model from repo_id or HF URL without loading it.
    Returns the model name and updates the model list.
    """
    try:
        data = request.get_json()
        repo_or_url = data.get("repo_or_url", "").strip()
        
        if not repo_or_url:
            return jsonify({"error": "Repository URL or repo_id cannot be empty"}), 400
        
        model_name = register_custom_model(repo_or_url)
        
        return jsonify({
            "success": True,
            "model_name": model_name,
            "repo_id": MODEL_CONFIGS[model_name]["repo_id"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model_info/<model_name>", methods=["GET"])
def api_model_info(model_name: str):
    """
    Return model information including default dimensions and steps.
    """
    try:
        if model_name not in MODEL_CONFIGS:
            return jsonify({"error": f"Unknown model: {model_name}"}), 404
        
        # Load pipeline to get dimensions and steps
        pipeline = load_pipeline(model_name)
        default_dims = get_default_dimensions(pipeline)
        default_steps = get_default_steps(pipeline)
        
        info = {
            "model_name": model_name,
            "repo_id": MODEL_CONFIGS[model_name]["repo_id"],
        }
        
        if default_dims:
            info["default_dimensions"] = default_dims
        else:
            info["default_dimensions"] = None
            info["note"] = "Could not determine default dimensions from model config"
        
        if default_steps:
            info["default_steps"] = default_steps
        else:
            info["default_steps"] = 8  # Fallback
        
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/hf_token", methods=["GET", "POST"])
def api_hf_token():
    """
    GET: Return the current Hugging Face token (masked for security).
    POST: Save a new Hugging Face token.
    """
    if request.method == "GET":
        token = load_hf_token()
        if token:
            # Return masked token for display (show first 4 and last 4 characters)
            masked_token = token[:4] + "*" * (len(token) - 8) + token[-4:] if len(token) > 8 else "****"
            return jsonify({"token": masked_token, "has_token": True})
        return jsonify({"token": None, "has_token": False})
    
    elif request.method == "POST":
        try:
            data = request.get_json()
            token = data.get("token", "").strip()
            
            if not token:
                return jsonify({"error": "Token cannot be empty"}), 400
            
            if save_hf_token(token):
                return jsonify({"success": True, "message": "Token saved successfully"})
            else:
                return jsonify({"error": "Failed to save token"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/api/prompts", methods=["GET"])
def api_prompts():
    """
    Return the list of predefined prompts from prompts.json.
    """
    prompts_file = BASE_DIR / "static" / "prompts.json"
    try:
        if prompts_file.exists():
            with open(prompts_file, "r", encoding="utf-8") as f:
                prompts = json.load(f)
                return jsonify({"success": True, "prompts": prompts})
        else:
            return jsonify({"success": False, "error": "Prompts file not found", "prompts": []}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "prompts": []}), 500


if __name__ == "__main__":
    # Load token at startup
    token = get_hf_token()
    if token:
        print("Hugging Face token loaded from config.json")
    else:
        print("No Hugging Face token found. Some gated models may require authentication.")
    
    # For development only; use a proper WSGI server in production
    app.run(host="127.0.0.1", port=7860, debug=True)
