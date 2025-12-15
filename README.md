# PyGen – Local Image Generation Studio

A Flask-based web application for generating images locally using Hugging Face diffusion models with GPU acceleration.

## Features

- **Multi-model support**: Z-Image-Turbo, Flux-Schnell, SDXL-Turbo, PixArt-XL (auto-download from Hugging Face)
- **Custom models**: Add any compatible Hugging Face model via URL or repo ID
- **Real-time monitoring**: Live CPU, RAM, and GPU stats (NVIDIA)
- **Web UI**: Modern interface with generation controls and image gallery
- **Batch generation**: Generate multiple images per prompt

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- ~8GB+ VRAM recommended

## Installation

```bash
pip install flask torch diffusers huggingface_hub psutil nvidia-ml-py
```

## Usage

```bash
python app.py
```

Open http://127.0.0.1:7860 in your browser.

## Project Structure

```
├── app.py              # Main Flask application
├── app2.py             # Minimal generation script
├── models/             # Downloaded model weights
├── static/
│   ├── css/            # Stylesheets
│   ├── js/             # Frontend scripts
│   └── generated/      # Output images
└── templates/          # HTML templates
```

## License

MIT

