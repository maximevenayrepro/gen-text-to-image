@echo off
setlocal

REM ================================================================
REM 0) Go to script directory
REM ================================================================
cd /d "%~dp0"

REM ================================================================
REM 1) Create virtual environment if missing
REM ================================================================
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -m venv .venv
)

REM ================================================================
REM 2) Activate virtual environment
REM ================================================================
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM ================================================================
REM 3) Upgrade pip
REM ================================================================
echo Upgrading pip...
python -m pip install --upgrade pip

REM ================================================================
REM 4) Clean possible conflicting packages
REM ================================================================
REM echo Removing old torch / torchvision / torchaudio / diffusers / huggingface-hub / accelerate...
REM python -m pip uninstall -y torch torchvision torchaudio diffusers huggingface-hub accelerate
REM python -m pip uninstall transformers

REM ================================================================
REM 5) Install base dependencies (WITHOUT torch / accelerate)
REM ================================================================
echo Installing base dependencies...
python -m pip install ^
  flask==3.0.3 ^
  transformers>=4.47.0 ^
  psutil ^
  nvidia-ml-py ^
  pillow ^
  numpy ^
  requests ^
  safetensors

REM ================================================================
REM 6) Install diffusers from GitHub (for ZImagePipeline)
REM ================================================================
echo Installing diffusers from GitHub...
python -m pip install git+https://github.com/huggingface/diffusers.git

REM ================================================================
REM 7) Install PyTorch 2.9.1 with CUDA 13.0 (test channel cu130)
REM ================================================================
echo Installing PyTorch 2.9.1 (CUDA 13 test build)...
python -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/test/cu130

REM ================================================================
REM 8) Install accelerate AFTER torch (no resolver warning)
REM ================================================================
echo Installing accelerate...
python -m pip install accelerate

REM ================================================================
REM 9) Check CUDA availability
REM ================================================================
echo Checking CUDA detection...
python -c "import torch; print('--------------------------------------------------------'); print('torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('torch.version.cuda:', torch.version.cuda); print('--------------------------------------------------------')"

REM ================================================================
REM 10) Run Flask app
REM ================================================================
echo Running app.py...
python app.py

echo.
echo Script finished. Press a key to close.
pause >nul
endlocal
