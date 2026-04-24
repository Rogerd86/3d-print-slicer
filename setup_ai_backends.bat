@echo off
setlocal enabledelayedexpansion
echo ============================================
echo   3D Print Slicer  --  AI Backend Setup
echo ============================================
echo.
echo This installs any of the 4 image-to-3D backends:
echo   - TRELLIS 2        (14-16 GB VRAM, best quality)
echo   - SAM 3D Objects   (16 GB VRAM)
echo   - PartCrafter      (12 GB VRAM, pre-separated parts)
echo   - TripoSR          (6-8 GB VRAM, fastest)
echo.
echo Requirements: NVIDIA GPU, ~10 GB disk, internet.
echo This script will STOP on the first error so you can read it.
echo.
pause

REM =====================================================================
REM  Preflight checks  --  fail fast with a clear message
REM =====================================================================
echo.
echo ------------- PREFLIGHT CHECKS -------------

REM --- Python -------------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL]  Python not found on PATH.
    echo         Install Python 3.10-3.12 from python.org
    echo         ^(3.13 is too new for several AI backends^).
    echo         Tick "Add Python to PATH" during install.
    goto :fail
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK]    Python !PYVER! detected
echo.
echo NOTE: Python 3.13 is NOT supported by all backends yet.
echo       PartCrafter and TRELLIS 2 want Python 3.10-3.12.
echo       If you're on 3.13, expect build failures on torchmcubes etc.

REM --- Git ----------------------------------------------------------
git --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL]  Git not found on PATH.
    echo         Install from https://git-scm.com/download/win
    echo         Use default options; Git must be on PATH.
    goto :fail
)
echo [OK]    Git detected

REM --- NVIDIA driver -----------------------------------------------
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARN]  nvidia-smi not found. If you have an NVIDIA GPU,
    echo         update your driver from nvidia.com/Download.
    echo         Continuing -- install will still download CUDA wheels.
) else (
    echo [OK]    NVIDIA driver detected
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
)

REM --- pip up to date ----------------------------------------------
echo.
echo ------------- UPGRADING PIP -------------
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [FAIL]  Could not upgrade pip. Check your internet connection.
    goto :fail
)

REM --- ai_backends directory ---------------------------------------
set AI_DIR=%~dp0ai_backends
if not exist "%AI_DIR%" mkdir "%AI_DIR%"
cd /d "%AI_DIR%"
echo.
echo Installing into: %AI_DIR%

REM =====================================================================
REM  Step 1: PyTorch with CUDA 12.4
REM =====================================================================
echo.
echo ============================================
echo   Step 1 / 6 : PyTorch (CUDA 12.4)
echo ============================================
echo.
echo The previous version of this script installed CUDA 12.1 wheels,
echo which TRELLIS 2 and PartCrafter don't work with. Using cu124.
echo.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>nul
if errorlevel 1 (
    echo PyTorch not installed -- installing now...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo [FAIL]  PyTorch install failed.
        echo         If you're on Python 3.13, downgrade to 3.12 --
        echo         torch wheels for 3.13 are still rolling out.
        goto :fail
    )
) else (
    echo PyTorch already present. Skipping reinstall.
    echo ^(If you hit CUDA-mismatch errors later, run:
    echo    python -m pip install torch torchvision --upgrade --index-url https://download.pytorch.org/whl/cu124 ^)
)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

REM =====================================================================
REM  Step 2: TripoSR
REM =====================================================================
echo.
echo ============================================
echo   Step 2 / 6 : TripoSR
echo ============================================
echo.
set /p INSTALL_TRI="Install TripoSR (6-8GB VRAM, fastest)? (y/n): "
if /i "%INSTALL_TRI%"=="y" (
    if not exist "TripoSR" (
        echo Cloning TripoSR...
        git clone https://github.com/VAST-AI-Research/TripoSR.git
        if errorlevel 1 (
            echo [FAIL]  git clone for TripoSR failed.
            goto :fail
        )
    ) else (
        echo TripoSR already downloaded.
    )
    cd TripoSR
    echo.
    echo TripoSR needs torchmcubes -- this can fail on Windows with Python 3.13.
    echo If you see a C++ build error below, either:
    echo   a^) Install Visual Studio Build Tools + Windows SDK, OR
    echo   b^) Downgrade to Python 3.12 where a prebuilt wheel exists.
    echo.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [WARN]  TripoSR requirements install failed. Check the error above.
        echo         You can still continue -- other backends don't need this.
    )
    cd ..
) else (
    echo Skipped.
)

REM =====================================================================
REM  Step 3: TRELLIS 2
REM =====================================================================
echo.
echo ============================================
echo   Step 3 / 6 : TRELLIS 2  (recommended for 16GB GPUs)
echo ============================================
echo.
set /p INSTALL_TRELLIS="Install TRELLIS 2? (y/n): "
if /i "%INSTALL_TRELLIS%"=="y" (
    if not exist "TRELLIS.2" (
        echo Cloning TRELLIS 2...
        git clone https://github.com/microsoft/TRELLIS.2.git
        if errorlevel 1 (
            echo [FAIL]  git clone for TRELLIS 2 failed.
            goto :fail
        )
    ) else (
        echo TRELLIS 2 already downloaded.
    )
    cd TRELLIS.2
    echo.
    echo Installing TRELLIS 2 requirements ^(this is slow^)...
    if exist requirements.txt (
        python -m pip install -r requirements.txt
        if errorlevel 1 (
            echo [WARN]  TRELLIS 2 requirements.txt failed.
            echo         Common cause: flash-attn needs VS Build Tools.
            echo         See README for the workaround ^(CPU-friendly attn^).
        )
    ) else (
        echo [WARN]  No requirements.txt in TRELLIS.2 -- falling back to setup.py
    )
    echo Installing TRELLIS 2 package in editable mode...
    python -m pip install -e .
    if errorlevel 1 (
        echo [WARN]  pip install -e . failed. Backend may still work
        echo         via direct import. Check the error above.
    )
    echo.
    echo Model weights download automatically on first generation run.
    cd ..
) else (
    echo Skipped.
)

REM =====================================================================
REM  Step 4: SAM 3D Objects
REM =====================================================================
echo.
echo ============================================
echo   Step 4 / 6 : SAM 3D Objects  (Meta, 16GB VRAM)
echo ============================================
echo.
set /p INSTALL_SAM="Install SAM 3D Objects? (y/n): "
if /i "%INSTALL_SAM%"=="y" (
    if not exist "sam-3d-objects" (
        echo Cloning SAM 3D Objects...
        git clone https://github.com/facebookresearch/sam-3d-objects.git
        if errorlevel 1 (
            echo [FAIL]  git clone for SAM 3D Objects failed.
            goto :fail
        )
    ) else (
        echo SAM 3D Objects already downloaded.
    )
    cd sam-3d-objects
    if exist requirements.txt (
        python -m pip install -r requirements.txt
        if errorlevel 1 (
            echo [WARN]  SAM 3D requirements failed.
        )
    )
    python -m pip install -e ".[inference]"
    if errorlevel 1 (
        python -m pip install -e .
    )
    echo.
    echo NEXT: download model weights from HuggingFace:
    echo   1^) pip install huggingface-hub
    echo   2^) huggingface-cli login
    echo   3^) huggingface-cli download facebook/sam-3d-objects
    cd ..
) else (
    echo Skipped.
)

REM =====================================================================
REM  Step 5: PartCrafter
REM =====================================================================
echo.
echo ============================================
echo   Step 5 / 6 : PartCrafter  (12GB VRAM, pre-separated parts)
echo ============================================
echo.
set /p INSTALL_PC="Install PartCrafter? (y/n): "
if /i "%INSTALL_PC%"=="y" (
    if not exist "PartCrafter" (
        echo Cloning PartCrafter...
        git clone https://github.com/wgsxm/PartCrafter.git
        if errorlevel 1 (
            echo [FAIL]  git clone for PartCrafter failed.
            goto :fail
        )
    ) else (
        echo PartCrafter already downloaded.
    )
    cd PartCrafter
    if exist requirements.txt (
        python -m pip install -r requirements.txt
        if errorlevel 1 (
            echo [WARN]  PartCrafter requirements failed.
            echo         PartCrafter wants Python 3.11 + torch 2.5.1 + CUDA 12.4.
            echo         Mismatched Python / torch is the usual cause here.
        )
    )
    cd ..
) else (
    echo Skipped.
)

REM =====================================================================
REM  Step 6: OpenSCAD  (text-to-3D, CPU only)
REM =====================================================================
echo.
echo ============================================
echo   Step 6 / 6 : OpenSCAD  (optional, text-to-3D)
echo ============================================
echo.
echo OpenSCAD is a separate program -- NOT a pip package.
echo Download and install from: https://openscad.org/downloads.html
echo Once installed, the "Generate 3D from Text" feature will find it
echo automatically. No GPU required.
echo.

REM =====================================================================
REM  Summary
REM =====================================================================
echo.
echo ============================================
echo   Setup finished
echo ============================================
echo.
echo What got installed in %AI_DIR%:
if exist "%AI_DIR%\TripoSR"          echo   [OK] TripoSR
if exist "%AI_DIR%\TRELLIS.2"        echo   [OK] TRELLIS 2
if exist "%AI_DIR%\sam-3d-objects"   echo   [OK] SAM 3D Objects
if exist "%AI_DIR%\PartCrafter"      echo   [OK] PartCrafter
echo.
echo If anything above said [WARN] or [FAIL], scroll up to see
echo the real error message. The most common causes are:
echo   - Python 3.13 is too new  ^(downgrade to 3.12^)
echo   - No Visual Studio Build Tools for native wheels
echo   - CUDA 12.1 vs 12.4 mismatch  ^(this script now uses cu124^)
echo   - First-time HuggingFace download hit rate limit  ^(retry^)
echo.
pause
exit /b 0

:fail
echo.
echo ============================================
echo   SETUP STOPPED
echo ============================================
echo.
echo Fix the error above, then re-run this script.
echo Nothing was left in a bad state -- it's safe to run again.
echo.
pause
exit /b 1
