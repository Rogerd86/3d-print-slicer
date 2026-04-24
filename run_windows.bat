@echo off
setlocal enabledelayedexpansion
echo ============================================
echo   3D Print Slicer  --  Install and Run
echo ============================================

REM =====================================================================
REM  Python preflight
REM =====================================================================
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python not found on PATH.
    echo   1^) Install Python 3.10-3.12 from python.org
    echo   2^) Tick "Add Python to PATH" during install
    echo   3^) Re-run this script
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Python !PYVER! detected

REM Warn if Python 3.13+ -- several deps don't have wheels yet
echo !PYVER! | findstr /r "^3\.1[3-9]" >nul
if not errorlevel 1 (
    echo.
    echo WARNING: Python !PYVER! is newer than the tested range ^(3.10-3.12^).
    echo          Some packages ^(manifold3d, pymeshfix, torchmcubes^) may fail
    echo          to build. If install fails below, install Python 3.12 and
    echo          point this script at it.
    echo.
)

REM =====================================================================
REM  Dependency install
REM =====================================================================
echo.
echo ------------- Installing dependencies -------------
echo.

python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Could not upgrade pip. Check your internet connection.
    pause
    exit /b 1
)

echo Installing core packages...
python -m pip install "numpy>=1.24.0" "PyQt5>=5.15.0" "trimesh>=4.0.0" ^
                      "scipy>=1.10.0" "rtree>=1.0.0" "networkx>=3.0"
if errorlevel 1 (
    echo ERROR: Core package install failed. Scroll up for the real error.
    pause
    exit /b 1
)

echo.
echo Installing 3D viewport ^(PyVista^)...
python -m pip install "pyvista>=0.40.0" "pyvistaqt>=0.11.0" "qtawesome>=1.2.0"
if errorlevel 1 (
    echo ERROR: PyVista install failed. Scroll up for the real error.
    pause
    exit /b 1
)

echo.
echo Installing mesh-repair tools...
python -m pip install "pymeshfix>=0.17.0" "PyOpenGL>=3.1.7"
if errorlevel 1 (
    echo WARNING: pymeshfix failed to install.
    echo          The Deep Repair button will be unavailable,
    echo          but the app will still launch.
)

echo.
echo Installing manifold3d ^(best boolean engine, optional^)...
python -m pip install "manifold3d>=3.4.0"
if errorlevel 1 (
    echo   manifold3d failed -- trying latest without version pin...
    python -m pip install manifold3d --upgrade
    if errorlevel 1 (
        echo.
        echo   NOTE: manifold3d is optional. Without it:
        echo         - Self-intersection cleanup in Print-Ready Repair will be skipped
        echo         - Groove cuts and connector holes fall back to trimesh's auto engine
        echo         - App still works for basic slicing, repair, and export.
        echo.
    )
)

echo.
echo Installing optional extras ^(shapely for 2D geometry^)...
python -m pip install shapely mapbox-earcut --no-build-isolation
if errorlevel 1 (
    echo   ^(Optional extras failed to build -- safe to ignore.^)
)

echo.
echo ============================================
echo   Launching 3D Print Slicer...
echo ============================================
echo.
python main.py
if errorlevel 1 (
    echo.
    echo ============================================
    echo   App exited with an error ^(scroll up^).
    echo ============================================
)
pause
