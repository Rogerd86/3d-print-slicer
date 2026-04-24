@echo off
echo ============================================
echo   3D Print Slicer — Quick Install
echo ============================================
echo.

REM Check Python
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Please install Python 3.10+ from python.org
    echo Make sure to tick "Add Python to PATH" during install!
    pause
    exit /b 1
)

echo Installing dependencies...
echo.
python -m pip install --upgrade pip

echo Installing core dependencies...
python -m pip install -r requirements.txt

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo To run the app:
echo   python main.py
echo.
echo Or double-click run_windows.bat
echo.
pause
