@echo off
title Catalysis Data Toolkit
echo.
echo  ============================================
echo   Catalysis Data Toolkit
echo  ============================================
echo.

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo  Please install Python from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b
)

:: Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\activate.bat" (
    echo  Creating virtual environment in .venv\ ...
    echo  ^(This is a one-time setup - takes 1-2 minutes^)
    echo.
    python -m venv .venv
    if errorlevel 1 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b
    )
    echo  Virtual environment created.
    echo.
)

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Install / upgrade dependencies if any are missing
python -c "import flask, yaml, numpy, scipy, matplotlib, requests, pymatgen, pandas" >nul 2>&1
if errorlevel 1 (
    echo  Installing dependencies into .venv\
    echo  ^(First run: pymatgen is ~500 MB - please be patient^)
    echo.
    pip install flask pyyaml numpy pandas matplotlib requests pymatgen --quiet
    if errorlevel 1 (
        echo.
        echo  ERROR: Dependency installation failed.
        echo  Check your internet connection and try again.
        pause
        exit /b
    )
    echo.
    echo  Dependencies installed successfully.
    echo.
)

echo  Starting Catalysis Data Toolkit...
echo  Opening browser at http://localhost:5000
echo.
echo  ^(Close this window to stop the server^)
echo.

python app.py

pause
