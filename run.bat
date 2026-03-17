@echo off
title Catalysis Data Toolkit
echo.
echo  ============================================
echo   Catalysis Data Toolkit
echo  ============================================
echo.

:: ── Try conda first (needed for GSAS-II) ────────────────────────────────
:: Check if conda is available
where conda >nul 2>&1
if errorlevel 1 goto :USE_VENV

:: ── CONDA PATH ──────────────────────────────────────────────────────────
echo  [Conda detected - using conda environment]
echo.

:: Get full path to this directory (handles spaces)
set "TOOLKIT_DIR=%~dp0"
set "CONDA_ENV=%TOOLKIT_DIR%.conda_env"

:: Create conda environment if it doesn't exist
if not exist "%CONDA_ENV%\python.exe" (
    if not exist "%CONDA_ENV%\Scripts\python.exe" (
        echo  Creating conda environment...
        echo  ^(First-time setup - this takes a few minutes^)
        echo.
        conda create -p "%CONDA_ENV%" python=3.11 -y -q
        if errorlevel 1 (
            echo  ERROR: Failed to create conda environment.
            echo  Falling back to venv...
            goto :USE_VENV
        )
        echo  Conda environment created.
        echo.
    )
)

:: Activate conda environment
call conda activate "%CONDA_ENV%"
if errorlevel 1 (
    echo  ERROR: Failed to activate conda environment.
    echo  Try running: conda init cmd.exe
    echo  Then close and reopen this window.
    echo  Falling back to venv...
    goto :USE_VENV
)

:: ── Check if GSAS-II is already installed ───────────────────────────────
python -c "import GSASII.GSASIIscriptable" >nul 2>&1
if not errorlevel 1 goto :GSAS_DONE
python -c "import GSASIIscriptable" >nul 2>&1
if not errorlevel 1 goto :GSAS_DONE

:: ── GSAS-II not found — try to install ──────────────────────────────────
echo  Installing GSAS-II ^(one-time, may take several minutes^)...
echo.

:: Try conda package gsas2pkg first (replaces old gsas2full)
echo  Trying: conda install gsas2pkg -c briantoby ...
conda install gsas2pkg -c briantoby -y
if errorlevel 1 goto :GSAS_TRY_GIT

:: Verify conda install worked
python -c "import GSASIIscriptable" >nul 2>&1
if not errorlevel 1 (
    echo  GSAS-II installed successfully via conda.
    echo.
    goto :GSAS_DONE
)
python -c "import GSASII.GSASIIscriptable" >nul 2>&1
if not errorlevel 1 (
    echo  GSAS-II installed successfully via conda.
    echo.
    goto :GSAS_DONE
)

:GSAS_TRY_GIT
:: Conda failed — try git clone + pip install
where git >nul 2>&1
if errorlevel 1 (
    echo.
    echo  WARNING: GSAS-II conda install failed and git not found.
    echo  To install GSAS-II manually, run: conda install gsas2pkg -c briantoby
    echo  Or install git from https://git-scm.com/downloads and re-run.
    echo  GSAS-II button will be hidden, but Le Bail and Rietveld still work.
    echo.
    goto :GSAS_DONE
)

echo  Conda package failed, trying GitHub install...
echo.
set "GSAS_CLONE=%TOOLKIT_DIR%.gsas2_src"
if not exist "%GSAS_CLONE%" (
    git clone --depth 1 https://github.com/AdvancedPhotonSource/GSAS-II.git "%GSAS_CLONE%"
)
pip install "%GSAS_CLONE%"

:: Verify git install worked
python -c "import GSASII.GSASIIscriptable" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo  GSAS-II installed successfully via pip.
    echo.
    goto :GSAS_DONE
)

echo.
echo  WARNING: GSAS-II installation failed.
echo  GSAS-II button will be hidden, but Le Bail and Rietveld still work.
echo.

:GSAS_DONE

:: ── Install pip dependencies if missing ─────────────────────────────────
python -c "import flask, yaml, numpy, scipy, matplotlib, requests, pymatgen, pandas" >nul 2>&1
if errorlevel 1 (
    echo  Installing Python dependencies...
    echo  ^(First run: pymatgen is ~500 MB - please be patient^)
    echo.
    pip install flask pyyaml numpy pandas scipy matplotlib requests pymatgen
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

goto :START_APP

:: ── VENV FALLBACK (no conda available) ──────────────────────────────────
:USE_VENV
echo  [Using Python venv - install Miniforge for GSAS-II support]
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
    pip install flask pyyaml numpy pandas scipy matplotlib requests pymatgen --quiet
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

:: ── LAUNCH ──────────────────────────────────────────────────────────────
:START_APP
echo  Starting Catalysis Data Toolkit...
echo  Opening browser at http://localhost:5000
echo.
echo  ^(Close this window to stop the server^)
echo.

python app.py

echo.
echo  ============================================
echo  Server stopped. If this was unexpected,
echo  check the error message above.
echo  ============================================
pause
