@echo off
title Catalysis Data Toolkit
echo.
echo  ============================================
echo   Catalysis Data Toolkit
echo   Starting local server...
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

:: Install dependencies if needed
echo  Checking dependencies...
pip install flask pyyaml numpy pandas matplotlib --quiet

echo.
echo  Opening browser at http://localhost:5000
echo  (Close this window to stop the server)
echo.

python app.py
pause
