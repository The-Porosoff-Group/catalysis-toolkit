@echo off
title Catalysis Data Toolkit - DEBUG MODE
echo.
echo  ============================================
echo   DEBUG MODE - Window will stay open
echo  ============================================
echo.

echo  Step 1: Checking conda...
where conda
echo.

echo  Step 2: Current directory is:
echo  %CD%
echo.

echo  Step 3: Trying conda activate...
set "CONDA_ENV=%~dp0.conda_env"
echo  Env path: "%CONDA_ENV%"
echo.

call conda activate "%CONDA_ENV%"
echo  Activate result: %errorlevel%
echo.

echo  Step 4: Checking which python...
where python
echo.

echo  Step 5: Checking GSAS-II...
python -c "exec('try:\n import GSASII.GSASIIscriptable as G\n print(\"GSAS-II OK (new-style)\")\nexcept ImportError:\n import GSASIIscriptable as G\n print(\"GSAS-II OK (legacy)\")')"
echo.

echo  Step 6: Checking other deps...
python -c "import flask; print('flask OK')"
python -c "import numpy; print('numpy OK')"
python -c "import pymatgen; print('pymatgen OK')"
echo.

echo  Step 7: Starting app...
python app.py

echo.
echo  ============================================
echo  DONE - Check messages above for errors
echo  ============================================
echo.
pause
