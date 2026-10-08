@echo off
setlocal

cd /d "%~dp0"

echo [1/5] Checking Python 3.12...
set "PY_VERSION="
for /f "tokens=*" %%V in ('py -3.12 --version 2^>^&1') do set "PY_VERSION=%%V"
echo %PY_VERSION%
echo %PY_VERSION% | findstr /C:"Python 3.12" >nul
if errorlevel 1 (
    echo ERROR: Python 3.12 was not found.
    echo Please install Python 3.12 or make sure "py -3.12" works, then run this file again.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [2/5] Creating virtual environment...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create the virtual environment.
        exit /b 1
    )
) else (
    echo [2/5] Reusing existing virtual environment.
)

echo [3/5] Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to update pip.
    exit /b 1
)

echo [4/5] Installing requirements.txt...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.txt.
    exit /b 1
)

echo [5/5] Setup complete.
echo.
echo Success: PAC environment is ready.
echo To activate the virtual environment, run:
echo .venv\Scripts\activate
echo.
echo To run Python with this environment, use:
echo .venv\Scripts\python.exe

endlocal
