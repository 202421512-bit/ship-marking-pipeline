@echo off
setlocal EnableExtensions
REM PAC2 Windows environment bootstrap. Put this file directly in PAC2\.
REM Keep this script in ASCII / CRLF for compatibility with cmd.exe.

pushd "%~dp0"
if errorlevel 1 goto fail_path

echo ======================================================
echo [PAC2] Working directory: %CD%
echo [PAC2] Required input folders: Source\formal, Source\handwritten
 echo ======================================================

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt is missing in this folder.
    goto fail
)

if exist ".venv\Scripts\python.exe" goto have_venv

REM Prefer Python 3.11 via the Windows Python launcher.
py -3.11 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 goto venv_with_launcher

REM Fallback to python on PATH, but only if exactly Python 3.11.
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 goto venv_with_python

echo [ERROR] Python 3.11 was not found.
echo [ACTION] Install Python 3.11, enable the Python launcher, and rerun.
goto fail

:venv_with_launcher
echo [PAC2] Creating .venv with py -3.11 ...
py -3.11 -m venv ".venv"
if errorlevel 1 goto fail_venv
goto have_venv

:venv_with_python
echo [PAC2] Creating .venv with python 3.11 ...
python -m venv ".venv"
if errorlevel 1 goto fail_venv
goto have_venv

:have_venv
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
echo [PAC2] Checking virtual environment Python version ...
"%VENV_PY%" -c "import sys; print(sys.version); sys.exit(0 if sys.version_info[:2] == (3,11) else 1)"
if errorlevel 1 (
    echo [ERROR] Existing .venv is not Python 3.11. It was NOT deleted.
    echo [ACTION] Rename the existing .venv manually if you want a fresh one.
    goto fail
)

echo [PAC2] Updating pip / setuptools / wheel ...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto fail_install

echo [PAC2] Installing requirements. This may take several minutes ...
"%VENV_PY%" -m pip install -r "requirements.txt"
if errorlevel 1 goto fail_install

echo [PAC2] Validating required imports and torch device ...
"%VENV_PY%" -c "import torch, cv2, sklearn, skimage, numpy, pandas; print('torch:',torch.__version__,'cuda_available:',torch.cuda.is_available()); print('opencv:',cv2.__version__,'sklearn:',sklearn.__version__)"
if errorlevel 1 goto fail_install

"%VENV_PY%" -m pip check
if errorlevel 1 goto fail_install

if not exist "Source\formal" (
    echo [WARN] Source\formal not found. Add the 100 formal images there.
) else (
    echo [PAC2] Found Source\formal
)
if not exist "Source\handwritten" (
    echo [WARN] Source\handwritten not found. Add the 100 handwritten images there.
) else (
    echo [PAC2] Found Source\handwritten
)

echo.
echo [SUCCESS] PAC2 environment ready.
echo [NEXT] Select interpreter: .venv\Scripts\python.exe in VS Code.
echo [NEXT] Run data audit after the AI agent creates scripts\01_audit_dataset.py.
popd
exit /b 0

:fail_path
echo [ERROR] Cannot enter the script folder.
exit /b 1
:fail_venv
echo [ERROR] Failed to create virtual environment.
goto fail
:fail_install
echo [ERROR] Installation or verification failed. See the messages above.
goto fail
:fail
popd
exit /b 1
