@echo off
setlocal
REM PAC2 ground-truth stroke labelling. Usage: label_gt.bat [image]   (default: the three priority images in turn)
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo [ERROR] .venv missing & goto end)
if "%~1"=="" (
  echo === example1 ===
  ".venv\Scripts\python.exe" scripts\gt_label_tool.py example1
  echo === field 2 ===
  ".venv\Scripts\python.exe" scripts\gt_label_tool.py 2.png
  echo === field 1 ===
  ".venv\Scripts\python.exe" scripts\gt_label_tool.py 1.png
) else (
  ".venv\Scripts\python.exe" scripts\gt_label_tool.py %1
)
:end
popd
endlocal
