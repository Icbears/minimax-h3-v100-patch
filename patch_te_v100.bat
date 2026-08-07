@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "PATCH_SCRIPT=%~dp0patch_te_v100.py"
set "DEFAULT_COMFYUI_INSTALL=C:\Users\Administrator\ComfyUI-Installs\ComfyUI"
set "DEFAULT_MODEL_FILE=C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py"

if "%~1"=="" (
    call :run_python --model-file "%DEFAULT_MODEL_FILE%"
) else if /i "%~1"=="--check" (
    call :run_python --model-file "%DEFAULT_MODEL_FILE%" %*
) else if /i "%~1"=="--revert" (
    call :run_python --model-file "%DEFAULT_MODEL_FILE%" %*
) else (
    call :run_python %*
)
set "PATCH_RC=%ERRORLEVEL%"
echo.
if not "%V100_PATCH_NO_PAUSE%"=="1" pause
exit /b %PATCH_RC%

:run_python
if defined COMFYUI_PYTHON if exist "%COMFYUI_PYTHON%" (
    "%COMFYUI_PYTHON%" "%PATCH_SCRIPT%" %*
    exit /b !ERRORLEVEL!
)
for %%P in (
    "%DEFAULT_COMFYUI_INSTALL%\.venv\Scripts\python.exe"
    "%DEFAULT_COMFYUI_INSTALL%\python_embeded\python.exe"
    "%~dp0python_embeded\python.exe"
    "%~dp0..\python_embeded\python.exe"
    "%~dp0..\..\python_embeded\python.exe"
    "%~dp0..\..\..\python_embeded\python.exe"
) do if exist "%%~fP" (
    "%%~fP" "%PATCH_SCRIPT%" %*
    exit /b !ERRORLEVEL!
)
where py.exe >nul 2>nul
if not errorlevel 1 (
    py -3 "%PATCH_SCRIPT%" %*
    exit /b !ERRORLEVEL!
)
where python.exe >nul 2>nul
if not errorlevel 1 (
    python "%PATCH_SCRIPT%" %*
    exit /b !ERRORLEVEL!
)
echo ERROR: Python 3 was not found. Set COMFYUI_PYTHON to python.exe and retry.
exit /b 9009
