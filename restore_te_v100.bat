@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "PATCH_SCRIPT=%~dp0patch_te_v100.py"
set "DEFAULT_COMFYUI_INSTALL=C:\Users\Administrator\ComfyUI-Installs\ComfyUI"
set "DEFAULT_MODEL_FILE=C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py"

if not exist "%PATCH_SCRIPT%" (
    echo ERROR: Required file not found: "%PATCH_SCRIPT%"
    echo Extract the complete patcher folder before running this BAT file.
    set "RESTORE_RC=2"
    goto :finish
)

if "%~1"=="" (
    call :run_python --model-file "%DEFAULT_MODEL_FILE%" --revert
) else (
    call :run_python %* --revert
)
set "RESTORE_RC=%ERRORLEVEL%"
goto :finish

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

:finish
echo.
if not "%V100_RESTORE_NO_PAUSE%"=="1" pause
exit /b %RESTORE_RC%
