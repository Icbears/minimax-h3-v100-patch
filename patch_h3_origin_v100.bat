@echo off
setlocal
cd /d "%~dp0"

set "PATCH_SCRIPT=%~dp0patch_h3_origin_v100.py"
call :run_python %*
set "PATCH_RC=%ERRORLEVEL%"
echo.
if not "%V100_PATCH_NO_PAUSE%"=="1" pause
exit /b %PATCH_RC%

:run_python
if defined COMFYUI_PYTHON if exist "%COMFYUI_PYTHON%" (
    "%COMFYUI_PYTHON%" "%PATCH_SCRIPT%" %*
    exit /b %ERRORLEVEL%
)
for %%P in (
    "%~dp0python_embeded\python.exe"
    "%~dp0..\python_embeded\python.exe"
    "%~dp0..\..\python_embeded\python.exe"
    "%~dp0..\..\..\python_embeded\python.exe"
) do if exist "%%~fP" (
    "%%~fP" "%PATCH_SCRIPT%" %*
    exit /b %ERRORLEVEL%
)
where py.exe >nul 2>nul
if not errorlevel 1 (
    py -3 "%PATCH_SCRIPT%" %*
    exit /b %ERRORLEVEL%
)
where python.exe >nul 2>nul
if not errorlevel 1 (
    python "%PATCH_SCRIPT%" %*
    exit /b %ERRORLEVEL%
)
echo ERROR: Python 3 was not found. Set COMFYUI_PYTHON to python.exe and retry.
exit /b 9009
