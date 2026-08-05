@echo off
setlocal
set "V100_PATCH_NO_PAUSE=1"

call "%~dp0patch_te_v100.bat" --revert %*
set "RESTORE_RC=%ERRORLEVEL%"
echo.
if not "%V100_RESTORE_NO_PAUSE%"=="1" pause
exit /b %RESTORE_RC%
