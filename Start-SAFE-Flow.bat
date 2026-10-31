@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" set "PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not defined PYTHON_EXE (
  where python >nul 2>nul
  if not errorlevel 1 for /f "delims=" %%P in ('where python') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
)
if not defined PYTHON_EXE (
  echo Python was not found. Install Python 3.10 or newer, then double-click this file again.
  echo During installation, enable "Add Python to PATH".
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo First-time setup: creating the SAFE Flow environment...
  "%PYTHON_EXE%" -m venv .venv
  if errorlevel 1 goto setup_error
)
"%CD%\.venv\Scripts\python.exe" -c "import flask, pandas, sklearn, joblib" >nul 2>nul
if errorlevel 1 (
  echo Installing SAFE Flow dependencies for first use...
  "%CD%\.venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto setup_error
)
echo Starting SAFE Flow at http://127.0.0.1:5000
start "" http://127.0.0.1:5000
"%CD%\.venv\Scripts\python.exe" app.py
goto end
:setup_error
echo Setup did not finish. Install Python 3.10+ with PATH enabled and try again.
pause
:end
endlocal
