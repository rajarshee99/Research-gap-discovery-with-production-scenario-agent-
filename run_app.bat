@echo off
setlocal

cd /d "%~dp0"

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "APP_FILE=%~dp0app.py"

if not exist "%VENV_PYTHON%" (
    echo Virtual environment Python was not found at:
    echo %VENV_PYTHON%
    echo.
    echo Create the venv first, then try again.
    pause
    exit /b 1
)

if not exist "%APP_FILE%" (
    echo app.py was not found at:
    echo %APP_FILE%
    pause
    exit /b 1
)

echo Starting Streamlit app...
echo.
"%VENV_PYTHON%" -m streamlit run "%APP_FILE%"

if errorlevel 1 (
    echo.
    echo The app exited with an error.
    pause
)

endlocal
