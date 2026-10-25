@echo off
REM Startup script for Windows

REM Ensure we are in the script's directory
cd /d "%~dp0"

REM Check if python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.10+ to run this project.
    pause
    exit /b 1
)

python run.py
if %errorlevel% neq 0 (
    echo Application exited with an error.
    pause
)
