@echo off
REM SecureHealth-ML Setup Script for Windows

echo ==================================================
echo SecureHealth-ML Setup
echo ==================================================
echo.

REM Check Python
echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created successfully
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo.

REM Install requirements
echo Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully
echo.

REM Test imports
echo Testing installation...
python -c "import torch; import flwr; import opacus; print('All packages imported successfully')"
if %errorlevel% neq 0 (
    echo Warning: Some packages may not have installed correctly
)
echo.

echo ==================================================
echo Setup Complete!
echo ==================================================
echo.
echo To get started:
echo   1. Activate the virtual environment:
echo      venv\Scripts\activate
echo.
echo   2. Run the simulation:
echo      python simulation.py
echo.
echo   Or start server and clients manually:
echo      Terminal 1: python server.py
echo      Terminal 2: python client.py --client-id 0
echo      Terminal 3: python client.py --client-id 1
echo.
echo For more information, see README.md and QUICKSTART.md
echo ==================================================
echo.
pause
