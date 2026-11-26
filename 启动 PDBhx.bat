@echo off
cd /d "%~dp0"
where pythonw >nul 2>nul
if errorlevel 1 (
    python main.py
    if errorlevel 1 pause
) else (
    start "" pythonw main.py
)
