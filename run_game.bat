@echo off
cd /d %~dp0

rem Uu tien .venv311 (Python 3.11) neu co, neu khong dung .venv hien tai.
if exist .venv311\Scripts\python.exe (
    set "PYEXE=.venv311\Scripts\python.exe"
) else if exist .venv\Scripts\python.exe (
    set "PYEXE=.venv\Scripts\python.exe"
) else (
    echo [ERROR] Khong tim thay moi truong .venv311 hoac .venv.
    echo Hay chay setup_training_env.bat truoc.
    pause
    exit /b 1
)

echo [INFO] Dung Python: %PYEXE%
"%PYEXE%" game.py
pause
