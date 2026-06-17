@echo off
cd /d %~dp0
if not exist .venv311\Scripts\python.exe (
    echo [ERROR] Chua co moi truong .venv311.
    echo Hay chay setup_training_env.bat truoc.
    pause
    exit /b 1
)

call .venv311\Scripts\activate
python game.py
pause
