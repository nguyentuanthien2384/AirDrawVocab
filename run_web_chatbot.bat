@echo off
cd /d %~dp0
if not exist .venv311\Scripts\python.exe (
    echo [ERROR] Chua co moi truong .venv311.
    echo Hay chay setup_training_env.bat truoc.
    pause
    exit /b 1
)

call .venv311\Scripts\activate

powershell -NoProfile -Command "try { $client = New-Object Net.Sockets.TcpClient('127.0.0.1', 8000); $client.Close(); exit 0 } catch { exit 1 }" >nul 2>nul
if %ERRORLEVEL%==0 (
    echo [INFO] Server da dang chay tai http://127.0.0.1:8000
    start "" http://127.0.0.1:8000
    pause
    exit /b 0
)

start "" http://127.0.0.1:8000
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
pause
