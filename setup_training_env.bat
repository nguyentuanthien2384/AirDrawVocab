@echo off
cd /d %~dp0
py -3.11 -m venv .venv311
call .venv311\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python check_environment.py
pause
