@echo off
chcp 65001 > nul
set PYTHONUTF8=1
cd /d "%~dp0spam_prototype"
echo ======================================================================
echo    Thai Spam Risk Assessment System
echo    Server starting at http://localhost:5000 ...
echo    (Logs will appear below in real-time)
echo ======================================================================
start http://localhost:5000
..\venv\Scripts\python.exe app.py
pause
