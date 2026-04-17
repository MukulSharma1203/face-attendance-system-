@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m streamlit run attendance_web.py
endlocal
