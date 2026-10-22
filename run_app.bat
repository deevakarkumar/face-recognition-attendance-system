@echo off
TITLE AI Smart Face Recognition Attendance System
echo ========================================================
echo     AI Smart Face Recognition Attendance System
echo ========================================================
echo Starting Streamlit Application...
echo.

IF EXIST "C:\Users\dell pc\anaconda3\python.exe" (
    "C:\Users\dell pc\anaconda3\python.exe" -m streamlit run app.py
) ELSE (
    python -m streamlit run app.py
)

pause
