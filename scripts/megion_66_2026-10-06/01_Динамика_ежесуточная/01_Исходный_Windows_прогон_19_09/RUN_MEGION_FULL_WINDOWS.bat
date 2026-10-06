@echo off
setlocal EnableExtensions
cd /d "%~dp0"
where py >nul 2>nul || (echo ERROR: Install Python 3.11 or 3.12 x64.& pause & exit /b 1)
py -3 -c "import struct,sys; assert struct.calcsize('P')==8; print(sys.version)" || (echo ERROR: Python must be x64.& pause & exit /b 1)
if not exist "deps\pe_2_main.dll" (echo ERROR: deps\pe_2_main.dll is missing.& pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv || (pause & exit /b 1)
call ".venv\Scripts\activate.bat" || (pause & exit /b 1)
python -m pip install --upgrade pip
python -m pip install -r requirements_windows_py311.txt
python -m pip install "deps\UniflocPy-1.3.25-py3-none-any.whl"
python preflight_megion.py || (echo ERROR: Input preflight failed. & pause & exit /b 1)
python run_megion_full.py
set CODE=%ERRORLEVEL%
if not "%CODE%"=="0" (echo ERROR: Calculation stopped with code %CODE%. Check newest RUN_...\logs. & pause & exit /b %CODE%)
echo.
echo DONE. Open the newest RUN_...\final\final_dataset__WITH_H2S_GAS_PHASE.csv
pause
