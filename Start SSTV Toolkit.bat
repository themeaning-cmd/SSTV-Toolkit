@echo off
if exist "%~dp0sstv\sstv_gui.py" (
    cd /d "%~dp0sstv"
) else (
    cd /d "%~dp0"
)

echo Checking required packages (only installs anything if it's missing)...
py -m pip install --quiet numpy scipy pillow sounddevice
if errorlevel 1 (
    echo.
    echo ============================================
    echo  Couldn't install the required packages.
    echo  See the error above.
    echo ============================================
    pause
    exit /b 1
)

echo Starting SSTV Toolkit...
py sstv_gui.py
if errorlevel 1 (
    echo.
    echo ============================================
    echo  SSTV Toolkit closed with an error - see above.
    echo  Screenshot this and send it back if you're
    echo  not sure what it means.
    echo ============================================
    pause
)
