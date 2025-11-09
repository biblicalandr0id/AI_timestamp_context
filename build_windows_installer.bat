@echo off
REM Build Windows Installer using Inno Setup
REM Prerequisites: Install Inno Setup from https://jrsoftware.org/isdl.php

echo ========================================
echo   AI Chatbot - Windows Installer Builder
echo ========================================
echo.

REM Check if Inno Setup is installed
set INNO_SETUP="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

if not exist %INNO_SETUP% (
    echo ERROR: Inno Setup not found!
    echo.
    echo Please install Inno Setup from:
    echo https://jrsoftware.org/isdl.php
    echo.
    echo Then run this script again.
    pause
    exit /b 1
)

echo [1/3] Checking prerequisites...
echo.

REM Check for required files
if not exist "windows_installer.iss" (
    echo ERROR: windows_installer.iss not found!
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found!
    pause
    exit /b 1
)

echo [2/3] Building installer...
echo.

REM Build the installer
%INNO_SETUP% "windows_installer.iss"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Build complete!
echo.

REM Check if output was created
if exist "installer_output\AI_Chatbot_Setup_3.0.exe" (
    echo ========================================
    echo   SUCCESS!
    echo ========================================
    echo.
    echo Installer created: installer_output\AI_Chatbot_Setup_3.0.exe
    echo.
    echo You can now distribute this installer to Windows users.
    echo.
    echo File size:
    dir "installer_output\AI_Chatbot_Setup_3.0.exe" | find "AI_Chatbot"
    echo.
) else (
    echo WARNING: Installer executable not found in expected location.
    echo Check installer_output directory manually.
)

pause
