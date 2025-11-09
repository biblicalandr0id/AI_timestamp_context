@echo off
REM AI Timestamp Context - Windows Installation Script

echo ==========================================
echo 🧠 AI Chatbot Installation Script
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed!
    echo Please install Python 3.8 or later from python.org
    pause
    exit /b 1
)

echo ✓ Found Python
python --version

REM Check pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed!
    echo Installing pip...
    python -m ensurepip --upgrade
)

echo ✓ Found pip

REM Create virtual environment
if not exist "venv" (
    echo.
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✓ Virtual environment created
)

echo.
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install core dependencies
echo.
echo 📦 Installing core dependencies...
pip install numpy networkx matplotlib pandas python-dateutil pyyaml psutil schedule

REM Install ML dependencies
echo.
echo 🤖 Installing AI/ML dependencies...
echo    (This may take several minutes...)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers accelerate

REM Install web framework
echo.
echo 🌐 Installing web framework...
pip install flask flask-cors flask-socketio python-socketio eventlet

REM Install testing
echo.
echo 🧪 Installing testing framework...
pip install pytest pytest-cov

echo.
echo ==========================================
echo ✅ Installation Complete!
echo ==========================================
echo.
echo 🚀 Quick Start:
echo.
echo    # Web Interface (Recommended)
echo    python launch_chatbot.py server
echo.
echo    # Command Line
echo    python launch_chatbot.py cli
echo.
echo    # View all options
echo    python launch_chatbot.py --help
echo.
echo 📖 Documentation:
echo    - README.md: System overview
echo    - CHATBOT.md: Chatbot guide
echo    - FEATURES.md: All features
echo.
echo 🎉 Happy Chatting!
echo ==========================================
echo.
pause
