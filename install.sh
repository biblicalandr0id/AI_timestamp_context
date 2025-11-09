#!/bin/bash
# AI Timestamp Context - Easy Installation Script
# Supports: Linux, macOS, Windows (WSL)

set -e

echo "=========================================="
echo "🧠 AI Chatbot Installation Script"
echo "=========================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3.8 or later:"
    echo "  - Linux/Mac: sudo apt install python3 python3-pip"
    echo "  - Windows: Download from python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed!"
    echo "Installing pip..."
    python3 -m ensurepip --upgrade || {
        echo "Please install pip manually"
        exit 1
    }
fi

echo "✓ Found pip"

# Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate 2>/dev/null || {
    echo "⚠️  Could not activate venv, proceeding without it"
}

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install core dependencies first (faster)
echo ""
echo "📦 Installing core dependencies..."
pip install numpy networkx matplotlib pandas python-dateutil pyyaml psutil schedule

# Install ML dependencies (may take longer)
echo ""
echo "🤖 Installing AI/ML dependencies..."
echo "   (This may take several minutes...)"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers accelerate

# Install web framework
echo ""
echo "🌐 Installing web framework..."
pip install flask flask-cors flask-socketio python-socketio eventlet

# Install testing
echo ""
echo "🧪 Installing testing framework..."
pip install pytest pytest-cov

# Make launcher executable
chmod +x launch_chatbot.py

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "🚀 Quick Start:"
echo ""
echo "   # Web Interface (Recommended)"
echo "   python launch_chatbot.py server"
echo ""
echo "   # Command Line"
echo "   python launch_chatbot.py cli"
echo ""
echo "   # View all options"
echo "   python launch_chatbot.py --help"
echo ""
echo "📖 Documentation:"
echo "   - README.md: System overview"
echo "   - CHATBOT.md: Chatbot guide"
echo "   - FEATURES.md: All features"
echo ""
echo "🎉 Happy Chatting!"
echo "=========================================="
