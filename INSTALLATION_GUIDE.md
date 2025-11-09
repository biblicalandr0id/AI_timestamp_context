# 🚀 AI Chatbot - Complete Installation Guide

This guide covers all installation methods for AI Chatbot across different platforms.

---

## 📋 Table of Contents

1. [Quick Install (Recommended)](#quick-install)
2. [Windows Installation](#windows-installation)
3. [macOS Installation](#macos-installation)
4. [Linux Installation](#linux-installation)
5. [Android/Termux Installation](#android-installation)
6. [Docker Installation](#docker-installation)
7. [Manual Installation](#manual-installation)
8. [Troubleshooting](#troubleshooting)

---

## ⚡ Quick Install (Recommended)

### All Platforms - Installation Wizard

The easiest way to install AI Chatbot with a graphical wizard:

```bash
# 1. Clone repository
git clone <repository-url>
cd AI_timestamp_context

# 2. Install PyQt6 (for the wizard)
pip install PyQt6

# 3. Run installation wizard
python installer_wizard.py
```

The wizard will:
- ✅ Check prerequisites
- ✅ Choose installation location
- ✅ Select components
- ✅ Install dependencies automatically
- ✅ Create shortcuts
- ✅ Configure settings

**Time:** 10-20 minutes (depending on internet speed)

---

## 🪟 Windows Installation

### Method 1: Installer (.exe) - Easiest ⭐

**Prerequisites:**
- Python 3.8 or higher ([Download](https://www.python.org/downloads/))
- Windows 10 or 11

**Steps:**

1. **Build the installer** (for developers):
```batch
build_windows_installer.bat
```

This creates `AI_Chatbot_Setup_3.0.exe`

2. **Run the installer:**
   - Double-click `AI_Chatbot_Setup_3.0.exe`
   - Follow the installation wizard
   - Choose installation location
   - Select components
   - Wait for installation (10-15 minutes)
   - Launch from desktop shortcut or Start Menu

3. **Post-installation:**
   - Desktop shortcut created automatically
   - Start Menu entry added
   - Launch with: `AI Chatbot` from Start Menu

### Method 2: Python Installation Wizard

```batch
# Install PyQt6 first
pip install PyQt6

# Run wizard
python installer_wizard.py
```

### Method 3: Manual Script

```batch
# Run the installation script
install.bat

# Launch desktop app
python desktop_app.py
```

**Shortcuts Created:**
- Desktop: `AI Chatbot.lnk`
- Start Menu: `AI Chatbot`

---

## 🍎 macOS Installation

### Method 1: DMG Installer - Easiest ⭐

**Prerequisites:**
- macOS 10.13 (High Sierra) or later
- Python 3.8+ ([Download](https://www.python.org/downloads/mac-osx/))
- Homebrew ([Install](https://brew.sh))

**Steps:**

1. **Build the DMG** (for developers):
```bash
# Install create-dmg
brew install create-dmg

# Build app and DMG
chmod +x build_macos_app.sh
./build_macos_app.sh
```

This creates:
- `AI Chatbot.app` bundle
- `AI_Chatbot_3.0_macOS.dmg` installer

2. **Install from DMG:**
   - Open `AI_Chatbot_3.0_macOS.dmg`
   - Drag `AI Chatbot.app` to Applications folder
   - First launch: Right-click → Open (to bypass Gatekeeper)
   - Run setup: `./Contents/Resources/setup.sh`

3. **Launch:**
   - From Applications: Double-click `AI Chatbot`
   - From Terminal: `open -a "AI Chatbot"`
   - From Command Line: `ai-chatbot`

### Method 2: Homebrew (coming soon)

```bash
brew tap your-org/ai-chatbot
brew install ai-chatbot
```

### Method 3: Manual Installation

```bash
# Clone repository
git clone <repository-url>
cd AI_timestamp_context

# Run installation script
chmod +x install.sh
./install.sh

# Launch
python desktop_app.py
```

---

## 🐧 Linux Installation

### Method 1: Package Manager - Easiest ⭐

#### Debian/Ubuntu (.deb)

**Build package:**
```bash
# Install build tools
sudo apt-get install dpkg-dev

# Build package
chmod +x build_linux_packages.sh
./build_linux_packages.sh deb
```

**Install:**
```bash
sudo dpkg -i build/ai-chatbot_3.0_all.deb
sudo apt-get install -f  # Fix dependencies
```

**Launch:**
```bash
ai-chatbot  # Desktop GUI
# or find in Applications menu
```

#### Fedora/RHEL/CentOS (.rpm)

**Build package:**
```bash
# Install build tools
sudo dnf install rpm-build  # Fedora
# or
sudo yum install rpm-build  # RHEL/CentOS

# Build package
chmod +x build_linux_packages.sh
./build_linux_packages.sh rpm
```

**Install:**
```bash
sudo rpm -i build/ai-chatbot-3.0-1.*.rpm
```

**Launch:**
```bash
ai-chatbot
```

#### Arch Linux (AUR - coming soon)

```bash
yay -S ai-chatbot
```

### Method 2: Manual Installation

```bash
# Clone repository
git clone <repository-url>
cd AI_timestamp_context

# Run installation script
chmod +x install.sh
./install.sh

# Launch
python desktop_app.py
```

### Method 3: Installation Wizard

```bash
# Install PyQt6
pip install PyQt6

# Run wizard
python installer_wizard.py
```

---

## 📱 Android Installation

### Method 1: Termux Installation Wizard - Easiest ⭐

**Prerequisites:**
- Android 7.0 or higher
- Termux app from [F-Droid](https://f-droid.org/en/packages/com.termux/)
- ~5GB free storage

**Steps:**

1. **Install Termux:**
   - Download from F-Droid (NOT Play Store)
   - Open Termux

2. **Run installation wizard:**
```bash
# Download installer
pkg install wget
wget <installer-url>/install_android.sh

# Make executable
chmod +x install_android.sh

# Run wizard
./install_android.sh
```

3. **Follow the wizard:**
   - Updates packages automatically
   - Installs dependencies
   - Sets up Python environment
   - Creates shortcuts
   - **Time:** 15-30 minutes

4. **Launch:**
```bash
# Start web server
cd ~/AI_Chatbot
./start_server.sh

# Open browser
# Navigate to: http://localhost:5000
```

5. **Create home screen widget:**
   - Install "Termux:Widget" from F-Droid
   - Add widget to home screen
   - Tap "AI_Chatbot" to launch instantly

### Method 2: Manual Termux Setup

See [MOBILE.md](MOBILE.md) for detailed manual instructions.

### Features on Android:
- ✅ Full chatbot functionality
- ✅ Web interface (mobile-optimized)
- ✅ CLI interface
- ✅ Background server
- ⚠️ Voice interface (requires additional setup)
- ❌ Vision features (performance intensive)

---

## 🐳 Docker Installation

### Method 1: Docker Compose - Easiest ⭐

**Prerequisites:**
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed

**Steps:**

```bash
# Clone repository
git clone <repository-url>
cd AI_timestamp_context

# Start with Docker Compose
docker-compose up -d

# Access at http://localhost:5000
```

**Management:**
```bash
# Stop
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Update
git pull
docker-compose up -d --build
```

### Method 2: Docker Run

```bash
# Build image
docker build -t ai-chatbot .

# Run container
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  --name ai-chatbot \
  ai-chatbot

# Access at http://localhost:5000
```

### Docker Features:
- ✅ Isolated environment
- ✅ Easy deployment
- ✅ Automatic updates
- ✅ Portable across platforms
- ✅ Data persistence with volumes

---

## 🔧 Manual Installation

For advanced users who want full control:

### Prerequisites:
- Python 3.8 or higher
- pip (Python package manager)
- git
- 5GB free disk space

### Steps:

```bash
# 1. Clone repository
git clone <repository-url>
cd AI_timestamp_context

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Create configuration
cat > config.yaml << EOF
model_name: microsoft/DialoGPT-small
temperature: 0.8
database_path: data/chatbot.db
EOF

# 7. Create directories
mkdir -p data models plugins logs

# 8. Test installation
python -c "import torch; import transformers; print('✓ Installation successful!')"

# 9. Launch
python desktop_app.py
# or
python launch_chatbot.py server
```

---

## 🛠️ Post-Installation Setup

### First Launch

1. **Download Models:**
   - On first launch, models will download automatically
   - This may take 5-10 minutes
   - Models are cached for future use

2. **Configure Settings:**
   - Open Settings (Ctrl+,)
   - Choose model size
   - Set temperature
   - Configure voice/vision features

3. **Test Features:**
```bash
# Test CLI
python launch_chatbot.py cli

# Test server
python launch_chatbot.py server

# Test voice (optional)
python voice_interface.py test

# Test vision (optional)
python vision_interface.py
```

### Create Desktop Shortcuts

#### Windows:
```batch
# Desktop shortcut created automatically by installer
# Or manually create shortcut to:
# Target: C:\Path\To\AI_Chatbot\venv\Scripts\pythonw.exe desktop_app.py
```

#### macOS:
```bash
# Create alias
alias ai-chatbot='cd ~/AI_Chatbot && source venv/bin/activate && python desktop_app.py'

# Add to .zshrc or .bash_profile
echo "alias ai-chatbot='cd ~/AI_Chatbot && source venv/bin/activate && python desktop_app.py'" >> ~/.zshrc
```

#### Linux:
```bash
# Desktop entry created automatically by package
# Or manually create ~/.local/share/applications/ai-chatbot.desktop
```

---

## 📊 Installation Verification

### Quick Test:

```bash
cd AI_Chatbot

# Activate venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Run tests
python -c "
import sys
print('Python version:', sys.version)

import torch
print('✓ PyTorch:', torch.__version__)

import transformers
print('✓ Transformers:', transformers.__version__)

from neural_chatbot import NeuralChatbot
print('✓ Neural Chatbot loaded')

from rag_system import create_rag_system
print('✓ RAG System loaded')

print('\n✅ Installation verified successfully!')
"
```

### Full Test:

```bash
# Run test suite
pytest test_suite.py -v

# Or run quick smoke test
python launch_chatbot.py cli
# Type "hello" and verify response
```

---

## 🐛 Troubleshooting

### Common Issues:

#### "Python not found"
**Solution:**
```bash
# Install Python 3.8+
# Windows: https://www.python.org/downloads/
# macOS: brew install python@3.10
# Linux: sudo apt-get install python3.10
```

#### "Permission denied"
**Solution:**
```bash
# macOS/Linux - make scripts executable
chmod +x install.sh
chmod +x *.sh

# Windows - run as Administrator
```

#### "Module not found"
**Solution:**
```bash
# Ensure venv is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

#### "Out of memory"
**Solution:**
```python
# Use smaller model in config.yaml
model_name: microsoft/DialoGPT-small  # Instead of medium/large
```

#### "Port 5000 already in use"
**Solution:**
```bash
# Change port in config or use different port
python launch_chatbot.py server --port 5001
```

#### Windows: "torch not found" or install fails
**Solution:**
```batch
# Install CPU version explicitly
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### macOS: "Command not found: create-dmg"
**Solution:**
```bash
# Install Homebrew first
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then install create-dmg
brew install create-dmg
```

#### Android: "Insufficient storage"
**Solution:**
- Free up at least 5GB space
- Move large files to external storage
- Or use smaller model configuration

### Getting Help:

- 📚 Read [QUICKSTART.md](QUICKSTART.md) for quick guide
- 📖 Read [CHATBOT.md](CHATBOT.md) for full documentation
- 📱 Read [MOBILE.md](MOBILE.md) for Android-specific help
- 🐛 Report issues on GitHub
- 💬 Join community chat

---

## 🎯 Quick Reference

### Platform-Specific Quick Start

| Platform | Fastest Method | Command |
|----------|---------------|---------|
| **Windows** | Installer | Run `AI_Chatbot_Setup_3.0.exe` |
| **macOS** | DMG | Open `AI_Chatbot_3.0_macOS.dmg` |
| **Linux (Debian)** | .deb | `sudo dpkg -i ai-chatbot_3.0_all.deb` |
| **Linux (Fedora)** | .rpm | `sudo rpm -i ai-chatbot-3.0-1.rpm` |
| **Android** | Termux script | `./install_android.sh` |
| **Docker** | Compose | `docker-compose up -d` |
| **Any** | Wizard | `python installer_wizard.py` |
| **Any** | Manual | `./install.sh` |

### Launch Commands

```bash
# Desktop GUI
python desktop_app.py

# Web server
python launch_chatbot.py server

# CLI
python launch_chatbot.py cli

# API only
python launch_chatbot.py api

# Training mode
python launch_chatbot.py train
```

---

## 📦 Installation Sizes

| Component | Download Size | Installed Size |
|-----------|--------------|----------------|
| Core app | ~500 KB | ~2 MB |
| Python packages | ~1.5 GB | ~3 GB |
| Small model | ~500 MB | ~500 MB |
| Medium model | ~1.5 GB | ~1.5 GB |
| Large model | ~3 GB | ~3 GB |
| **Total (Small)** | **~2 GB** | **~3.5 GB** |
| **Total (Large)** | **~5 GB** | **~6.5 GB** |

---

## ⏱️ Installation Time

| Platform | Method | Time |
|----------|--------|------|
| Windows | Installer | 15-20 min |
| Windows | Wizard | 10-15 min |
| macOS | DMG | 15-20 min |
| Linux | Package | 10-15 min |
| Android | Termux | 20-30 min |
| Docker | Compose | 10-15 min |
| Manual | Script | 10-15 min |

*Times vary based on internet speed and system performance*

---

## 🎓 Next Steps

After installation:

1. **Quick Start:** Read [QUICKSTART.md](QUICKSTART.md)
2. **Full Guide:** Read [CHATBOT.md](CHATBOT.md)
3. **Learn Features:** Read [FEATURES.md](FEATURES.md)
4. **Compare:** Read [SOTA_COMPARISON.md](SOTA_COMPARISON.md)
5. **Mobile Setup:** Read [MOBILE.md](MOBILE.md)

**Happy Chatting!** 🤖💬

---

*Installation Guide v3.0 - Last updated: 2025-11-09*
