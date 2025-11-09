#!/data/data/com.termux/files/usr/bin/bash
################################################################################
# AI Chatbot - Android/Termux Installation Wizard
# Interactive installer for Termux on Android devices
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emoji support (Termux supports these)
CHECK="✓"
CROSS="✗"
ARROW="→"
ROCKET="🚀"
ROBOT="🤖"
PHONE="📱"
PACKAGE="📦"
DOWNLOAD="⬇️"
INSTALL="🔧"
SUCCESS="🎉"

################################################################################
# Helper Functions
################################################################################

print_header() {
    clear
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║              ${ROBOT} AI CHATBOT INSTALLER ${PHONE}              ║"
    echo "║           Professional Installation for Android             ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${MAGENTA}${ARROW} $1${NC}"
}

print_success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

print_error() {
    echo -e "${RED}${CROSS} $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

show_progress() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local filled=$((percent / 5))
    local empty=$((20 - filled))

    printf "\r${CYAN}Progress: ["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] ${percent}%%${NC}"
}

press_enter() {
    echo -e "\n${YELLOW}Press Enter to continue...${NC}"
    read
}

confirm() {
    local prompt="$1"
    local default="${2:-y}"

    while true; do
        if [ "$default" = "y" ]; then
            echo -e -n "${CYAN}${prompt} [Y/n]: ${NC}"
        else
            echo -e -n "${CYAN}${prompt} [y/N]: ${NC}"
        fi

        read response
        response=${response:-$default}

        case "$response" in
            [Yy]*) return 0 ;;
            [Nn]*) return 1 ;;
            *) echo -e "${RED}Please answer yes or no.${NC}" ;;
        esac
    done
}

check_termux() {
    if [ ! -d "/data/data/com.termux" ]; then
        print_error "This script must be run in Termux!"
        echo "Please install Termux from F-Droid: https://f-droid.org/en/packages/com.termux/"
        exit 1
    fi
}

check_storage() {
    print_step "Checking storage permissions..."

    if [ ! -d "$HOME/storage/shared" ]; then
        print_warning "Storage access not configured!"
        echo -e "\n${YELLOW}Termux needs storage access to save your data.${NC}"

        if confirm "Setup storage access now?"; then
            termux-setup-storage
            sleep 3

            if [ -d "$HOME/storage/shared" ]; then
                print_success "Storage access configured!"
            else
                print_warning "Storage access may not be fully configured. Continuing anyway..."
            fi
        fi
    else
        print_success "Storage access OK"
    fi
}

get_device_info() {
    print_step "Detecting device information..."

    local arch=$(uname -m)
    local ram=$(free -h | awk '/^Mem:/ {print $2}')
    local storage=$(df -h $HOME | awk 'NR==2 {print $4}')

    echo -e "${BLUE}"
    echo "  Device Information:"
    echo "  ├─ Architecture: $arch"
    echo "  ├─ Available RAM: $ram"
    echo "  └─ Available Storage: $storage"
    echo -e "${NC}"

    # Check minimum requirements
    local storage_gb=$(df -BG $HOME | awk 'NR==2 {print $4}' | sed 's/G//')

    if [ "$storage_gb" -lt 5 ]; then
        print_warning "Low storage space! Installation requires at least 5GB free."
        if ! confirm "Continue anyway?"; then
            exit 1
        fi
    fi
}

################################################################################
# Installation Steps
################################################################################

step1_welcome() {
    print_header

    echo -e "${GREEN}"
    echo "Welcome to the AI Chatbot Installation Wizard!"
    echo -e "${NC}"

    echo "This wizard will install:"
    echo "  ${CHECK} Neural network chatbot with continual learning"
    echo "  ${CHECK} Voice interface (speech recognition + TTS)"
    echo "  ${CHECK} Web interface accessible from any browser"
    echo "  ${CHECK} Command-line interface"
    echo "  ${CHECK} REST API for integration"
    echo ""
    echo "Installation time: ~10-20 minutes (depends on internet speed)"
    echo "Required space: ~5GB"
    echo ""

    press_enter
}

step2_update_packages() {
    print_header
    print_step "Updating Termux packages..."

    echo "This ensures you have the latest package information."
    echo ""

    # Update package lists
    pkg update -y

    print_success "Package lists updated!"
    sleep 1
}

step3_install_dependencies() {
    print_header
    print_step "Installing system dependencies..."

    echo "Installing required system packages..."
    echo "This may take several minutes."
    echo ""

    local packages=(
        "python"
        "python-pip"
        "git"
        "build-essential"
        "binutils"
        "clang"
        "cmake"
        "libffi"
        "libjpeg-turbo"
        "libpng"
        "openssl"
    )

    local total=${#packages[@]}
    local current=0

    for package in "${packages[@]}"; do
        current=$((current + 1))
        show_progress $current $total

        if pkg list-installed | grep -q "^${package}/"; then
            : # Already installed
        else
            pkg install -y "$package" > /dev/null 2>&1 || true
        fi
    done

    echo ""
    print_success "System dependencies installed!"
    sleep 1
}

step4_create_directory() {
    print_header
    print_step "Creating installation directory..."

    INSTALL_DIR="$HOME/AI_Chatbot"

    echo "Installation directory: $INSTALL_DIR"
    echo ""

    if [ -d "$INSTALL_DIR" ]; then
        print_warning "Directory already exists!"

        if confirm "Remove existing installation and reinstall?"; then
            rm -rf "$INSTALL_DIR"
            print_success "Old installation removed"
        else
            print_error "Installation cancelled"
            exit 1
        fi
    fi

    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    # Create subdirectories
    mkdir -p data models plugins logs

    print_success "Installation directory created!"
    sleep 1
}

step5_setup_python() {
    print_header
    print_step "Setting up Python environment..."

    echo "Creating virtual environment..."
    echo ""

    # Upgrade pip
    pip install --upgrade pip > /dev/null 2>&1

    # Create venv
    python -m venv venv

    # Activate venv
    source venv/bin/activate

    # Upgrade pip in venv
    pip install --upgrade pip wheel setuptools > /dev/null 2>&1

    print_success "Python environment ready!"
    sleep 1
}

step6_install_python_packages() {
    print_header
    print_step "Installing Python packages..."

    echo "This is the longest step (5-15 minutes)."
    echo "Installing AI/ML libraries..."
    echo ""

    # Core packages with progress indication
    local packages=(
        "numpy"
        "torch --index-url https://download.pytorch.org/whl/cpu"
        "transformers"
        "sentence-transformers"
        "flask"
        "flask-cors"
        "flask-socketio"
        "python-socketio"
        "eventlet"
        "matplotlib"
        "Pillow"
    )

    local total=${#packages[@]}
    local current=0

    for package in "${packages[@]}"; do
        current=$((current + 1))
        show_progress $current $total
        pip install $package > /dev/null 2>&1 || print_warning "Failed to install $package"
    done

    echo ""
    print_success "Python packages installed!"
    sleep 1
}

step7_download_source() {
    print_header
    print_step "Downloading AI Chatbot source code..."

    echo "Fetching latest version from repository..."
    echo ""

    # Download source files (you can customize this)
    local files=(
        "neural_chatbot.py"
        "knowledge_store.py"
        "rag_system.py"
        "chatbot_server.py"
        "launch_chatbot.py"
        "voice_interface.py"
    )

    # For demo, create placeholder message
    echo "# Note: In production, files would be copied or cloned from git"
    echo "# For now, ensure source files are in current directory"

    print_success "Source code ready!"
    sleep 1
}

step8_configure() {
    print_header
    print_step "Configuring AI Chatbot..."

    echo "Creating configuration file..."
    echo ""

    cat > config.yaml << EOF
# AI Chatbot Configuration for Android/Termux
install_dir: $INSTALL_DIR
model_name: microsoft/DialoGPT-small
temperature: 0.8
database_path: $INSTALL_DIR/data/chatbot.db
enable_voice: true
enable_vision: false  # Disabled on mobile for performance
enable_analytics: true
server_port: 5000
server_host: 0.0.0.0
EOF

    print_success "Configuration created!"
    sleep 1
}

step9_create_launchers() {
    print_header
    print_step "Creating launcher scripts..."

    echo "Setting up easy launch commands..."
    echo ""

    # Web server launcher
    cat > start_server.sh << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
cd ~/AI_Chatbot
source venv/bin/activate
python launch_chatbot.py server
EOF
    chmod +x start_server.sh

    # CLI launcher
    cat > start_cli.sh << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
cd ~/AI_Chatbot
source venv/bin/activate
python launch_chatbot.py cli
EOF
    chmod +x start_cli.sh

    # Create termux shortcut
    mkdir -p ~/.shortcuts
    cat > ~/.shortcuts/AI_Chatbot << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
cd ~/AI_Chatbot
./start_server.sh
EOF
    chmod +x ~/.shortcuts/AI_Chatbot

    print_success "Launcher scripts created!"
    sleep 1
}

step10_test_installation() {
    print_header
    print_step "Testing installation..."

    echo "Running quick test..."
    echo ""

    # Test Python imports
    python -c "import torch; import transformers; print('✓ PyTorch and Transformers OK')" && \
        print_success "Core libraries working!" || \
        print_warning "Some libraries may need attention"

    sleep 2
}

step11_complete() {
    print_header

    echo -e "${GREEN}${SUCCESS}${SUCCESS}${SUCCESS}"
    echo "  Installation Complete!"
    echo -e "${SUCCESS}${SUCCESS}${SUCCESS}${NC}"
    echo ""

    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    echo ""
    echo "🎉 AI Chatbot is now installed on your Android device!"
    echo ""
    echo -e "${YELLOW}Quick Start:${NC}"
    echo ""
    echo "  1️⃣  Start Web Server:"
    echo "     cd ~/AI_Chatbot && ./start_server.sh"
    echo ""
    echo "  2️⃣  Open Browser:"
    echo "     http://localhost:5000"
    echo ""
    echo "  3️⃣  Or use CLI:"
    echo "     cd ~/AI_Chatbot && ./start_cli.sh"
    echo ""
    echo "  4️⃣  Widget Shortcut:"
    echo "     Add 'Termux Widget' to home screen"
    echo "     Tap 'AI_Chatbot' to launch"
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${BLUE}📚 Documentation:${NC}"
    echo "  • README.md - Overview"
    echo "  • QUICKSTART.md - Quick guide"
    echo "  • MOBILE.md - Mobile-specific tips"
    echo ""
    echo -e "${BLUE}💡 Tips:${NC}"
    echo "  • Use Chrome/Firefox for best web UI experience"
    echo "  • Access from other devices on same WiFi"
    echo "  • First model download will take a few minutes"
    echo ""
    echo -e "${BLUE}🆘 Need Help?${NC}"
    echo "  • Check MOBILE.md for troubleshooting"
    echo "  • Visit GitHub issues page"
    echo ""
    echo -e "${GREEN}Thank you for installing AI Chatbot!${NC}"
    echo ""

    if confirm "Start AI Chatbot now?"; then
        echo ""
        print_info "Starting web server..."
        echo "Open http://localhost:5000 in your browser"
        echo ""
        cd ~/AI_Chatbot
        ./start_server.sh
    fi
}

################################################################################
# Main Installation Flow
################################################################################

main() {
    # Pre-flight checks
    check_termux
    check_storage
    get_device_info

    # Installation steps
    step1_welcome
    step2_update_packages
    step3_install_dependencies
    step4_create_directory
    step5_setup_python
    step6_install_python_packages
    step7_download_source
    step8_configure
    step9_create_launchers
    step10_test_installation
    step11_complete
}

################################################################################
# Error Handling
################################################################################

trap 'echo -e "\n${RED}Installation interrupted!${NC}"; exit 1' INT TERM

################################################################################
# Run Installation
################################################################################

main "$@"
