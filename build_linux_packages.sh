#!/bin/bash
################################################################################
# AI Chatbot - Linux Package Builder
# Creates .deb (Debian/Ubuntu) and .rpm (Fedora/RHEL) packages
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

APP_NAME="ai-chatbot"
APP_VERSION="3.0"
APP_DESCRIPTION="State-of-the-art AI chatbot with continual learning"
MAINTAINER="AI Research <contact@example.com>"
HOMEPAGE="https://github.com/your-repo/AI_timestamp_context"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  AI Chatbot - Linux Package Builder${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Detect distribution
if [ -f /etc/debian_version ]; then
    DISTRO="debian"
elif [ -f /etc/redhat-release ]; then
    DISTRO="redhat"
else
    echo -e "${YELLOW}Warning: Could not detect distribution${NC}"
    DISTRO="unknown"
fi

echo "Detected distribution: $DISTRO"
echo ""

################################################################################
# Build .deb package (Debian/Ubuntu)
################################################################################

build_deb() {
    echo -e "${YELLOW}Building .deb package...${NC}"
    echo ""

    # Check for dpkg-deb
    if ! command -v dpkg-deb &> /dev/null; then
        echo -e "${RED}Error: dpkg-deb not found${NC}"
        echo "Install with: sudo apt-get install dpkg-dev"
        return 1
    fi

    local DEB_DIR="build/deb/${APP_NAME}_${APP_VERSION}"

    # Create directory structure
    echo -e "${BLUE}[1/5]${NC} Creating directory structure..."
    rm -rf "$DEB_DIR"
    mkdir -p "$DEB_DIR/DEBIAN"
    mkdir -p "$DEB_DIR/usr/share/${APP_NAME}"
    mkdir -p "$DEB_DIR/usr/share/applications"
    mkdir -p "$DEB_DIR/usr/share/pixmaps"
    mkdir -p "$DEB_DIR/usr/bin"

    # Create control file
    echo -e "${BLUE}[2/5]${NC} Creating control file..."
    cat > "$DEB_DIR/DEBIAN/control" << EOF
Package: ${APP_NAME}
Version: ${APP_VERSION}
Section: misc
Priority: optional
Architecture: all
Depends: python3 (>= 3.8), python3-pip, python3-venv
Maintainer: ${MAINTAINER}
Description: ${APP_DESCRIPTION}
 AI Chatbot is a state-of-the-art neural network chatbot with:
 - Real-time continual learning from conversations
 - Voice interface (speech recognition + TTS)
 - Vision/image understanding capabilities
 - Interactive knowledge graphs
 - Comprehensive analytics dashboard
 - Extensible plugin system
 - Complete privacy (100% local processing)
Homepage: ${HOMEPAGE}
EOF

    # Create postinst script
    cat > "$DEB_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/bash
set -e

APP_DIR="/usr/share/ai-chatbot"

echo "Setting up AI Chatbot..."

cd "$APP_DIR"

# Create virtual environment
python3 -m venv venv

# Install dependencies
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

# Create data directories
mkdir -p "$APP_DIR/data"
mkdir -p "$APP_DIR/models"
mkdir -p "$APP_DIR/plugins"
mkdir -p "$APP_DIR/logs"

# Set permissions
chmod -R 755 "$APP_DIR"

echo "AI Chatbot installation complete!"
echo "Launch with: ai-chatbot"

exit 0
EOF

    chmod +x "$DEB_DIR/DEBIAN/postinst"

    # Create prerm script
    cat > "$DEB_DIR/DEBIAN/prerm" << 'EOF'
#!/bin/bash
set -e

# Stop any running instances
pkill -f "ai-chatbot" || true

exit 0
EOF

    chmod +x "$DEB_DIR/DEBIAN/prerm"

    # Copy application files
    echo -e "${BLUE}[3/5]${NC} Copying application files..."
    cp *.py "$DEB_DIR/usr/share/${APP_NAME}/" 2>/dev/null || true
    cp *.md "$DEB_DIR/usr/share/${APP_NAME}/" 2>/dev/null || true
    cp requirements.txt "$DEB_DIR/usr/share/${APP_NAME}/"

    # Create launcher script
    echo -e "${BLUE}[4/5]${NC} Creating launcher..."
    cat > "$DEB_DIR/usr/bin/ai-chatbot" << 'EOF'
#!/bin/bash
cd /usr/share/ai-chatbot
source venv/bin/activate
python desktop_app.py "$@"
EOF

    chmod +x "$DEB_DIR/usr/bin/ai-chatbot"

    # Create desktop entry
    cat > "$DEB_DIR/usr/share/applications/${APP_NAME}.desktop" << EOF
[Desktop Entry]
Name=AI Chatbot
Comment=${APP_DESCRIPTION}
Exec=ai-chatbot
Icon=ai-chatbot
Terminal=false
Type=Application
Categories=Utility;Development;Science;
Keywords=ai;chatbot;machine-learning;nlp;
EOF

    # Create icon (placeholder)
    if [ -f "icon.png" ]; then
        cp icon.png "$DEB_DIR/usr/share/pixmaps/${APP_NAME}.png"
    fi

    # Build package
    echo -e "${BLUE}[5/5]${NC} Building .deb package..."
    dpkg-deb --build "$DEB_DIR" "build/${APP_NAME}_${APP_VERSION}_all.deb"

    echo -e "${GREEN}✓ .deb package created: build/${APP_NAME}_${APP_VERSION}_all.deb${NC}"
    echo ""
}

################################################################################
# Build .rpm package (Fedora/RHEL/CentOS)
################################################################################

build_rpm() {
    echo -e "${YELLOW}Building .rpm package...${NC}"
    echo ""

    # Check for rpmbuild
    if ! command -v rpmbuild &> /dev/null; then
        echo -e "${RED}Error: rpmbuild not found${NC}"
        echo "Install with: sudo dnf install rpm-build (Fedora)"
        echo "          or: sudo yum install rpm-build (RHEL/CentOS)"
        return 1
    fi

    local RPM_BUILD="build/rpm"
    local SPEC_FILE="$RPM_BUILD/SPECS/${APP_NAME}.spec"

    # Create directory structure
    echo -e "${BLUE}[1/4]${NC} Creating RPM build structure..."
    rm -rf "$RPM_BUILD"
    mkdir -p "$RPM_BUILD"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

    # Create source tarball
    echo -e "${BLUE}[2/4]${NC} Creating source tarball..."
    local TAR_DIR="${APP_NAME}-${APP_VERSION}"
    mkdir -p "build/tar/$TAR_DIR"

    cp *.py "build/tar/$TAR_DIR/" 2>/dev/null || true
    cp *.md "build/tar/$TAR_DIR/" 2>/dev/null || true
    cp requirements.txt "build/tar/$TAR_DIR/"

    cd build/tar
    tar czf "$TAR_DIR.tar.gz" "$TAR_DIR"
    mv "$TAR_DIR.tar.gz" "../rpm/SOURCES/"
    cd ../..

    # Create spec file
    echo -e "${BLUE}[3/4]${NC} Creating spec file..."
    cat > "$SPEC_FILE" << EOF
Name:           ${APP_NAME}
Version:        ${APP_VERSION}
Release:        1%{?dist}
Summary:        ${APP_DESCRIPTION}

License:        MIT
URL:            ${HOMEPAGE}
Source0:        %{name}-%{version}.tar.gz

BuildArch:      noarch
Requires:       python3 >= 3.8
Requires:       python3-pip
Requires:       python3-virtualenv

%description
AI Chatbot is a state-of-the-art neural network chatbot with:
- Real-time continual learning from conversations
- Voice interface (speech recognition + TTS)
- Vision/image understanding capabilities
- Interactive knowledge graphs
- Comprehensive analytics dashboard
- Extensible plugin system
- Complete privacy (100%% local processing)

%prep
%setup -q

%build
# Nothing to build (Python application)

%install
rm -rf \$RPM_BUILD_ROOT

# Create directories
mkdir -p \$RPM_BUILD_ROOT/usr/share/%{name}
mkdir -p \$RPM_BUILD_ROOT/usr/bin
mkdir -p \$RPM_BUILD_ROOT/usr/share/applications

# Copy files
cp -r * \$RPM_BUILD_ROOT/usr/share/%{name}/

# Create launcher
cat > \$RPM_BUILD_ROOT/usr/bin/ai-chatbot << 'LAUNCHER'
#!/bin/bash
cd /usr/share/ai-chatbot
source venv/bin/activate
python desktop_app.py "\$@"
LAUNCHER

chmod +x \$RPM_BUILD_ROOT/usr/bin/ai-chatbot

# Create desktop entry
cat > \$RPM_BUILD_ROOT/usr/share/applications/%{name}.desktop << 'DESKTOP'
[Desktop Entry]
Name=AI Chatbot
Comment=${APP_DESCRIPTION}
Exec=ai-chatbot
Terminal=false
Type=Application
Categories=Utility;Development;Science;
DESKTOP

%post
cd /usr/share/%{name}

# Create virtual environment
python3 -m venv venv

# Install dependencies
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

# Create data directories
mkdir -p data models plugins logs

echo "AI Chatbot installation complete!"
echo "Launch with: ai-chatbot"

%preun
# Stop any running instances
pkill -f "ai-chatbot" || true

%files
/usr/share/%{name}/*
/usr/bin/ai-chatbot
/usr/share/applications/%{name}.desktop

%changelog
* $(date "+%a %b %d %Y") ${MAINTAINER} - ${APP_VERSION}-1
- Initial release
EOF

    # Build RPM
    echo -e "${BLUE}[4/4]${NC} Building .rpm package..."
    rpmbuild --define "_topdir $(pwd)/$RPM_BUILD" -ba "$SPEC_FILE"

    # Find and copy the built RPM
    find "$RPM_BUILD/RPMS" -name "*.rpm" -exec cp {} "build/" \;

    echo -e "${GREEN}✓ .rpm package created in build/ directory${NC}"
    echo ""
}

################################################################################
# Main
################################################################################

main() {
    # Create build directory
    mkdir -p build

    # Build packages based on user choice or detected distro
    if [ "$1" == "deb" ] || [ "$DISTRO" == "debian" ]; then
        build_deb
    fi

    if [ "$1" == "rpm" ] || [ "$DISTRO" == "redhat" ]; then
        build_rpm
    fi

    if [ "$1" == "all" ] || [ -z "$1" ]; then
        build_deb || echo -e "${YELLOW}Skipping .deb${NC}"
        build_rpm || echo -e "${YELLOW}Skipping .rpm${NC}"
    fi

    # Summary
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Build Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Built packages:"
    ls -lh build/*.deb 2>/dev/null || true
    ls -lh build/*.rpm 2>/dev/null || true
    echo ""
    echo -e "${BLUE}Installation:${NC}"
    echo "  Debian/Ubuntu: sudo dpkg -i build/${APP_NAME}_${APP_VERSION}_all.deb"
    echo "  Fedora/RHEL:   sudo rpm -i build/${APP_NAME}-${APP_VERSION}-1.*.rpm"
    echo ""
    echo -e "${BLUE}After installation:${NC}"
    echo "  Launch with: ai-chatbot"
    echo ""
}

# Parse arguments
case "$1" in
    deb|rpm|all)
        main "$1"
        ;;
    help|--help|-h)
        echo "Usage: $0 [deb|rpm|all]"
        echo ""
        echo "  deb  - Build .deb package only"
        echo "  rpm  - Build .rpm package only"
        echo "  all  - Build both packages (default)"
        echo ""
        ;;
    *)
        main all
        ;;
esac
