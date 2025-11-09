#!/bin/bash
################################################################################
# AI Chatbot - macOS App Bundle and DMG Creator
# Creates a native .app bundle and distributable .dmg
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

APP_NAME="AI Chatbot"
APP_VERSION="3.0"
BUNDLE_ID="com.airesearch.chatbot"
DMG_NAME="AI_Chatbot_${APP_VERSION}_macOS"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  AI Chatbot - macOS Package Builder${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script must be run on macOS${NC}"
    exit 1
fi

# Check prerequisites
echo -e "${YELLOW}[1/7]${NC} Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

if ! command -v create-dmg &> /dev/null; then
    echo -e "${YELLOW}Installing create-dmg...${NC}"
    brew install create-dmg || {
        echo -e "${RED}Error: Failed to install create-dmg${NC}"
        echo "Please install Homebrew first: https://brew.sh"
        exit 1
    }
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"

# Create build directory
echo -e "${YELLOW}[2/7]${NC} Creating build directory..."
BUILD_DIR="build/macos"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
echo -e "${GREEN}✓ Build directory ready${NC}"

# Create app bundle structure
echo -e "${YELLOW}[3/7]${NC} Creating app bundle structure..."

APP_BUNDLE="$BUILD_DIR/${APP_NAME}.app"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"
mkdir -p "$APP_BUNDLE/Contents/Frameworks"

echo -e "${GREEN}✓ App bundle structure created${NC}"

# Create Info.plist
echo -e "${YELLOW}[4/7]${NC} Creating Info.plist..."

cat > "$APP_BUNDLE/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>AI_Chatbot</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${APP_VERSION}</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSRequiresAquaSystemAppearance</key>
    <false/>
</dict>
</plist>
EOF

echo -e "${GREEN}✓ Info.plist created${NC}"

# Create launcher script
echo -e "${YELLOW}[5/7]${NC} Creating launcher script..."

cat > "$APP_BUNDLE/Contents/MacOS/AI_Chatbot" << 'EOF'
#!/bin/bash

# Get the directory of the app bundle
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES_DIR="$DIR/../Resources"

# Change to resources directory
cd "$RESOURCES_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Launch the application
python3 desktop_app.py

# Keep terminal open on error
if [ $? -ne 0 ]; then
    echo "Press any key to exit..."
    read -n 1
fi
EOF

chmod +x "$APP_BUNDLE/Contents/MacOS/AI_Chatbot"

echo -e "${GREEN}✓ Launcher script created${NC}"

# Copy application files
echo -e "${YELLOW}[6/7]${NC} Copying application files..."

# Copy Python files
cp *.py "$APP_BUNDLE/Contents/Resources/" 2>/dev/null || true

# Copy documentation
cp *.md "$APP_BUNDLE/Contents/Resources/" 2>/dev/null || true

# Copy requirements
cp requirements.txt "$APP_BUNDLE/Contents/Resources/" 2>/dev/null || true

# Create setup script for first launch
cat > "$APP_BUNDLE/Contents/Resources/setup.sh" << 'EOF'
#!/bin/bash
# First-time setup script

echo "Setting up AI Chatbot..."

# Create virtual environment
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
EOF

chmod +x "$APP_BUNDLE/Contents/Resources/setup.sh"

# Create icon (placeholder - would need actual .icns file)
# For production, create icon with: iconutil -c icns icon.iconset
if [ -f "icon.icns" ]; then
    cp icon.icns "$APP_BUNDLE/Contents/Resources/"
fi

echo -e "${GREEN}✓ Application files copied${NC}"

# Create DMG
echo -e "${YELLOW}[7/7]${NC} Creating DMG installer..."

# Create temporary directory for DMG contents
DMG_TEMP="$BUILD_DIR/dmg_temp"
mkdir -p "$DMG_TEMP"

# Copy app bundle
cp -R "$APP_BUNDLE" "$DMG_TEMP/"

# Create Applications symlink
ln -s /Applications "$DMG_TEMP/Applications"

# Create README for DMG
cat > "$DMG_TEMP/README.txt" << EOF
AI Chatbot ${APP_VERSION} for macOS

Installation:
1. Drag "AI Chatbot.app" to the Applications folder
2. Open AI Chatbot from Applications
3. On first launch, run setup: ./Contents/Resources/setup.sh
4. Enjoy!

For more information, see:
- QUICKSTART.md
- CHATBOT.md
- SOTA_COMPARISON.md

Visit: https://github.com/your-repo/AI_timestamp_context
EOF

# Create DMG
create-dmg \
    --volname "${APP_NAME} ${APP_VERSION}" \
    --window-pos 200 120 \
    --window-size 800 400 \
    --icon-size 100 \
    --icon "${APP_NAME}.app" 200 190 \
    --hide-extension "${APP_NAME}.app" \
    --app-drop-link 600 185 \
    "${DMG_NAME}.dmg" \
    "$DMG_TEMP" \
    || {
        echo -e "${YELLOW}Warning: create-dmg failed, creating simple DMG${NC}"
        hdiutil create -volname "${APP_NAME}" -srcfolder "$DMG_TEMP" -ov -format UDZO "${DMG_NAME}.dmg"
    }

# Move DMG to final location
mv "${DMG_NAME}.dmg" "$BUILD_DIR/" 2>/dev/null || true

# Cleanup
rm -rf "$DMG_TEMP"

echo -e "${GREEN}✓ DMG created${NC}"

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "App Bundle: $APP_BUNDLE"
echo "DMG Installer: $BUILD_DIR/${DMG_NAME}.dmg"
echo ""
echo -e "${BLUE}To test the app:${NC}"
echo "  open \"$APP_BUNDLE\""
echo ""
echo -e "${BLUE}To test the DMG:${NC}"
echo "  open \"$BUILD_DIR/${DMG_NAME}.dmg\""
echo ""
echo -e "${YELLOW}Note: First launch requires running setup.sh to install dependencies${NC}"
echo ""
