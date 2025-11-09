#!/usr/bin/env python3
"""
AI Chatbot Installation Wizard Bootstrap
This script automatically installs dependencies for the installer itself,
then launches the full installation wizard.
"""

import sys
import subprocess
import os
import platform

def print_banner():
    """Print installation banner"""
    print("="*70)
    print("  🤖 AI CHATBOT - PROFESSIONAL INSTALLATION WIZARD")
    print("="*70)
    print()

def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    if version < (3, 8):
        print(f"❌ Error: Python 3.8+ required, found {version.major}.{version.minor}")
        print()
        print("Please install Python 3.8 or higher from:")
        print("  https://www.python.org/downloads/")
        input("\nPress Enter to exit...")
        sys.exit(1)

    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")

def install_dependencies():
    """Install installer dependencies"""
    print("\n📦 Checking installer dependencies...")

    dependencies = ['PyQt6']

    for dep in dependencies:
        try:
            __import__(dep.lower().replace('-', '_'))
            print(f"  ✓ {dep} already installed")
        except ImportError:
            print(f"  ⬇️  Installing {dep}...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", dep],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"  ✓ {dep} installed successfully")
            except subprocess.CalledProcessError:
                print(f"  ❌ Failed to install {dep}")
                print()
                print("Please install manually:")
                print(f"  pip install {dep}")
                input("\nPress Enter to exit...")
                sys.exit(1)

def launch_installer():
    """Launch the main installer"""
    print("\n🚀 Launching installation wizard...")
    print()

    # Ask about connection speed
    print("Do you have a slow internet connection?")
    print("  [1] Fast connection (use default installer with 10min timeouts)")
    print("  [2] Slow connection (use NO TIMEOUT installer - recommended!)")
    print()

    choice = input("Enter 1 or 2 (default: 2 for slow): ").strip()

    installer_choice = 'installer_no_timeout.py'  # Default to no timeout

    if choice == '1':
        installer_choice = 'enhanced_installer.py'
        print("\n✓ Using fast connection installer (10min timeouts)")
    else:
        print("\n✓ Using NO TIMEOUT installer (recommended for most users)")
        print("  This installer will NEVER timeout - perfect for slow connections!")

    print()

    try:
        if os.path.exists(installer_choice):
            subprocess.run([sys.executable, installer_choice])
        elif os.path.exists('installer_wizard.py'):
            subprocess.run([sys.executable, 'installer_wizard.py'])
        else:
            subprocess.run([sys.executable, 'enhanced_installer.py'])
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error launching installer: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)

def main():
    """Main bootstrap function"""
    try:
        print_banner()
        check_python_version()
        install_dependencies()
        launch_installer()
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == '__main__':
    main()
