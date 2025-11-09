"""
Enhanced AI Chatbot Installer - Production Grade
Complete installation solution with automatic dependency management,
error recovery, Python installation, PATH configuration, and more!
"""

import sys
import os
import subprocess
import platform
import shutil
import urllib.request
import urllib.error
import json
import tempfile
import hashlib
import zipfile
import tarfile
import time
import ctypes
import winreg
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from enum import Enum

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("Installing PyQt6 for installer UI...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt6"])
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *


class InstallationState(Enum):
    """Installation state tracking"""
    NOT_STARTED = "not_started"
    CHECKING_PYTHON = "checking_python"
    DOWNLOADING_PYTHON = "downloading_python"
    INSTALLING_PYTHON = "installing_python"
    CHECKING_DEPENDENCIES = "checking_dependencies"
    CREATING_VENV = "creating_venv"
    INSTALLING_PACKAGES = "installing_packages"
    COPYING_FILES = "copying_files"
    CREATING_SHORTCUTS = "creating_shortcuts"
    CONFIGURING_PATH = "configuring_path"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class NetworkManager:
    """Handle network operations with retries and fallbacks"""

    MIRROR_URLS = {
        'python': [
            'https://www.python.org/ftp/python/',
            'https://python.mirror.aarnet.edu.au/ftp/python/',
        ],
        'pypi': [
            'https://pypi.org/simple/',
            'https://pypi.python.org/simple/',
            'https://mirrors.aliyun.com/pypi/simple/',
        ]
    }

    @staticmethod
    def download_with_retry(url: str, dest: str, max_retries: int = 5,
                           progress_callback=None) -> bool:
        """Download file with retry logic and progress tracking"""
        for attempt in range(max_retries):
            try:
                # Create request with headers
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 AI-Chatbot-Installer/1.0'}
                )

                # Open connection
                with urllib.request.urlopen(req, timeout=30) as response:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    chunk_size = 8192

                    with open(dest, 'wb') as f:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break

                            f.write(chunk)
                            downloaded += len(chunk)

                            if progress_callback and total_size > 0:
                                progress = int((downloaded / total_size) * 100)
                                progress_callback(progress, downloaded, total_size)

                return True

            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Failed to download after {max_retries} attempts: {e}")
                    return False

            except Exception as e:
                print(f"Unexpected error during download: {e}")
                return False

        return False

    @staticmethod
    def verify_checksum(file_path: str, expected_hash: str, algorithm='sha256') -> bool:
        """Verify file integrity with checksum"""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)

            return hash_obj.hexdigest().lower() == expected_hash.lower()
        except Exception as e:
            print(f"Checksum verification failed: {e}")
            return False


class PythonManager:
    """Manage Python installation and configuration"""

    PYTHON_VERSIONS = {
        'windows_x64': {
            '3.11': 'https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe',
            '3.10': 'https://www.python.org/ftp/python/3.10.13/python-3.10.13-amd64.exe',
        },
        'windows_x86': {
            '3.11': 'https://www.python.org/ftp/python/3.11.7/python-3.11.7.exe',
        }
    }

    @staticmethod
    def detect_python() -> Tuple[bool, Optional[str], Optional[str]]:
        """Detect if Python is installed and return version and path"""
        try:
            # Try python3 first
            result = subprocess.run(
                ['python3', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                version = result.stdout.strip().split()[1]
                python_path = shutil.which('python3')
                return True, version, python_path

        except FileNotFoundError:
            pass

        try:
            # Try python
            result = subprocess.run(
                ['python', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                version = result.stdout.strip().split()[1]
                python_path = shutil.which('python')

                # Check version is >= 3.8
                major, minor = map(int, version.split('.')[:2])
                if major >= 3 and minor >= 8:
                    return True, version, python_path

        except (FileNotFoundError, ValueError):
            pass

        return False, None, None

    @staticmethod
    def download_python(dest_dir: str, progress_callback=None) -> Optional[str]:
        """Download Python installer"""
        system = platform.system()
        machine = platform.machine()

        if system == 'Windows':
            arch_key = 'windows_x64' if machine.endswith('64') else 'windows_x86'
            url = PythonManager.PYTHON_VERSIONS[arch_key]['3.11']

            dest_file = os.path.join(dest_dir, 'python_installer.exe')

            print(f"Downloading Python from {url}...")
            success = NetworkManager.download_with_retry(
                url, dest_file, progress_callback=progress_callback
            )

            return dest_file if success else None

        return None

    @staticmethod
    def install_python_windows(installer_path: str, install_dir: str) -> bool:
        """Install Python silently on Windows"""
        try:
            # Run installer with silent flags
            cmd = [
                installer_path,
                '/quiet',
                'InstallAllUsers=0',
                f'TargetDir={install_dir}',
                'PrependPath=1',
                'Include_test=0',
                'Include_pip=1',
                'Include_launcher=1'
            ]

            result = subprocess.run(cmd, check=True, capture_output=True)
            return result.returncode == 0

        except subprocess.CalledProcessError as e:
            print(f"Python installation failed: {e}")
            return False

    @staticmethod
    def add_to_path_windows(directory: str) -> bool:
        """Add directory to Windows PATH"""
        try:
            # Get current user PATH
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r'Environment',
                0,
                winreg.KEY_ALL_ACCESS
            )

            try:
                current_path, _ = winreg.QueryValueEx(key, 'Path')
            except FileNotFoundError:
                current_path = ''

            # Add new directory if not already present
            paths = current_path.split(os.pathsep)
            if directory not in paths:
                paths.append(directory)
                new_path = os.pathsep.join(paths)

                winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)

            winreg.CloseKey(key)

            # Broadcast environment change
            HWND_BROADCAST = 0xFFFF
            WM_SETTINGCHANGE = 0x001A
            ctypes.windll.user32.SendMessageW(
                HWND_BROADCAST, WM_SETTINGCHANGE, 0, 'Environment'
            )

            return True

        except Exception as e:
            print(f"Failed to add to PATH: {e}")
            return False


class IconGenerator:
    """Generate application icons"""

    @staticmethod
    def create_icon(output_path: str, size: int = 256):
        """Create a simple AI Chatbot icon"""
        try:
            from PIL import Image, ImageDraw, ImageFont

            # Create image
            img = Image.new('RGB', (size, size), color='#2196F3')
            draw = ImageDraw.Draw(img)

            # Draw robot face
            # Eyes
            eye_size = size // 8
            eye_y = size // 3
            draw.ellipse(
                [size//3 - eye_size, eye_y - eye_size, size//3 + eye_size, eye_y + eye_size],
                fill='white'
            )
            draw.ellipse(
                [2*size//3 - eye_size, eye_y - eye_size, 2*size//3 + eye_size, eye_y + eye_size],
                fill='white'
            )

            # Smile
            mouth_y = 2*size//3
            draw.arc(
                [size//4, mouth_y - size//8, 3*size//4, mouth_y + size//8],
                start=0, end=180, fill='white', width=size//30
            )

            # Save as PNG
            img.save(output_path, 'PNG')

            # Create .ico for Windows
            if platform.system() == 'Windows':
                ico_path = output_path.replace('.png', '.ico')
                img.save(ico_path, format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])

            return True

        except Exception as e:
            print(f"Icon creation failed: {e}")
            # Create a simple fallback icon
            return IconGenerator.create_simple_icon(output_path, size)

    @staticmethod
    def create_simple_icon(output_path: str, size: int = 256):
        """Create ultra-simple fallback icon without PIL"""
        try:
            # Create a simple SVG icon
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{size}" height="{size}" fill="#2196F3"/>
  <circle cx="{size//3}" cy="{size//3}" r="{size//10}" fill="white"/>
  <circle cx="{2*size//3}" cy="{size//3}" r="{size//10}" fill="white"/>
  <path d="M {size//4} {2*size//3} Q {size//2} {3*size//4} {3*size//4} {2*size//3}"
        stroke="white" stroke-width="{size//30}" fill="none"/>
  <text x="{size//2}" y="{size-20}" text-anchor="middle" fill="white" font-size="24" font-family="Arial">AI</text>
</svg>'''

            svg_path = output_path.replace('.png', '.svg')
            with open(svg_path, 'w') as f:
                f.write(svg_content)

            return True

        except Exception as e:
            print(f"Simple icon creation failed: {e}")
            return False


class ShortcutManager:
    """Create desktop shortcuts and Start Menu entries"""

    @staticmethod
    def create_windows_shortcut(target: str, shortcut_path: str,
                               icon_path: Optional[str] = None,
                               description: str = ""):
        """Create Windows shortcut (.lnk)"""
        try:
            import win32com.client

            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = target
            shortcut.WorkingDirectory = os.path.dirname(target)
            shortcut.Description = description

            if icon_path:
                shortcut.IconLocation = icon_path

            shortcut.save()
            return True

        except ImportError:
            # Fallback: Install pywin32
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])
                return ShortcutManager.create_windows_shortcut(
                    target, shortcut_path, icon_path, description
                )
            except:
                pass

        except Exception as e:
            print(f"Shortcut creation failed: {e}")

        return False

    @staticmethod
    def create_desktop_shortcut(app_name: str, target: str, icon_path: Optional[str] = None):
        """Create desktop shortcut"""
        system = platform.system()

        if system == 'Windows':
            desktop = Path.home() / "Desktop"
            shortcut_path = desktop / f"{app_name}.lnk"
            return ShortcutManager.create_windows_shortcut(
                target, str(shortcut_path), icon_path, f"Launch {app_name}"
            )

        elif system == 'Darwin':  # macOS
            desktop = Path.home() / "Desktop"
            app_path = desktop / f"{app_name}.app"
            # Create simple app bundle
            os.makedirs(app_path / "Contents" / "MacOS", exist_ok=True)

            launcher_script = app_path / "Contents" / "MacOS" / app_name
            with open(launcher_script, 'w') as f:
                f.write(f"#!/bin/bash\n{target}")

            os.chmod(launcher_script, 0o755)
            return True

        else:  # Linux
            desktop = Path.home() / "Desktop"
            desktop_file = desktop / f"{app_name}.desktop"

            with open(desktop_file, 'w') as f:
                f.write(f"""[Desktop Entry]
Name={app_name}
Exec={target}
Icon={icon_path or ''}
Terminal=false
Type=Application
Categories=Utility;Development;
""")

            os.chmod(desktop_file, 0o755)
            return True

    @staticmethod
    def create_start_menu_entry(app_name: str, target: str, icon_path: Optional[str] = None):
        """Create Start Menu / Applications entry"""
        system = platform.system()

        if system == 'Windows':
            start_menu = Path(os.environ.get('APPDATA')) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
            shortcut_path = start_menu / f"{app_name}.lnk"
            return ShortcutManager.create_windows_shortcut(
                target, str(shortcut_path), icon_path, f"Launch {app_name}"
            )

        elif system == 'Linux':
            apps_dir = Path.home() / ".local" / "share" / "applications"
            apps_dir.mkdir(parents=True, exist_ok=True)

            desktop_file = apps_dir / f"{app_name.lower().replace(' ', '-')}.desktop"

            with open(desktop_file, 'w') as f:
                f.write(f"""[Desktop Entry]
Name={app_name}
Exec={target}
Icon={icon_path or ''}
Terminal=false
Type=Application
Categories=Utility;Development;Science;
Keywords=ai;chatbot;machine-learning;
""")

            return True

        return False


class EnhancedInstallationWorker(QThread):
    """Enhanced installation worker with comprehensive error handling"""

    progress = pyqtSignal(int, str, str)  # percentage, status, detail
    finished = pyqtSignal(bool, str)  # success, message
    state_changed = pyqtSignal(str)  # state name

    def __init__(self, install_dir: str, components: List[str], config: dict):
        super().__init__()
        self.install_dir = Path(install_dir)
        self.components = components
        self.config = config
        self.state = InstallationState.NOT_STARTED
        self.rollback_actions = []

    def emit_progress(self, percentage: int, status: str, detail: str = ""):
        """Emit progress with logging"""
        self.progress.emit(percentage, status, detail)
        print(f"[{percentage}%] {status}: {detail}")

    def add_rollback_action(self, action, *args):
        """Add action to rollback stack"""
        self.rollback_actions.append((action, args))

    def rollback(self):
        """Rollback installation"""
        print("Rolling back installation...")
        for action, args in reversed(self.rollback_actions):
            try:
                action(*args)
            except Exception as e:
                print(f"Rollback action failed: {e}")

    def run(self):
        """Execute installation with comprehensive error handling"""
        try:
            # Step 1: Check/Install Python (0-20%)
            self.state = InstallationState.CHECKING_PYTHON
            self.state_changed.emit(self.state.value)

            if not self.check_and_install_python():
                self.finished.emit(False, "Failed to setup Python environment")
                self.rollback()
                return

            # Step 2: Create directories (20-25%)
            self.emit_progress(20, "Creating installation directories", str(self.install_dir))

            if not self.create_directories():
                self.finished.emit(False, "Failed to create directories")
                self.rollback()
                return

            # Step 3: Create virtual environment (25-35%)
            self.state = InstallationState.CREATING_VENV
            self.state_changed.emit(self.state.value)

            if not self.create_virtual_environment():
                self.finished.emit(False, "Failed to create virtual environment")
                self.rollback()
                return

            # Step 4: Install dependencies (35-70%)
            self.state = InstallationState.INSTALLING_PACKAGES
            self.state_changed.emit(self.state.value)

            if not self.install_dependencies():
                self.finished.emit(False, "Failed to install dependencies")
                self.rollback()
                return

            # Step 5: Copy files (70-75%)
            self.state = InstallationState.COPYING_FILES
            self.state_changed.emit(self.state.value)

            if not self.copy_application_files():
                self.finished.emit(False, "Failed to copy application files")
                self.rollback()
                return

            # Step 6: Create icons (75-80%)
            self.emit_progress(75, "Creating application icon")

            if not self.create_icon():
                print("Warning: Icon creation failed, continuing anyway")

            # Step 7: Create shortcuts (80-85%)
            self.state = InstallationState.CREATING_SHORTCUTS
            self.state_changed.emit(self.state.value)

            if not self.create_shortcuts():
                print("Warning: Shortcut creation failed, continuing anyway")

            # Step 8: Configure PATH (85-90%)
            self.state = InstallationState.CONFIGURING_PATH
            self.state_changed.emit(self.state.value)

            if not self.configure_path():
                print("Warning: PATH configuration failed, continuing anyway")

            # Step 9: Create configuration (90-95%)
            self.emit_progress(90, "Creating configuration files")

            if not self.create_configuration():
                self.finished.emit(False, "Failed to create configuration")
                self.rollback()
                return

            # Step 10: Validate installation (95-100%)
            self.state = InstallationState.VALIDATING
            self.state_changed.emit(self.state.value)

            if not self.validate_installation():
                self.finished.emit(False, "Installation validation failed")
                self.rollback()
                return

            # Complete!
            self.state = InstallationState.COMPLETED
            self.state_changed.emit(self.state.value)
            self.emit_progress(100, "Installation complete!", "")
            self.finished.emit(True, "AI Chatbot installed successfully!")

        except Exception as e:
            self.state = InstallationState.FAILED
            self.state_changed.emit(self.state.value)
            self.finished.emit(False, f"Installation failed: {str(e)}")
            self.rollback()

    def check_and_install_python(self) -> bool:
        """Check for Python and install if needed"""
        self.emit_progress(5, "Checking Python installation")

        python_installed, version, python_path = PythonManager.detect_python()

        if python_installed:
            self.emit_progress(15, f"Python {version} detected", python_path)
            self.config['python_path'] = python_path
            return True

        # Python not found, need to install
        self.emit_progress(5, "Python not found, downloading...")

        temp_dir = tempfile.mkdtemp()
        self.add_rollback_action(shutil.rmtree, temp_dir)

        def download_progress(percent, downloaded, total):
            self.emit_progress(
                5 + int(percent * 0.1),  # 5-15%
                "Downloading Python",
                f"{downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB"
            )

        installer_path = PythonManager.download_python(temp_dir, download_progress)

        if not installer_path:
            return False

        # Install Python
        self.emit_progress(15, "Installing Python (this may take a few minutes)")

        python_install_dir = self.install_dir / "Python"
        success = PythonManager.install_python_windows(str(installer_path), str(python_install_dir))

        if success:
            self.config['python_path'] = str(python_install_dir / "python.exe")
            self.emit_progress(20, "Python installed successfully")
            return True

        return False

    def create_directories(self) -> bool:
        """Create installation directory structure"""
        try:
            directories = [
                self.install_dir,
                self.install_dir / "data",
                self.install_dir / "models",
                self.install_dir / "plugins",
                self.install_dir / "logs",
                self.install_dir / "backups"
            ]

            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.add_rollback_action(shutil.rmtree, directory)

            return True

        except Exception as e:
            print(f"Directory creation failed: {e}")
            return False

    def create_virtual_environment(self) -> bool:
        """Create Python virtual environment"""
        self.emit_progress(25, "Creating virtual environment")

        try:
            venv_path = self.install_dir / "venv"
            python_exe = self.config.get('python_path', 'python')

            # Create venv
            result = subprocess.run(
                [python_exe, "-m", "venv", str(venv_path)],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                print(f"venv creation failed: {result.stderr}")
                return False

            self.emit_progress(35, "Virtual environment created")
            return True

        except Exception as e:
            print(f"Virtual environment creation failed: {e}")
            return False

    def install_dependencies(self) -> bool:
        """Install Python dependencies with error handling"""
        self.emit_progress(35, "Installing dependencies (this will take 10-15 minutes)")

        try:
            # Get pip executable
            if platform.system() == 'Windows':
                pip_exe = self.install_dir / "venv" / "Scripts" / "pip.exe"
            else:
                pip_exe = self.install_dir / "venv" / "bin" / "pip"

            if not pip_exe.exists():
                return False

            # Upgrade pip first
            self.emit_progress(40, "Upgrading pip")
            subprocess.run(
                [str(pip_exe), "install", "--upgrade", "pip"],
                capture_output=True,
                timeout=120
            )

            # Install requirements
            requirements_file = Path(__file__).parent / "requirements.txt"

            if requirements_file.exists():
                self.emit_progress(45, "Installing Python packages", "This is the longest step")

                # Install in batches for better progress tracking
                packages = requirements_file.read_text().splitlines()
                packages = [p.strip() for p in packages if p.strip() and not p.startswith('#')]

                total_packages = len(packages)
                for i, package in enumerate(packages):
                    progress = 45 + int((i / total_packages) * 25)  # 45-70%
                    self.emit_progress(progress, f"Installing {package.split('>=')[0]}")

                    try:
                        subprocess.run(
                            [str(pip_exe), "install", package],
                            capture_output=True,
                            timeout=600
                        )
                    except subprocess.TimeoutExpired:
                        print(f"Timeout installing {package}, skipping")
                        continue
                    except Exception as e:
                        print(f"Failed to install {package}: {e}")
                        # Continue with other packages

            self.emit_progress(70, "Dependencies installed")
            return True

        except Exception as e:
            print(f"Dependency installation failed: {e}")
            return False

    def copy_application_files(self) -> bool:
        """Copy application files to installation directory"""
        self.emit_progress(70, "Copying application files")

        try:
            source_dir = Path(__file__).parent

            # Files to copy
            python_files = list(source_dir.glob("*.py"))
            doc_files = list(source_dir.glob("*.md"))

            files_to_copy = python_files + doc_files + [
                source_dir / "requirements.txt",
                source_dir / "config.yaml" if (source_dir / "config.yaml").exists() else None
            ]

            files_to_copy = [f for f in files_to_copy if f and f.exists()]

            total_files = len(files_to_copy)
            for i, file in enumerate(files_to_copy):
                progress = 70 + int((i / total_files) * 5)  # 70-75%
                self.emit_progress(progress, f"Copying {file.name}")

                dest = self.install_dir / file.name
                shutil.copy2(file, dest)

            return True

        except Exception as e:
            print(f"File copying failed: {e}")
            return False

    def create_icon(self) -> bool:
        """Create application icon"""
        try:
            icon_path = self.install_dir / "icon.png"
            return IconGenerator.create_icon(str(icon_path))
        except Exception as e:
            print(f"Icon creation failed: {e}")
            return False

    def create_shortcuts(self) -> bool:
        """Create desktop and Start Menu shortcuts"""
        self.emit_progress(80, "Creating shortcuts")

        try:
            # Determine Python executable
            if platform.system() == 'Windows':
                python_exe = self.install_dir / "venv" / "Scripts" / "pythonw.exe"
            else:
                python_exe = self.install_dir / "venv" / "bin" / "python"

            # Create launcher script
            launcher_path = self.install_dir / "launch.bat" if platform.system() == 'Windows' else self.install_dir / "launch.sh"

            if platform.system() == 'Windows':
                launcher_content = f'''@echo off
cd /d "{self.install_dir}"
"{python_exe}" desktop_app.py
'''
            else:
                launcher_content = f'''#!/bin/bash
cd "{self.install_dir}"
"{python_exe}" desktop_app.py
'''

            launcher_path.write_text(launcher_content)
            if platform.system() != 'Windows':
                os.chmod(launcher_path, 0o755)

            # Create desktop shortcut
            icon_path = self.install_dir / "icon.png"
            ShortcutManager.create_desktop_shortcut(
                "AI Chatbot",
                str(launcher_path),
                str(icon_path) if icon_path.exists() else None
            )

            # Create Start Menu entry
            ShortcutManager.create_start_menu_entry(
                "AI Chatbot",
                str(launcher_path),
                str(icon_path) if icon_path.exists() else None
            )

            self.emit_progress(85, "Shortcuts created")
            return True

        except Exception as e:
            print(f"Shortcut creation failed: {e}")
            return False

    def configure_path(self) -> bool:
        """Configure system PATH"""
        self.emit_progress(85, "Configuring system PATH")

        try:
            if platform.system() == 'Windows':
                venv_scripts = self.install_dir / "venv" / "Scripts"
                PythonManager.add_to_path_windows(str(venv_scripts))

            self.emit_progress(90, "PATH configured")
            return True

        except Exception as e:
            print(f"PATH configuration failed: {e}")
            return False

    def create_configuration(self) -> bool:
        """Create configuration files"""
        try:
            config_content = f"""# AI Chatbot Configuration
install_dir: {self.install_dir}
model_name: {self.config.get('model', 'microsoft/DialoGPT-small')}
temperature: 0.8
database_path: {self.install_dir / 'data' / 'chatbot.db'}
enable_voice: true
enable_vision: true
enable_analytics: true
"""

            config_file = self.install_dir / "config.yaml"
            config_file.write_text(config_content)

            return True

        except Exception as e:
            print(f"Configuration creation failed: {e}")
            return False

    def validate_installation(self) -> bool:
        """Validate installation"""
        self.emit_progress(95, "Validating installation")

        try:
            # Check Python
            if platform.system() == 'Windows':
                python_exe = self.install_dir / "venv" / "Scripts" / "python.exe"
            else:
                python_exe = self.install_dir / "venv" / "bin" / "python"

            if not python_exe.exists():
                return False

            # Test Python import
            test_script = "import sys; import torch; import transformers; print('OK')"
            result = subprocess.run(
                [str(python_exe), "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and 'OK' in result.stdout:
                self.emit_progress(100, "Validation successful")
                return True

            return False

        except Exception as e:
            print(f"Validation failed: {e}")
            return False


# Import the original wizard pages and adapt them
from installer_wizard import (
    WelcomePage, LicensePage, InstallLocationPage,
    ComponentsPage, ConfigurationPage, CompletePage
)


class EnhancedInstallationPage(QWizardPage):
    """Enhanced installation page with detailed progress"""

    def __init__(self):
        super().__init__()
        self.setTitle("Installing AI Chatbot")
        self.setSubTitle("Please wait while installation completes...")

        layout = QVBoxLayout()

        # Overall progress
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        layout.addWidget(QLabel("Overall Progress:"))
        layout.addWidget(self.overall_progress)

        # Current task
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(self.status_label)

        # Detail label
        self.detail_label = QLabel("")
        layout.addWidget(self.detail_label)

        # Detailed log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        layout.addWidget(QLabel("Installation Log:"))
        layout.addWidget(self.log_text)

        # Estimated time remaining
        self.time_label = QLabel("")
        layout.addWidget(self.time_label)

        layout.addStretch()
        self.setLayout(layout)

        self.worker = None
        self.install_success = False
        self.start_time = None

    def initializePage(self):
        """Start enhanced installation"""
        wizard = self.wizard()

        install_location = wizard.field("install_location")

        components = ["core"]
        if wizard.field("install_venv"):
            components.append("venv")
        if wizard.field("install_deps"):
            components.append("dependencies")

        config = {
            "model": wizard.field("model_selection"),
            "enable_voice": wizard.field("install_voice"),
            "enable_vision": wizard.field("install_vision"),
        }

        # Start installation
        self.start_time = time.time()
        self.worker = EnhancedInstallationWorker(install_location, components, config)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.installation_finished)
        self.worker.state_changed.connect(self.state_changed)
        self.worker.start()

        # Disable navigation
        wizard.button(QWizard.WizardButton.BackButton).setEnabled(False)
        wizard.button(QWizard.WizardButton.CancelButton).setEnabled(False)

    def update_progress(self, percentage: int, status: str, detail: str):
        """Update progress display"""
        self.overall_progress.setValue(percentage)
        self.status_label.setText(status)
        self.detail_label.setText(detail)
        self.log_text.append(f"[{percentage}%] {status}")
        if detail:
            self.log_text.append(f"     {detail}")

        # Estimate time remaining
        if self.start_time and percentage > 0:
            elapsed = time.time() - self.start_time
            total_estimated = elapsed / (percentage / 100)
            remaining = total_estimated - elapsed

            if remaining > 60:
                self.time_label.setText(f"Estimated time remaining: {int(remaining / 60)} minutes")
            else:
                self.time_label.setText(f"Estimated time remaining: {int(remaining)} seconds")

    def state_changed(self, state: str):
        """Update when state changes"""
        state_messages = {
            "checking_python": "Checking Python installation...",
            "downloading_python": "Downloading Python installer...",
            "installing_python": "Installing Python...",
            "creating_venv": "Creating virtual environment...",
            "installing_packages": "Installing Python packages...",
            "copying_files": "Copying application files...",
            "creating_shortcuts": "Creating shortcuts...",
            "configuring_path": "Configuring system PATH...",
            "validating": "Validating installation...",
            "completed": "Installation completed!",
            "failed": "Installation failed!"
        }

        message = state_messages.get(state, state)
        self.log_text.append(f"\n{'='*50}")
        self.log_text.append(f"STATE: {message}")
        self.log_text.append(f"{'='*50}\n")

    def installation_finished(self, success: bool, message: str):
        """Handle installation completion"""
        self.install_success = success

        if success:
            self.log_text.append(f"\n✅ {message}")
            self.status_label.setText("✅ Installation completed successfully!")
            self.status_label.setStyleSheet("color: green;")
            self.wizard().button(QWizard.WizardButton.NextButton).setEnabled(True)
        else:
            self.log_text.append(f"\n❌ {message}")
            self.status_label.setText(f"❌ Installation failed")
            self.status_label.setStyleSheet("color: red;")

            QMessageBox.critical(
                self,
                "Installation Failed",
                f"Installation failed:\n\n{message}\n\nPlease check the log for details."
            )

    def isComplete(self):
        return self.install_success


class EnhancedInstallerWizard(QWizard):
    """Enhanced installer wizard"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Chatbot Installer - Production Grade")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(800, 600)

        # Add pages
        self.addPage(WelcomePage())
        self.addPage(LicensePage())
        self.addPage(InstallLocationPage())
        self.addPage(ComponentsPage())
        self.addPage(ConfigurationPage())
        self.addPage(EnhancedInstallationPage())  # Enhanced!
        self.addPage(CompletePage())

        self.setButtonText(QWizard.WizardButton.FinishButton, "Finish")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("AI Chatbot Installer")
    app.setOrganizationName("AI Research")

    wizard = EnhancedInstallerWizard()
    wizard.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
