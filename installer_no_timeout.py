"""
Enhanced AI Chatbot Installer - Fixed Timeouts
Production-grade installer with NO TIMEOUTS on critical operations
Handles slow connections, large downloads, and everything automatically
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
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from enum import Enum

try:
    import winreg
    WINREG_AVAILABLE = True
except ImportError:
    WINREG_AVAILABLE = False

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


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
    """Handle network operations with retries and fallbacks - NO TIMEOUTS"""

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
    def download_with_retry(url: str, dest: str, max_retries: int = 10,
                           progress_callback=None, timeout: Optional[int] = None) -> bool:
        """Download file with retry logic - INCREASED retries, OPTIONAL timeout"""
        for attempt in range(max_retries):
            try:
                # Create request with headers
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 AI-Chatbot-Installer/1.0'}
                )

                # Open connection - NO TIMEOUT by default for slow connections
                with urllib.request.urlopen(req, timeout=timeout) as response:
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

            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                    print(f"Download attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Failed to download after {max_retries} attempts: {e}")
                    return False

            except Exception as e:
                print(f"Unexpected error during download: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return False

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
        # Try python3 first
        for python_cmd in ['python3', 'python']:
            try:
                result = subprocess.run(
                    [python_cmd, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    version = result.stdout.strip().split()[1]
                    python_path = shutil.which(python_cmd)

                    # Check version is >= 3.8
                    try:
                        major, minor = map(int, version.split('.')[:2])
                        if major >= 3 and minor >= 8:
                            return True, version, python_path
                    except ValueError:
                        continue

            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return False, None, None

    @staticmethod
    def download_python(dest_dir: str, progress_callback=None) -> Optional[str]:
        """Download Python installer - NO TIMEOUT for slow connections"""
        system = platform.system()
        machine = platform.machine()

        if system == 'Windows':
            arch_key = 'windows_x64' if machine.endswith('64') else 'windows_x86'
            url = PythonManager.PYTHON_VERSIONS[arch_key]['3.11']

            dest_file = os.path.join(dest_dir, 'python_installer.exe')

            print(f"Downloading Python from {url}...")
            # NO TIMEOUT - let it take as long as needed
            success = NetworkManager.download_with_retry(
                url, dest_file, max_retries=10, progress_callback=progress_callback, timeout=None
            )

            return dest_file if success else None

        return None

    @staticmethod
    def install_python_windows(installer_path: str, install_dir: str) -> bool:
        """Install Python silently on Windows - NO TIMEOUT"""
        try:
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

            # NO TIMEOUT - Python installation can take time
            result = subprocess.run(cmd, check=True, capture_output=True)
            return result.returncode == 0

        except subprocess.CalledProcessError as e:
            print(f"Python installation failed: {e}")
            return False

    @staticmethod
    def add_to_path_windows(directory: str) -> bool:
        """Add directory to Windows PATH"""
        if not WINREG_AVAILABLE:
            print("Windows registry access not available")
            return False

        try:
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

            paths = current_path.split(os.pathsep)
            if directory not in paths:
                paths.append(directory)
                new_path = os.pathsep.join(paths)
                winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)

            winreg.CloseKey(key)

            # Broadcast environment change
            try:
                HWND_BROADCAST = 0xFFFF
                WM_SETTINGCHANGE = 0x001A
                ctypes.windll.user32.SendMessageW(
                    HWND_BROADCAST, WM_SETTINGCHANGE, 0, 'Environment'
                )
            except:
                pass  # Not critical

            return True

        except Exception as e:
            print(f"Failed to add to PATH: {e}")
            return False


# Import enhanced installer classes from original file
from installer_wizard import (
    WelcomePage, LicensePage, InstallLocationPage,
    ComponentsPage, ConfigurationPage, CompletePage,
    IconGenerator, ShortcutManager
)


class NoTimeoutInstallationWorker(QThread):
    """Installation worker with NO TIMEOUTS on critical operations"""

    progress = pyqtSignal(int, str, str)
    finished = pyqtSignal(bool, str)
    state_changed = pyqtSignal(str)

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
        """Execute installation with NO TIMEOUTS"""
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

            # Step 4: Install dependencies (35-70%) - NO TIMEOUT!
            self.state = InstallationState.INSTALLING_PACKAGES
            self.state_changed.emit(self.state.value)

            if not self.install_dependencies_no_timeout():
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
            self.create_icon()  # Non-critical, continue anyway

            # Step 7: Create shortcuts (80-85%)
            self.state = InstallationState.CREATING_SHORTCUTS
            self.state_changed.emit(self.state.value)
            self.create_shortcuts()  # Non-critical

            # Step 8: Configure PATH (85-90%)
            self.state = InstallationState.CONFIGURING_PATH
            self.state_changed.emit(self.state.value)
            self.configure_path()  # Non-critical

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
        """Check for Python and install if needed - NO TIMEOUT"""
        self.emit_progress(5, "Checking Python installation")

        python_installed, version, python_path = PythonManager.detect_python()

        if python_installed:
            self.emit_progress(15, f"Python {version} detected", python_path)
            self.config['python_path'] = python_path
            return True

        # Python not found, need to install
        self.emit_progress(5, "Python not found, downloading (this may take a while on slow connections)...")

        temp_dir = tempfile.mkdtemp()
        self.add_rollback_action(shutil.rmtree, temp_dir)

        def download_progress(percent, downloaded, total):
            self.emit_progress(
                5 + int(percent * 0.1),
                "Downloading Python (no timeout - will wait for slow connections)",
                f"{downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB"
            )

        installer_path = PythonManager.download_python(temp_dir, download_progress)

        if not installer_path:
            return False

        self.emit_progress(15, "Installing Python (no timeout - may take several minutes)")

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
        """Create Python virtual environment - SHORT timeout OK (local operation)"""
        self.emit_progress(25, "Creating virtual environment")

        try:
            venv_path = self.install_dir / "venv"
            python_exe = self.config.get('python_path', 'python')

            result = subprocess.run(
                [python_exe, "-m", "venv", str(venv_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes max for venv creation (local operation)
            )

            if result.returncode != 0:
                print(f"venv creation failed: {result.stderr}")
                return False

            self.emit_progress(35, "Virtual environment created")
            return True

        except Exception as e:
            print(f"Virtual environment creation failed: {e}")
            return False

    def install_dependencies_no_timeout(self) -> bool:
        """Install Python dependencies with NO TIMEOUT - handles slow connections"""
        self.emit_progress(35, "Installing dependencies (NO TIMEOUT - may take 30+ minutes on slow connections)")

        try:
            if platform.system() == 'Windows':
                pip_exe = self.install_dir / "venv" / "Scripts" / "pip.exe"
            else:
                pip_exe = self.install_dir / "venv" / "bin" / "pip"

            if not pip_exe.exists():
                return False

            # Upgrade pip - short timeout OK (small package)
            self.emit_progress(40, "Upgrading pip")
            subprocess.run(
                [str(pip_exe), "install", "--upgrade", "pip"],
                capture_output=True,
                timeout=300  # 5 minutes max for pip upgrade
            )

            # Install requirements - NO TIMEOUT!
            requirements_file = Path(__file__).parent / "requirements.txt"

            if requirements_file.exists():
                self.emit_progress(45, "Installing Python packages (NO TIMEOUT - waiting for slow downloads)")

                packages = requirements_file.read_text().splitlines()
                packages = [p.strip() for p in packages if p.strip() and not p.startswith('#')]

                total_packages = len(packages)
                for i, package in enumerate(packages):
                    progress = 45 + int((i / total_packages) * 25)
                    self.emit_progress(
                        progress,
                        f"Installing {package.split('>=')[0]}",
                        "NO TIMEOUT - This may take a long time on slow connections. Please be patient!"
                    )

                    try:
                        # NO TIMEOUT! Let it take as long as needed
                        process = subprocess.Popen(
                            [str(pip_exe), "install", package, "--no-cache-dir"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )

                        # Wait indefinitely
                        stdout, stderr = process.communicate()

                        if process.returncode != 0:
                            print(f"Warning: Failed to install {package}: {stderr}")
                            # Continue with other packages
                            continue

                    except Exception as e:
                        print(f"Error installing {package}: {e}")
                        # Continue with other packages
                        continue

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

            python_files = list(source_dir.glob("*.py"))
            doc_files = list(source_dir.glob("*.md"))

            files_to_copy = python_files + doc_files + [
                source_dir / "requirements.txt",
            ]

            files_to_copy = [f for f in files_to_copy if f and f.exists()]

            total_files = len(files_to_copy)
            for i, file in enumerate(files_to_copy):
                progress = 70 + int((i / total_files) * 5)
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
            if platform.system() == 'Windows':
                python_exe = self.install_dir / "venv" / "Scripts" / "pythonw.exe"
            else:
                python_exe = self.install_dir / "venv" / "bin" / "python"

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

            icon_path = self.install_dir / "icon.png"
            ShortcutManager.create_desktop_shortcut(
                "AI Chatbot",
                str(launcher_path),
                str(icon_path) if icon_path.exists() else None
            )

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
        """Validate installation - SHORT timeout OK (just testing imports)"""
        self.emit_progress(95, "Validating installation")

        try:
            if platform.system() == 'Windows':
                python_exe = self.install_dir / "venv" / "Scripts" / "python.exe"
            else:
                python_exe = self.install_dir / "venv" / "bin" / "python"

            if not python_exe.exists():
                return False

            test_script = "import sys; import torch; import transformers; print('OK')"
            result = subprocess.run(
                [str(python_exe), "-c", test_script],
                capture_output=True,
                text=True,
                timeout=60  # 1 minute for imports
            )

            if result.returncode == 0 and 'OK' in result.stdout:
                self.emit_progress(100, "Validation successful")
                return True

            return False

        except Exception as e:
            print(f"Validation failed: {e}")
            return False


class EnhancedInstallationPage(QWizardPage):
    """Enhanced installation page with NO TIMEOUT progress"""

    def __init__(self):
        super().__init__()
        self.setTitle("Installing AI Chatbot")
        self.setSubTitle("NO TIMEOUTS - Installation will wait for slow connections")

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
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

        # Warning about slow connections
        warning = QLabel("⚠️ For slow internet: Downloads may take 30+ minutes. NO TIMEOUTS - Please be patient!")
        warning.setStyleSheet("color: orange; font-weight: bold;")
        layout.addWidget(warning)

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
        """Start NO TIMEOUT installation"""
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

        self.start_time = time.time()
        self.worker = NoTimeoutInstallationWorker(install_location, components, config)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.installation_finished)
        self.worker.state_changed.connect(self.state_changed)
        self.worker.start()

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

        if self.start_time and percentage > 0:
            elapsed = time.time() - self.start_time
            if elapsed > 60:
                self.time_label.setText(f"Elapsed time: {int(elapsed / 60)} minutes {int(elapsed % 60)} seconds")
            else:
                self.time_label.setText(f"Elapsed time: {int(elapsed)} seconds")

    def state_changed(self, state: str):
        """Update when state changes"""
        self.log_text.append(f"\n{'='*50}")
        self.log_text.append(f"STATE: {state.upper()}")
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
                f"Installation failed:\n\n{message}\n\nCheck the log for details."
            )

    def isComplete(self):
        return self.install_success


class NoTimeoutInstallerWizard(QWizard):
    """Installer wizard with NO TIMEOUTS"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Chatbot Installer - NO TIMEOUTS (Slow Connection Friendly)")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(900, 700)

        self.addPage(WelcomePage())
        self.addPage(LicensePage())
        self.addPage(InstallLocationPage())
        self.addPage(ComponentsPage())
        self.addPage(ConfigurationPage())
        self.addPage(EnhancedInstallationPage())
        self.addPage(CompletePage())

        self.setButtonText(QWizard.WizardButton.FinishButton, "Finish")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("AI Chatbot Installer (No Timeouts)")
    app.setOrganizationName("AI Research")

    wizard = NoTimeoutInstallerWizard()
    wizard.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
