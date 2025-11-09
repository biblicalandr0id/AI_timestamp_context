"""
Desktop Installation Wizard
Professional GUI-based installer for AI Chatbot
Multi-step wizard with progress tracking, dependency checking, and configuration
"""

import sys
import os
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

try:
    from PyQt6.QtWidgets import (
        QApplication, QWizard, QWizardPage, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QCheckBox, QRadioButton,
        QTextEdit, QProgressBar, QFileDialog, QGroupBox, QMessageBox,
        QButtonGroup, QComboBox
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QPixmap
except ImportError:
    print("PyQt6 not installed. Install with: pip install PyQt6")
    sys.exit(1)


class InstallationWorker(QThread):
    """Background worker for installation tasks"""
    progress = pyqtSignal(int, str)  # progress percentage, status message
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, install_dir: str, components: List[str], config: dict):
        super().__init__()
        self.install_dir = Path(install_dir)
        self.components = components
        self.config = config

    def run(self):
        """Run installation"""
        try:
            # Step 1: Create directories
            self.progress.emit(10, "Creating installation directories...")
            self.install_dir.mkdir(parents=True, exist_ok=True)
            (self.install_dir / "data").mkdir(exist_ok=True)
            (self.install_dir / "models").mkdir(exist_ok=True)
            (self.install_dir / "plugins").mkdir(exist_ok=True)
            (self.install_dir / "logs").mkdir(exist_ok=True)

            # Step 2: Check Python
            self.progress.emit(20, "Checking Python installation...")
            python_version = sys.version_info
            if python_version < (3, 8):
                self.finished.emit(False, f"Python 3.8+ required. Found {python_version.major}.{python_version.minor}")
                return

            # Step 3: Create virtual environment
            if "venv" in self.components:
                self.progress.emit(30, "Creating virtual environment...")
                venv_path = self.install_dir / "venv"
                try:
                    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
                except Exception as e:
                    self.finished.emit(False, f"Failed to create venv: {e}")
                    return

            # Step 4: Install dependencies
            if "dependencies" in self.components:
                self.progress.emit(50, "Installing dependencies (this may take several minutes)...")

                # Determine pip executable
                if platform.system() == "Windows":
                    pip_exe = self.install_dir / "venv" / "Scripts" / "pip.exe"
                else:
                    pip_exe = self.install_dir / "venv" / "bin" / "pip"

                if pip_exe.exists():
                    try:
                        # Upgrade pip first
                        subprocess.run([str(pip_exe), "install", "--upgrade", "pip"],
                                     check=True, capture_output=True)

                        # Install requirements
                        requirements_file = Path(__file__).parent / "requirements.txt"
                        if requirements_file.exists():
                            subprocess.run([str(pip_exe), "install", "-r", str(requirements_file)],
                                         check=True, capture_output=True, timeout=600)
                        else:
                            # Install core packages
                            core_packages = [
                                "torch", "transformers", "sentence-transformers",
                                "flask", "flask-socketio", "PyQt6",
                                "numpy", "pandas", "plotly"
                            ]
                            subprocess.run([str(pip_exe), "install"] + core_packages,
                                         check=True, capture_output=True, timeout=600)
                    except subprocess.TimeoutExpired:
                        self.finished.emit(False, "Installation timeout. Please try again with better internet connection.")
                        return
                    except Exception as e:
                        self.finished.emit(False, f"Failed to install dependencies: {e}")
                        return

            # Step 5: Copy application files
            self.progress.emit(70, "Copying application files...")
            source_dir = Path(__file__).parent
            files_to_copy = [
                "neural_chatbot.py",
                "knowledge_store.py",
                "rag_system.py",
                "chatbot_server.py",
                "launch_chatbot.py",
                "desktop_app.py",
                "voice_interface.py",
                "vision_interface.py",
                "plugin_system.py",
                "analytics_dashboard.py",
                "knowledge_graph_visualizer.py"
            ]

            for file in files_to_copy:
                src = source_dir / file
                if src.exists():
                    shutil.copy2(src, self.install_dir / file)

            # Copy documentation
            docs_to_copy = ["README.md", "CHATBOT.md", "QUICKSTART.md", "SOTA_COMPARISON.md"]
            for doc in docs_to_copy:
                src = source_dir / doc
                if src.exists():
                    shutil.copy2(src, self.install_dir / doc)

            # Step 6: Create configuration file
            self.progress.emit(80, "Creating configuration...")
            config_content = f"""# AI Chatbot Configuration
install_dir: {self.install_dir}
model_name: {self.config.get('model', 'microsoft/DialoGPT-small')}
temperature: {self.config.get('temperature', 0.8)}
database_path: {self.install_dir / 'data' / 'chatbot.db'}
enable_voice: {self.config.get('enable_voice', True)}
enable_vision: {self.config.get('enable_vision', True)}
enable_analytics: {self.config.get('enable_analytics', True)}
"""
            config_file = self.install_dir / "config.yaml"
            config_file.write_text(config_content)

            # Step 7: Create shortcuts
            self.progress.emit(90, "Creating shortcuts...")
            self._create_shortcuts()

            # Step 8: Download initial model (optional)
            if "download_model" in self.components:
                self.progress.emit(95, "Downloading language model (first time only)...")
                # This will be done on first launch to avoid long wait

            # Complete
            self.progress.emit(100, "Installation complete!")
            self.finished.emit(True, "AI Chatbot installed successfully!")

        except Exception as e:
            self.finished.emit(False, f"Installation failed: {str(e)}")

    def _create_shortcuts(self):
        """Create application shortcuts"""
        if platform.system() == "Windows":
            self._create_windows_shortcut()
        elif platform.system() == "Darwin":
            self._create_mac_shortcut()
        else:
            self._create_linux_shortcut()

    def _create_windows_shortcut(self):
        """Create Windows shortcut"""
        try:
            import winshell
            from win32com.client import Dispatch

            desktop = winshell.desktop()
            shortcut_path = os.path.join(desktop, "AI Chatbot.lnk")

            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = str(self.install_dir / "venv" / "Scripts" / "python.exe")
            shortcut.Arguments = str(self.install_dir / "desktop_app.py")
            shortcut.WorkingDirectory = str(self.install_dir)
            shortcut.save()
        except:
            pass  # Shortcut creation is optional

    def _create_mac_shortcut(self):
        """Create macOS app bundle"""
        # macOS app creation would be done via separate script
        pass

    def _create_linux_shortcut(self):
        """Create Linux desktop entry"""
        try:
            desktop_entry = f"""[Desktop Entry]
Name=AI Chatbot
Comment=State-of-the-art AI Chatbot
Exec={self.install_dir}/venv/bin/python {self.install_dir}/desktop_app.py
Icon={self.install_dir}/icon.png
Terminal=false
Type=Application
Categories=Utility;Development;
"""
            desktop_dir = Path.home() / ".local" / "share" / "applications"
            desktop_dir.mkdir(parents=True, exist_ok=True)
            (desktop_dir / "ai-chatbot.desktop").write_text(desktop_entry)
        except:
            pass  # Desktop entry is optional


class WelcomePage(QWizardPage):
    """Welcome page"""

    def __init__(self):
        super().__init__()
        self.setTitle("Welcome to AI Chatbot Installer")
        self.setSubTitle("This wizard will guide you through the installation process.")

        layout = QVBoxLayout()

        # Logo/Banner (if available)
        banner = QLabel()
        banner.setText("🤖 AI CHATBOT")
        banner.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(banner)

        # Description
        desc = QLabel(
            "AI Chatbot is a state-of-the-art neural network chatbot with:\n\n"
            "✓ Real-time continual learning\n"
            "✓ Voice and vision capabilities\n"
            "✓ Interactive knowledge graphs\n"
            "✓ Comprehensive analytics\n"
            "✓ Extensible plugin system\n"
            "✓ Complete privacy (100% local)\n\n"
            "Click 'Next' to begin installation."
        )
        desc.setWordWrap(True)
        desc.setFont(QFont("Arial", 11))
        layout.addWidget(desc)

        layout.addStretch()
        self.setLayout(layout)


class LicensePage(QWizardPage):
    """License agreement page"""

    def __init__(self):
        super().__init__()
        self.setTitle("License Agreement")
        self.setSubTitle("Please review the license terms.")

        layout = QVBoxLayout()

        # License text
        license_text = QTextEdit()
        license_text.setReadOnly(True)
        license_text.setPlainText("""MIT License

Copyright (c) 2025 AI Chatbot Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")
        layout.addWidget(license_text)

        # Acceptance checkbox
        self.accept_checkbox = QCheckBox("I accept the terms of the license agreement")
        self.accept_checkbox.stateChanged.connect(self.checkComplete)
        layout.addWidget(self.accept_checkbox)

        self.setLayout(layout)
        self.registerField("license_accepted*", self.accept_checkbox)

    def checkComplete(self):
        self.completeChanged.emit()

    def isComplete(self):
        return self.accept_checkbox.isChecked()


class InstallLocationPage(QWizardPage):
    """Installation location page"""

    def __init__(self):
        super().__init__()
        self.setTitle("Installation Location")
        self.setSubTitle("Choose where to install AI Chatbot.")

        layout = QVBoxLayout()

        # Default location
        if platform.system() == "Windows":
            default_path = os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"), "AI_Chatbot")
        else:
            default_path = os.path.join(str(Path.home()), "AI_Chatbot")

        # Location input
        loc_layout = QHBoxLayout()
        loc_layout.addWidget(QLabel("Install to:"))

        self.location_edit = QLineEdit(default_path)
        self.location_edit.textChanged.connect(self.checkComplete)
        loc_layout.addWidget(self.location_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_location)
        loc_layout.addWidget(browse_btn)

        layout.addLayout(loc_layout)

        # Space info
        self.space_label = QLabel()
        self.update_space_info()
        layout.addWidget(self.space_label)

        layout.addStretch()
        self.setLayout(layout)
        self.registerField("install_location*", self.location_edit)

    def browse_location(self):
        """Browse for installation location"""
        directory = QFileDialog.getExistingDirectory(self, "Select Installation Directory")
        if directory:
            self.location_edit.setText(os.path.join(directory, "AI_Chatbot"))
            self.update_space_info()

    def update_space_info(self):
        """Update disk space information"""
        try:
            path = Path(self.location_edit.text())
            parent = path.parent if not path.exists() else path

            if parent.exists():
                stat = shutil.disk_usage(parent)
                free_gb = stat.free / (1024**3)
                self.space_label.setText(f"Available space: {free_gb:.1f} GB (Requires ~5 GB)")

                if free_gb < 5:
                    self.space_label.setStyleSheet("color: red;")
                else:
                    self.space_label.setStyleSheet("color: green;")
        except:
            self.space_label.setText("Cannot determine available space")

    def checkComplete(self):
        self.completeChanged.emit()

    def isComplete(self):
        path = self.location_edit.text()
        return len(path) > 0


class ComponentsPage(QWizardPage):
    """Components selection page"""

    def __init__(self):
        super().__init__()
        self.setTitle("Select Components")
        self.setSubTitle("Choose which components to install.")

        layout = QVBoxLayout()

        # Core components (required)
        core_group = QGroupBox("Core Components (Required)")
        core_layout = QVBoxLayout()

        self.core_checkbox = QCheckBox("AI Chatbot Core (Required)")
        self.core_checkbox.setChecked(True)
        self.core_checkbox.setEnabled(False)
        core_layout.addWidget(self.core_checkbox)

        self.venv_checkbox = QCheckBox("Create Virtual Environment")
        self.venv_checkbox.setChecked(True)
        core_layout.addWidget(self.venv_checkbox)

        self.deps_checkbox = QCheckBox("Install Dependencies (Recommended)")
        self.deps_checkbox.setChecked(True)
        core_layout.addWidget(self.deps_checkbox)

        core_group.setLayout(core_layout)
        layout.addWidget(core_group)

        # Optional components
        optional_group = QGroupBox("Optional Components")
        optional_layout = QVBoxLayout()

        self.desktop_checkbox = QCheckBox("Desktop Application")
        self.desktop_checkbox.setChecked(True)
        optional_layout.addWidget(self.desktop_checkbox)

        self.voice_checkbox = QCheckBox("Voice Interface (Speech Recognition + TTS)")
        self.voice_checkbox.setChecked(True)
        optional_layout.addWidget(self.voice_checkbox)

        self.vision_checkbox = QCheckBox("Vision/Image Understanding (CLIP + BLIP)")
        self.vision_checkbox.setChecked(True)
        optional_layout.addWidget(self.vision_checkbox)

        self.analytics_checkbox = QCheckBox("Analytics Dashboard")
        self.analytics_checkbox.setChecked(True)
        optional_layout.addWidget(self.analytics_checkbox)

        self.plugins_checkbox = QCheckBox("Plugin System")
        self.plugins_checkbox.setChecked(True)
        optional_layout.addWidget(self.plugins_checkbox)

        optional_group.setLayout(optional_layout)
        layout.addWidget(optional_group)

        # Space required
        self.size_label = QLabel("Total download size: ~2-4 GB")
        layout.addWidget(self.size_label)

        layout.addStretch()
        self.setLayout(layout)

        self.registerField("install_venv", self.venv_checkbox)
        self.registerField("install_deps", self.deps_checkbox)
        self.registerField("install_desktop", self.desktop_checkbox)
        self.registerField("install_voice", self.voice_checkbox)
        self.registerField("install_vision", self.vision_checkbox)


class ConfigurationPage(QWizardPage):
    """Configuration page"""

    def __init__(self):
        super().__init__()
        self.setTitle("Configuration")
        self.setSubTitle("Configure AI Chatbot settings.")

        layout = QVBoxLayout()

        # Model selection
        model_group = QGroupBox("Language Model")
        model_layout = QVBoxLayout()

        model_layout.addWidget(QLabel("Select model size (can be changed later):"))

        self.model_combo = QComboBox()
        self.model_combo.addItem("Small (117M params, ~500MB, Fast)", "microsoft/DialoGPT-small")
        self.model_combo.addItem("Medium (345M params, ~1.5GB, Balanced)", "microsoft/DialoGPT-medium")
        self.model_combo.addItem("Large (762M params, ~3GB, Best Quality)", "microsoft/DialoGPT-large")
        model_layout.addWidget(self.model_combo)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Shortcuts
        shortcuts_group = QGroupBox("Shortcuts")
        shortcuts_layout = QVBoxLayout()

        self.desktop_shortcut = QCheckBox("Create desktop shortcut")
        self.desktop_shortcut.setChecked(True)
        shortcuts_layout.addWidget(self.desktop_shortcut)

        self.start_menu = QCheckBox("Add to Start Menu / Applications")
        self.start_menu.setChecked(True)
        shortcuts_layout.addWidget(self.start_menu)

        shortcuts_group.setLayout(shortcuts_layout)
        layout.addWidget(shortcuts_group)

        # Auto-start options
        autostart_group = QGroupBox("Additional Options")
        autostart_layout = QVBoxLayout()

        self.launch_after = QCheckBox("Launch AI Chatbot after installation")
        self.launch_after.setChecked(True)
        autostart_layout.addWidget(self.launch_after)

        self.show_readme = QCheckBox("Show README file")
        self.show_readme.setChecked(True)
        autostart_layout.addWidget(self.show_readme)

        autostart_group.setLayout(autostart_layout)
        layout.addWidget(autostart_group)

        layout.addStretch()
        self.setLayout(layout)

        self.registerField("model_selection", self.model_combo, "currentData")
        self.registerField("launch_after_install", self.launch_after)


class InstallationPage(QWizardPage):
    """Installation progress page"""

    def __init__(self):
        super().__init__()
        self.setTitle("Installing")
        self.setSubTitle("Please wait while AI Chatbot is being installed...")

        layout = QVBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Initializing installation...")
        layout.addWidget(self.status_label)

        # Detailed log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)

        layout.addStretch()
        self.setLayout(layout)

        self.worker = None
        self.install_success = False

    def initializePage(self):
        """Start installation when page is shown"""
        # Get configuration from wizard
        wizard = self.wizard()
        install_location = wizard.field("install_location")

        # Determine components to install
        components = ["core"]
        if wizard.field("install_venv"):
            components.append("venv")
        if wizard.field("install_deps"):
            components.append("dependencies")

        # Configuration
        config = {
            "model": wizard.field("model_selection"),
            "enable_voice": wizard.field("install_voice"),
            "enable_vision": wizard.field("install_vision"),
        }

        # Start installation worker
        self.worker = InstallationWorker(install_location, components, config)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.installation_finished)
        self.worker.start()

        # Disable back/cancel during installation
        wizard.button(QWizard.WizardButton.BackButton).setEnabled(False)
        wizard.button(QWizard.WizardButton.CancelButton).setEnabled(False)

    def update_progress(self, value: int, message: str):
        """Update progress display"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        self.log_text.append(f"[{value}%] {message}")

    def installation_finished(self, success: bool, message: str):
        """Handle installation completion"""
        self.install_success = success

        if success:
            self.log_text.append(f"\n✅ {message}")
            self.status_label.setText("✅ Installation completed successfully!")
            self.wizard().button(QWizard.WizardButton.NextButton).setEnabled(True)
        else:
            self.log_text.append(f"\n❌ {message}")
            self.status_label.setText(f"❌ Installation failed: {message}")
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")

            # Show error dialog
            QMessageBox.critical(
                self,
                "Installation Failed",
                f"Installation failed:\n\n{message}\n\nPlease check the log for details."
            )

    def isComplete(self):
        """Page is complete when installation is done"""
        return self.install_success


class CompletePage(QWizardPage):
    """Completion page"""

    def __init__(self):
        super().__init__()
        self.setTitle("Installation Complete")
        self.setSubTitle("AI Chatbot has been successfully installed!")

        layout = QVBoxLayout()

        # Success message
        success = QLabel("✅ AI Chatbot Installation Complete!")
        success.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        success.setStyleSheet("color: green;")
        layout.addWidget(success)

        # Instructions
        instructions = QLabel(
            "\n🎉 Congratulations! AI Chatbot is now installed.\n\n"
            "To launch AI Chatbot:\n"
            "• Click the desktop shortcut (if created)\n"
            "• Find it in your Start Menu / Applications\n"
            "• Or run: python desktop_app.py\n\n"
            "For help and documentation:\n"
            "• Read QUICKSTART.md for a quick guide\n"
            "• Read CHATBOT.md for complete documentation\n"
            "• Read SOTA_COMPARISON.md to see what makes this special\n\n"
            "Thank you for installing AI Chatbot!"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Launch checkbox
        self.launch_checkbox = QCheckBox("Launch AI Chatbot now")
        self.launch_checkbox.setChecked(True)
        layout.addWidget(self.launch_checkbox)

        layout.addStretch()
        self.setLayout(layout)

    def validatePage(self):
        """Launch app if checkbox is checked"""
        if self.launch_checkbox.isChecked():
            install_location = self.wizard().field("install_location")

            try:
                # Launch desktop app
                if platform.system() == "Windows":
                    python_exe = os.path.join(install_location, "venv", "Scripts", "python.exe")
                else:
                    python_exe = os.path.join(install_location, "venv", "bin", "python")

                app_path = os.path.join(install_location, "desktop_app.py")

                if os.path.exists(python_exe) and os.path.exists(app_path):
                    subprocess.Popen([python_exe, app_path])
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Launch Failed",
                    f"Could not launch application:\n{e}\n\nPlease launch manually."
                )

        return True


class InstallerWizard(QWizard):
    """Main installer wizard"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Chatbot Installer")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(700, 500)

        # Add pages
        self.addPage(WelcomePage())
        self.addPage(LicensePage())
        self.addPage(InstallLocationPage())
        self.addPage(ComponentsPage())
        self.addPage(ConfigurationPage())
        self.addPage(InstallationPage())
        self.addPage(CompletePage())

        # Set button text
        self.setButtonText(QWizard.WizardButton.FinishButton, "Finish")
        self.setButtonText(QWizard.WizardButton.CancelButton, "Cancel")
        self.setButtonText(QWizard.WizardButton.BackButton, "< Back")
        self.setButtonText(QWizard.WizardButton.NextButton, "Next >")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("AI Chatbot Installer")

    wizard = InstallerWizard()
    wizard.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
