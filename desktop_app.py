"""
Professional Desktop Application with PyQt6
Beautiful, native-feeling desktop app for the AI Chatbot
"""

import sys
import json
from datetime import datetime
from typing import Optional
from pathlib import Path

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter,
        QListWidget, QTabWidget, QMenuBar, QMenu, QStatusBar,
        QSystemTrayIcon, QStyle, QMessageBox, QDialog, QSpinBox,
        QComboBox, QCheckBox, QProgressBar, QSlider, QGroupBox
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
    from PyQt6.QtGui import QIcon, QFont, QColor, QPalette, QAction, QTextCursor
except ImportError:
    print("PyQt6 not installed. Install with: pip install PyQt6")
    sys.exit(1)

# Import chatbot components
from rag_system import create_rag_system
from neural_chatbot import ChatbotConfig
from rag_system import RAGConfig


class ChatWorker(QThread):
    """Background thread for chatbot processing"""
    response_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, rag_system, user_input, conversation_history):
        super().__init__()
        self.rag_system = rag_system
        self.user_input = user_input
        self.conversation_history = conversation_history

    def run(self):
        try:
            result = self.rag_system.generate_with_retrieval(
                user_input=self.user_input,
                conversation_history=self.conversation_history,
                user_id="desktop_user"
            )
            self.response_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class SettingsDialog(QDialog):
    """Settings configuration dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Settings")
        self.setModal(True)
        self.resize(500, 400)

        layout = QVBoxLayout()

        # Model Settings
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large"
        ])
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_combo)

        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setMinimum(0)
        self.temperature_slider.setMaximum(200)
        self.temperature_slider.setValue(80)
        self.temperature_label = QLabel("Temperature: 0.8")
        self.temperature_slider.valueChanged.connect(
            lambda v: self.temperature_label.setText(f"Temperature: {v/100:.1f}")
        )
        model_layout.addWidget(self.temperature_label)
        model_layout.addWidget(self.temperature_slider)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # RAG Settings
        rag_group = QGroupBox("RAG Settings")
        rag_layout = QVBoxLayout()

        self.retrieval_spin = QSpinBox()
        self.retrieval_spin.setMinimum(1)
        self.retrieval_spin.setMaximum(20)
        self.retrieval_spin.setValue(5)
        rag_layout.addWidget(QLabel("Retrieval Top-K:"))
        rag_layout.addWidget(self.retrieval_spin)

        self.knowledge_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.knowledge_weight_slider.setMinimum(0)
        self.knowledge_weight_slider.setMaximum(100)
        self.knowledge_weight_slider.setValue(70)
        self.knowledge_weight_label = QLabel("Knowledge Weight: 0.7")
        self.knowledge_weight_slider.valueChanged.connect(
            lambda v: self.knowledge_weight_label.setText(f"Knowledge Weight: {v/100:.1f}")
        )
        rag_layout.addWidget(self.knowledge_weight_label)
        rag_layout.addWidget(self.knowledge_weight_slider)

        rag_group.setLayout(rag_layout)
        layout.addWidget(rag_group)

        # UI Settings
        ui_group = QGroupBox("Interface Settings")
        ui_layout = QVBoxLayout()

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "Blue", "Green"])
        ui_layout.addWidget(QLabel("Theme:"))
        ui_layout.addWidget(self.theme_combo)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setMinimum(8)
        self.font_size_spin.setMaximum(24)
        self.font_size_spin.setValue(12)
        ui_layout.addWidget(QLabel("Font Size:"))
        ui_layout.addWidget(self.font_size_spin)

        ui_group.setLayout(ui_layout)
        layout.addWidget(ui_group)

        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_settings(self):
        return {
            'model': self.model_combo.currentText(),
            'temperature': self.temperature_slider.value() / 100,
            'retrieval_top_k': self.retrieval_spin.value(),
            'knowledge_weight': self.knowledge_weight_slider.value() / 100,
            'theme': self.theme_combo.currentText(),
            'font_size': self.font_size_spin.value()
        }


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.rag_system = None
        self.conversation_history = []
        self.current_worker = None

        self.init_ui()
        self.init_chatbot()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🧠 AI Chatbot - Desktop")
        self.setGeometry(100, 100, 1200, 800)

        # Create menu bar
        self.create_menus()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()

        # Left panel - Conversation list
        left_panel = self.create_left_panel()

        # Center panel - Chat
        center_panel = self.create_center_panel()

        # Right panel - Info
        right_panel = self.create_right_panel()

        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # System tray
        self.create_system_tray()

        # Apply default theme
        self.apply_theme("Light")

    def create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("📁 File")

        new_chat_action = QAction("🆕 New Chat", self)
        new_chat_action.setShortcut("Ctrl+N")
        new_chat_action.triggered.connect(self.new_chat)
        file_menu.addAction(new_chat_action)

        save_chat_action = QAction("💾 Save Chat", self)
        save_chat_action.setShortcut("Ctrl+S")
        save_chat_action.triggered.connect(self.save_chat)
        file_menu.addAction(save_chat_action)

        load_chat_action = QAction("📂 Load Chat", self)
        load_chat_action.setShortcut("Ctrl+O")
        load_chat_action.triggered.connect(self.load_chat)
        file_menu.addAction(load_chat_action)

        file_menu.addSeparator()

        export_action = QAction("📤 Export", self)
        export_action.triggered.connect(self.export_conversation)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("🚪 Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("✏️ Edit")

        clear_action = QAction("🗑️ Clear Chat", self)
        clear_action.triggered.connect(self.clear_chat)
        edit_menu.addAction(clear_action)

        settings_action = QAction("⚙️ Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings)
        edit_menu.addAction(settings_action)

        # View menu
        view_menu = menubar.addMenu("👁️ View")

        themes = ["Light", "Dark", "Blue", "Green"]
        for theme in themes:
            theme_action = QAction(f"{theme} Theme", self)
            theme_action.triggered.connect(lambda checked, t=theme: self.apply_theme(t))
            view_menu.addAction(theme_action)

        # Help menu
        help_menu = menubar.addMenu("❓ Help")

        about_action = QAction("ℹ️ About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        docs_action = QAction("📚 Documentation", self)
        docs_action.triggered.connect(self.show_docs)
        help_menu.addAction(docs_action)

    def create_left_panel(self):
        """Create left sidebar with conversation list"""
        panel = QWidget()
        layout = QVBoxLayout()

        title = QLabel("💬 Conversations")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        self.conversation_list = QListWidget()
        self.conversation_list.addItem("Current Session")
        self.conversation_list.currentItemChanged.connect(self.switch_conversation)
        layout.addWidget(self.conversation_list)

        new_btn = QPushButton("➕ New Conversation")
        new_btn.clicked.connect(self.new_chat)
        layout.addWidget(new_btn)

        panel.setLayout(layout)
        return panel

    def create_center_panel(self):
        """Create center chat panel"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 12))
        layout.addWidget(self.chat_display)

        # Input area
        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.setFont(QFont("Arial", 12))
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        self.send_button = QPushButton("📤 Send")
        self.send_button.setFixedWidth(100)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        panel.setLayout(layout)
        return panel

    def create_right_panel(self):
        """Create right sidebar with stats and info"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Tabs
        tabs = QTabWidget()

        # Stats tab
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()

        self.stats_label = QLabel("📊 Statistics\n\nConversations: 0\nKnowledge Items: 0\nAvg Confidence: 0%")
        self.stats_label.setFont(QFont("Arial", 10))
        stats_layout.addWidget(self.stats_label)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.update_stats)
        stats_layout.addWidget(refresh_btn)

        stats_layout.addStretch()
        stats_widget.setLayout(stats_layout)
        tabs.addTab(stats_widget, "📊 Stats")

        # Info tab
        info_widget = QWidget()
        info_layout = QVBoxLayout()

        self.info_label = QLabel("ℹ️ Response Info\n\nNo response yet")
        self.info_label.setFont(QFont("Arial", 10))
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)

        info_layout.addStretch()
        info_widget.setLayout(info_layout)
        tabs.addTab(info_widget, "ℹ️ Info")

        layout.addWidget(tabs)
        panel.setLayout(layout)
        return panel

    def create_system_tray(self):
        """Create system tray icon"""
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))

        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self.show)
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(QApplication.quit)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def init_chatbot(self):
        """Initialize the chatbot system"""
        self.status_bar.showMessage("Initializing AI Chatbot...")

        try:
            chatbot_config = ChatbotConfig(
                model_name="microsoft/DialoGPT-small",
                temperature=0.8
            )

            rag_config = RAGConfig(
                retrieval_top_k=5,
                knowledge_weight=0.7
            )

            self.rag_system = create_rag_system(
                db_path="desktop_knowledge.db",
                chatbot_config=chatbot_config,
                rag_config=rag_config
            )

            self.status_bar.showMessage("✅ AI Chatbot Ready!", 3000)
            self.add_system_message("🤖 AI Chatbot initialized and ready to chat!")

        except Exception as e:
            self.status_bar.showMessage(f"❌ Error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to initialize chatbot:\n{e}")

    def send_message(self):
        """Send a message to the chatbot"""
        user_input = self.input_field.text().strip()

        if not user_input:
            return

        if not self.rag_system:
            QMessageBox.warning(self, "Warning", "Chatbot not initialized!")
            return

        # Display user message
        self.add_user_message(user_input)
        self.input_field.clear()

        # Show progress
        self.progress_bar.show()
        self.send_button.setEnabled(False)
        self.status_bar.showMessage("🤔 Thinking...")

        # Process in background
        self.current_worker = ChatWorker(
            self.rag_system,
            user_input,
            self.conversation_history
        )
        self.current_worker.response_ready.connect(self.handle_response)
        self.current_worker.error_occurred.connect(self.handle_error)
        self.current_worker.start()

        # Update conversation history
        self.conversation_history.append({'role': 'user', 'content': user_input})

    def handle_response(self, result):
        """Handle chatbot response"""
        self.progress_bar.hide()
        self.send_button.setEnabled(True)
        self.status_bar.showMessage("✅ Response received", 2000)

        response = result['response']
        confidence = result['confidence']
        sources = result['retrieved_knowledge_count']

        # Display bot message
        self.add_bot_message(response, confidence, sources)

        # Update conversation history
        self.conversation_history.append({'role': 'bot', 'content': response})

        # Update info panel
        self.update_info_panel(result)

        # Update stats
        self.update_stats()

    def handle_error(self, error_msg):
        """Handle error"""
        self.progress_bar.hide()
        self.send_button.setEnabled(True)
        self.status_bar.showMessage(f"❌ Error: {error_msg}")
        self.add_system_message(f"❌ Error: {error_msg}")

    def add_user_message(self, text):
        """Add user message to chat display"""
        self.chat_display.append(f'<div style="text-align: right; margin: 10px;">'
                                 f'<b style="color: #2196F3;">You:</b><br>'
                                 f'<span style="background: #E3F2FD; padding: 8px; border-radius: 8px; display: inline-block;">{text}</span>'
                                 f'</div>')
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def add_bot_message(self, text, confidence=0.0, sources=0):
        """Add bot message to chat display"""
        conf_percent = int(confidence * 100)
        self.chat_display.append(f'<div style="margin: 10px;">'
                                 f'<b style="color: #4CAF50;">🤖 Bot:</b><br>'
                                 f'<span style="background: #F1F8E9; padding: 8px; border-radius: 8px; display: inline-block;">{text}</span><br>'
                                 f'<small style="color: #666;">Confidence: {conf_percent}% | Sources: {sources}</small>'
                                 f'</div>')
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def add_system_message(self, text):
        """Add system message to chat display"""
        self.chat_display.append(f'<div style="text-align: center; margin: 10px; color: #999;">'
                                 f'<i>{text}</i>'
                                 f'</div>')

    def update_info_panel(self, result):
        """Update info panel with response details"""
        info_text = f"""ℹ️ Response Info

Confidence: {result['confidence']:.0%}
Sources: {result['retrieved_knowledge_count']}
Semantic Similarity: {result.get('semantic_similarity', 0):.2f}
RAG Enhanced: {result.get('rag_enhanced', False)}
Timestamp: {result.get('timestamp', 'N/A')}
"""
        self.info_label.setText(info_text)

    def update_stats(self):
        """Update statistics panel"""
        if not self.rag_system:
            return

        stats = self.rag_system.chatbot.get_stats()

        stats_text = f"""📊 Statistics

Conversations: {stats['total_conversations']}
Knowledge Items: {stats['knowledge_items']}
Episodic Memories: {stats['episodic_memory_size']}
Experience Buffer: {stats['experience_buffer_size']}
Avg Confidence: {stats.get('average_confidence', 0):.0%}
Device: {stats['device']}
"""
        self.stats_label.setText(stats_text)

    def new_chat(self):
        """Start a new chat"""
        self.conversation_history = []
        self.chat_display.clear()
        self.add_system_message("🆕 New conversation started")
        self.status_bar.showMessage("New conversation started", 2000)

    def clear_chat(self):
        """Clear chat display"""
        reply = QMessageBox.question(
            self, "Clear Chat",
            "Are you sure you want to clear the chat?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.chat_display.clear()
            self.status_bar.showMessage("Chat cleared", 2000)

    def save_chat(self):
        """Save conversation to file"""
        from PyQt6.QtWidgets import QFileDialog

        if not self.conversation_history:
            QMessageBox.information(self, "Info", "No conversation to save!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Conversation",
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'conversation': self.conversation_history
                    }, f, indent=2, ensure_ascii=False)
                self.status_bar.showMessage(f"✅ Chat saved to {Path(file_path).name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save chat:\n{e}")

    def load_chat(self):
        """Load conversation from file"""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Conversation",
            "", "JSON Files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.conversation_history = data.get('conversation', [])

                # Rebuild chat display
                self.chat_display.clear()
                for msg in self.conversation_history:
                    if msg['role'] == 'user':
                        self.add_user_message(msg['content'])
                    elif msg['role'] == 'bot':
                        self.add_bot_message(msg['content'])

                self.status_bar.showMessage(f"✅ Chat loaded from {Path(file_path).name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load chat:\n{e}")

    def export_conversation(self):
        """Export conversation to multiple formats"""
        from PyQt6.QtWidgets import QFileDialog

        if not self.conversation_history:
            QMessageBox.information(self, "Info", "No conversation to export!")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Conversation",
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "Text Files (*.txt);;Markdown (*.md);;HTML (*.html);;CSV (*.csv)"
        )

        if file_path:
            try:
                if "Text Files" in selected_filter:
                    self._export_as_text(file_path)
                elif "Markdown" in selected_filter:
                    self._export_as_markdown(file_path)
                elif "HTML" in selected_filter:
                    self._export_as_html(file_path)
                elif "CSV" in selected_filter:
                    self._export_as_csv(file_path)

                self.status_bar.showMessage(f"✅ Exported to {Path(file_path).name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")

    def _export_as_text(self, file_path):
        """Export as plain text"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"AI Chatbot Conversation Export\n")
            f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            for msg in self.conversation_history:
                role = "YOU" if msg['role'] == 'user' else "BOT"
                f.write(f"{role}: {msg['content']}\n\n")

    def _export_as_markdown(self, file_path):
        """Export as markdown"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# AI Chatbot Conversation\n\n")
            f.write(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            for msg in self.conversation_history:
                if msg['role'] == 'user':
                    f.write(f"### 👤 You\n\n{msg['content']}\n\n")
                else:
                    f.write(f"### 🤖 Bot\n\n{msg['content']}\n\n")

    def _export_as_html(self, file_path):
        """Export as HTML"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI Chatbot Conversation</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .message { margin: 15px 0; padding: 12px; border-radius: 8px; }
        .user { background: #E3F2FD; text-align: right; }
        .bot { background: #F1F8E9; }
        .role { font-weight: bold; margin-bottom: 5px; }
        .timestamp { color: #999; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>🧠 AI Chatbot Conversation</h1>
    <p class="timestamp">Exported: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    <hr>
""")

            for msg in self.conversation_history:
                role_class = msg['role']
                role_name = "👤 You" if msg['role'] == 'user' else "🤖 Bot"
                f.write(f'    <div class="message {role_class}">\n')
                f.write(f'        <div class="role">{role_name}</div>\n')
                f.write(f'        <div>{msg["content"]}</div>\n')
                f.write(f'    </div>\n')

            f.write("""
</body>
</html>""")

    def _export_as_csv(self, file_path):
        """Export as CSV"""
        import csv
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Role', 'Message', 'Timestamp'])

            for i, msg in enumerate(self.conversation_history):
                writer.writerow([
                    msg['role'],
                    msg['content'],
                    datetime.now().isoformat()
                ])

    def switch_conversation(self, current, previous):
        """Switch between conversations"""
        if current is None:
            return

        # In a full implementation, this would load the selected conversation
        # For now, we'll just show a placeholder
        self.status_bar.showMessage(f"Switched to: {current.text()}", 2000)

    def show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            self.apply_theme(settings['theme'])
            self.status_bar.showMessage("Settings saved", 2000)

    def apply_theme(self, theme_name):
        """Apply UI theme"""
        if theme_name == "Dark":
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; }
                QTextEdit, QLineEdit { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555; }
                QPushButton { background-color: #4CAF50; color: white; border: none; padding: 8px; border-radius: 4px; }
                QPushButton:hover { background-color: #45a049; }
            """)
        elif theme_name == "Blue":
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #e3f2fd; }
                QPushButton { background-color: #2196F3; color: white; border: none; padding: 8px; border-radius: 4px; }
                QPushButton:hover { background-color: #1976D2; }
            """)
        elif theme_name == "Green":
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #f1f8e9; }
                QPushButton { background-color: #4CAF50; color: white; border: none; padding: 8px; border-radius: 4px; }
                QPushButton:hover { background-color: #45a049; }
            """)
        else:  # Light
            self.setStyleSheet("")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About AI Chatbot",
                         "🧠 AI Chatbot Desktop v3.0\n\n"
                         "State-of-the-art neural network chatbot\n"
                         "with continual learning and RAG.\n\n"
                         "Built with PyQt6 and PyTorch")

    def show_docs(self):
        """Show documentation"""
        QMessageBox.information(self, "Documentation",
                               "📚 Documentation\n\n"
                               "See CHATBOT.md for complete guide\n"
                               "See QUICKSTART.md for quick start\n"
                               "See MOBILE.md for mobile setup")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("AI Chatbot")
    app.setOrganizationName("AI Research")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
