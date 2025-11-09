"""
Extensible Plugin System
Allows dynamic loading of plugins to extend chatbot functionality
Supports hooks, filters, custom commands, and more
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Callable, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod
import json


class PluginInterface(ABC):
    """Base interface that all plugins must implement"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description"""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]):
        """
        Initialize the plugin

        Args:
            config: Plugin configuration
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Cleanup when plugin is unloaded"""
        pass


class MessageProcessorPlugin(PluginInterface):
    """Plugin that processes messages (before/after chatbot)"""

    @abstractmethod
    def process_input(self, user_message: str, context: Dict) -> str:
        """
        Process user input before sending to chatbot

        Args:
            user_message: Original user message
            context: Conversation context

        Returns:
            Processed message
        """
        pass

    @abstractmethod
    def process_output(self, bot_response: str, context: Dict) -> str:
        """
        Process bot response before sending to user

        Args:
            bot_response: Original bot response
            context: Conversation context

        Returns:
            Processed response
        """
        pass


class CommandPlugin(PluginInterface):
    """Plugin that adds custom commands"""

    @abstractmethod
    def get_commands(self) -> Dict[str, Callable]:
        """
        Get custom commands

        Returns:
            Dictionary mapping command names to handler functions
        """
        pass

    @abstractmethod
    def handle_command(self, command: str, args: List[str], context: Dict) -> str:
        """
        Handle a command

        Args:
            command: Command name
            args: Command arguments
            context: Conversation context

        Returns:
            Command response
        """
        pass


class KnowledgeSourcePlugin(PluginInterface):
    """Plugin that provides additional knowledge sources"""

    @abstractmethod
    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the knowledge source

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of knowledge items
        """
        pass


class PluginManager:
    """Manages loading, unloading, and execution of plugins"""

    def __init__(self, plugin_dir: str = "plugins"):
        """
        Initialize plugin manager

        Args:
            plugin_dir: Directory containing plugins
        """
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(exist_ok=True)

        self.plugins: Dict[str, PluginInterface] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        self.filters: Dict[str, List[Callable]] = {}
        self.commands: Dict[str, tuple] = {}  # command -> (plugin, handler)

        # Create sample plugins directory structure
        self._create_example_plugins()

    def _create_example_plugins(self):
        """Create example plugin templates"""
        # Create __init__.py
        init_file = self.plugin_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Plugins directory\n")

        # Create example plugin
        example_file = self.plugin_dir / "example_plugin.py"
        if not example_file.exists():
            example_file.write_text('''"""
Example Plugin
Demonstrates plugin capabilities
"""

from plugin_system import MessageProcessorPlugin

class ExamplePlugin(MessageProcessorPlugin):
    """Example message processor plugin"""

    @property
    def name(self) -> str:
        return "Example Plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Example plugin that adds emojis to messages"

    def initialize(self, config: dict):
        print(f"✅ {self.name} v{self.version} initialized")
        self.enabled = config.get("enabled", True)

    def cleanup(self):
        print(f"🔄 {self.name} cleaning up")

    def process_input(self, user_message: str, context: dict) -> str:
        """Add emoji to user input"""
        if not self.enabled:
            return user_message
        return f"💬 {user_message}"

    def process_output(self, bot_response: str, context: dict) -> str:
        """Add emoji to bot response"""
        if not self.enabled:
            return bot_response
        return f"🤖 {bot_response}"


# Plugin entry point
def load_plugin():
    """Return plugin instance"""
    return ExamplePlugin()
''')

    def load_plugin(self, plugin_name: str, config: Optional[Dict] = None) -> bool:
        """
        Load a plugin

        Args:
            plugin_name: Name of the plugin module
            config: Plugin configuration

        Returns:
            True if loaded successfully
        """
        try:
            # Add plugin directory to path
            if str(self.plugin_dir) not in sys.path:
                sys.path.insert(0, str(self.plugin_dir))

            # Import plugin module
            module = importlib.import_module(plugin_name)

            # Get plugin instance
            if hasattr(module, 'load_plugin'):
                plugin = module.load_plugin()
            else:
                # Try to find plugin class
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, PluginInterface) and obj != PluginInterface:
                        plugin = obj()
                        break
                else:
                    print(f"❌ No plugin class found in {plugin_name}")
                    return False

            # Initialize plugin
            plugin.initialize(config or {})

            # Store plugin
            self.plugins[plugin.name] = plugin

            # Register plugin capabilities
            self._register_plugin(plugin)

            print(f"✅ Loaded plugin: {plugin.name} v{plugin.version}")
            return True

        except Exception as e:
            print(f"❌ Failed to load plugin {plugin_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _register_plugin(self, plugin: PluginInterface):
        """Register plugin capabilities"""
        # Register command handlers
        if isinstance(plugin, CommandPlugin):
            commands = plugin.get_commands()
            for cmd_name, handler in commands.items():
                self.commands[cmd_name] = (plugin, handler)
                print(f"   📝 Registered command: /{cmd_name}")

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin

        Args:
            plugin_name: Name of the plugin

        Returns:
            True if unloaded successfully
        """
        if plugin_name not in self.plugins:
            print(f"⚠️ Plugin {plugin_name} not loaded")
            return False

        plugin = self.plugins[plugin_name]

        # Cleanup
        plugin.cleanup()

        # Remove from registry
        del self.plugins[plugin_name]

        # Remove commands
        self.commands = {k: v for k, v in self.commands.items() if v[0] != plugin}

        print(f"✅ Unloaded plugin: {plugin_name}")
        return True

    def reload_plugin(self, plugin_name: str, config: Optional[Dict] = None) -> bool:
        """
        Reload a plugin

        Args:
            plugin_name: Name of the plugin
            config: Plugin configuration

        Returns:
            True if reloaded successfully
        """
        if plugin_name in self.plugins:
            self.unload_plugin(plugin_name)

        return self.load_plugin(plugin_name, config)

    def process_message_input(self, user_message: str, context: Dict) -> str:
        """
        Process user input through all message processor plugins

        Args:
            user_message: Original message
            context: Conversation context

        Returns:
            Processed message
        """
        processed = user_message

        for plugin in self.plugins.values():
            if isinstance(plugin, MessageProcessorPlugin):
                try:
                    processed = plugin.process_input(processed, context)
                except Exception as e:
                    print(f"⚠️ Plugin {plugin.name} input processing error: {e}")

        return processed

    def process_message_output(self, bot_response: str, context: Dict) -> str:
        """
        Process bot response through all message processor plugins

        Args:
            bot_response: Original response
            context: Conversation context

        Returns:
            Processed response
        """
        processed = bot_response

        for plugin in self.plugins.values():
            if isinstance(plugin, MessageProcessorPlugin):
                try:
                    processed = plugin.process_output(processed, context)
                except Exception as e:
                    print(f"⚠️ Plugin {plugin.name} output processing error: {e}")

        return processed

    def handle_command(self, command: str, args: List[str], context: Dict) -> Optional[str]:
        """
        Handle a custom command

        Args:
            command: Command name (without /)
            args: Command arguments
            context: Conversation context

        Returns:
            Command response or None if not found
        """
        if command not in self.commands:
            return None

        plugin, handler = self.commands[command]

        try:
            if isinstance(plugin, CommandPlugin):
                return plugin.handle_command(command, args, context)
            else:
                return handler(args, context)
        except Exception as e:
            return f"❌ Command error: {e}"

    def query_knowledge_sources(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query all knowledge source plugins

        Args:
            query: Search query
            top_k: Number of results per source

        Returns:
            Combined results from all sources
        """
        results = []

        for plugin in self.plugins.values():
            if isinstance(plugin, KnowledgeSourcePlugin):
                try:
                    plugin_results = plugin.query(query, top_k)
                    results.extend(plugin_results)
                except Exception as e:
                    print(f"⚠️ Plugin {plugin.name} query error: {e}")

        return results

    def list_plugins(self) -> List[Dict[str, str]]:
        """Get list of loaded plugins"""
        return [
            {
                'name': plugin.name,
                'version': plugin.version,
                'description': plugin.description,
                'type': type(plugin).__name__
            }
            for plugin in self.plugins.values()
        ]

    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a plugin"""
        if plugin_name not in self.plugins:
            return None

        plugin = self.plugins[plugin_name]

        return {
            'name': plugin.name,
            'version': plugin.version,
            'description': plugin.description,
            'type': type(plugin).__name__,
            'methods': [m for m in dir(plugin) if not m.startswith('_')]
        }

    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in plugin directory

        Returns:
            List of plugin module names
        """
        plugins = []

        for file in self.plugin_dir.glob("*.py"):
            if file.name.startswith("_"):
                continue

            plugin_name = file.stem
            plugins.append(plugin_name)

        return plugins

    def load_all_plugins(self, config: Optional[Dict[str, Dict]] = None):
        """
        Load all discovered plugins

        Args:
            config: Dictionary mapping plugin names to configs
        """
        available = self.discover_plugins()
        config = config or {}

        print(f"🔍 Discovered {len(available)} plugins")

        for plugin_name in available:
            plugin_config = config.get(plugin_name, {})
            self.load_plugin(plugin_name, plugin_config)

    def save_plugin_config(self, output_path: str = "plugin_config.json"):
        """Save current plugin configuration"""
        config = {
            plugin.name: {
                'enabled': True,
                'version': plugin.version
            }
            for plugin in self.plugins.values()
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Plugin config saved to {output_path}")

    def load_plugin_config(self, config_path: str = "plugin_config.json"):
        """Load plugin configuration from file"""
        if not Path(config_path).exists():
            print(f"⚠️ Config file {config_path} not found")
            return

        with open(config_path, 'r') as f:
            config = json.load(f)

        for plugin_name, plugin_config in config.items():
            if plugin_config.get('enabled', True):
                self.load_plugin(plugin_name, plugin_config)


def demo_plugin_system():
    """Demo the plugin system"""
    print("Plugin System Demo")
    print("=" * 60)

    # Create plugin manager
    manager = PluginManager("plugins")

    # Discover plugins
    print("\n🔍 Discovering plugins...")
    available = manager.discover_plugins()
    print(f"   Found: {available}")

    # Load example plugin
    print("\n📦 Loading example plugin...")
    manager.load_plugin("example_plugin", {"enabled": True})

    # List loaded plugins
    print("\n📋 Loaded plugins:")
    for plugin_info in manager.list_plugins():
        print(f"   - {plugin_info['name']} v{plugin_info['version']}")
        print(f"     {plugin_info['description']}")

    # Test message processing
    print("\n💬 Testing message processing...")
    user_msg = "Hello, chatbot!"
    processed_input = manager.process_message_input(user_msg, {})
    print(f"   Input: {user_msg}")
    print(f"   Processed: {processed_input}")

    bot_response = "Hello! How can I help you?"
    processed_output = manager.process_message_output(bot_response, {})
    print(f"   Response: {bot_response}")
    print(f"   Processed: {processed_output}")

    # Save config
    print("\n💾 Saving plugin configuration...")
    manager.save_plugin_config()

    print("\n✅ Plugin system demo complete!")


if __name__ == '__main__':
    demo_plugin_system()
