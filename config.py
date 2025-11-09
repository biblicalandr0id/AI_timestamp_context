"""
Configuration Management System
Centralized configuration with environment variable support
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ConsciousnessConfig:
    """Configuration for consciousness engine"""
    working_memory_size: int = 5
    short_term_memory_size: int = 20
    enable_sentiment_analysis: bool = True
    enable_pattern_detection: bool = True
    consciousness_threshold: float = 0.3


@dataclass
class SemanticConfig:
    """Configuration for semantic analysis"""
    embedding_dim: int = 384
    use_sentence_transformers: bool = True
    model_name: str = "all-MiniLM-L6-v2"
    cache_embeddings: bool = True
    max_topics: int = 10


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent system"""
    max_agents: int = 10
    default_agent_panel: bool = True
    enable_agent_learning: bool = False
    interaction_log_size: int = 1000


@dataclass
class EmotionConfig:
    """Configuration for emotion tracking"""
    history_window_size: int = 20
    intensity_threshold: float = 0.1
    enable_vad_analysis: bool = True  # Valence-Arousal-Dominance
    emotion_shift_threshold: float = 0.5


@dataclass
class APIConfig:
    """Configuration for API server"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    enable_cors: bool = True
    max_request_size_mb: int = 16
    rate_limit_per_minute: int = 60


@dataclass
class DatabaseConfig:
    """Configuration for database"""
    enabled: bool = False
    db_type: str = "sqlite"  # sqlite, postgres, mongodb
    connection_string: str = "sqlite:///ai_context.db"
    pool_size: int = 5
    auto_commit: bool = True


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_size_mb: int = 100
    enable_async_processing: bool = False
    worker_threads: int = 4


@dataclass
class SystemConfig:
    """Main system configuration"""
    # Component configs
    consciousness: ConsciousnessConfig = field(default_factory=ConsciousnessConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    multi_agent: MultiAgentConfig = field(default_factory=MultiAgentConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # General settings
    log_level: str = "INFO"
    data_directory: str = "./data"
    export_directory: str = "./exports"
    enable_telemetry: bool = False


class ConfigManager:
    """Manages system configuration"""

    def __init__(self, config_file: Optional[str] = None):
        self.config = SystemConfig()
        self.config_file = config_file

        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        else:
            self.load_from_env()

    def load_from_file(self, filepath: str):
        """Load configuration from file (JSON or YAML)"""
        file_ext = Path(filepath).suffix.lower()

        try:
            with open(filepath, 'r') as f:
                if file_ext == '.json':
                    data = json.load(f)
                elif file_ext in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {file_ext}")

            self._update_config_from_dict(data)
            print(f"✓ Configuration loaded from: {filepath}")

        except Exception as e:
            print(f"✗ Error loading config: {e}")

    def load_from_env(self):
        """Load configuration from environment variables"""
        # Consciousness settings
        if val := os.getenv('CONSCIOUSNESS_WORKING_MEMORY_SIZE'):
            self.config.consciousness.working_memory_size = int(val)
        if val := os.getenv('CONSCIOUSNESS_ENABLE_SENTIMENT'):
            self.config.consciousness.enable_sentiment_analysis = val.lower() == 'true'

        # Semantic settings
        if val := os.getenv('SEMANTIC_EMBEDDING_DIM'):
            self.config.semantic.embedding_dim = int(val)
        if val := os.getenv('SEMANTIC_MODEL_NAME'):
            self.config.semantic.model_name = val

        # API settings
        if val := os.getenv('API_HOST'):
            self.config.api.host = val
        if val := os.getenv('API_PORT'):
            self.config.api.port = int(val)
        if val := os.getenv('API_DEBUG'):
            self.config.api.debug = val.lower() == 'true'

        # Database settings
        if val := os.getenv('DATABASE_ENABLED'):
            self.config.database.enabled = val.lower() == 'true'
        if val := os.getenv('DATABASE_CONNECTION_STRING'):
            self.config.database.connection_string = val

        # General settings
        if val := os.getenv('LOG_LEVEL'):
            self.config.log_level = val
        if val := os.getenv('DATA_DIRECTORY'):
            self.config.data_directory = val

    def _update_config_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in data.items():
            if hasattr(self.config, section):
                section_config = getattr(self.config, section)
                if isinstance(values, dict):
                    for key, value in values.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)

    def save_to_file(self, filepath: str, format: str = 'json'):
        """Save configuration to file"""
        config_dict = asdict(self.config)

        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

            with open(filepath, 'w') as f:
                if format == 'json':
                    json.dump(config_dict, f, indent=2)
                elif format in ['yaml', 'yml']:
                    yaml.dump(config_dict, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            print(f"✓ Configuration saved to: {filepath}")

        except Exception as e:
            print(f"✗ Error saving config: {e}")

    def get(self, section: str, key: Optional[str] = None) -> Any:
        """Get configuration value"""
        if not hasattr(self.config, section):
            return None

        section_config = getattr(self.config, section)

        if key is None:
            return section_config

        return getattr(section_config, key, None)

    def set(self, section: str, key: str, value: Any):
        """Set configuration value"""
        if hasattr(self.config, section):
            section_config = getattr(self.config, section)
            if hasattr(section_config, key):
                setattr(section_config, key, value)

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = SystemConfig()

    def print_config(self):
        """Print current configuration"""
        config_dict = asdict(self.config)

        print("\n" + "=" * 60)
        print("CURRENT CONFIGURATION")
        print("=" * 60)

        def print_section(name, data, indent=0):
            print("  " * indent + f"{name}:")
            for key, value in data.items():
                if isinstance(value, dict):
                    print_section(key, value, indent + 1)
                else:
                    print("  " * (indent + 1) + f"{key}: {value}")

        for section, data in config_dict.items():
            if isinstance(data, dict):
                print_section(section, data)
            else:
                print(f"{section}: {data}")

        print("=" * 60)

    def validate_config(self) -> tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []

        # Validate consciousness config
        if self.config.consciousness.working_memory_size < 1:
            errors.append("working_memory_size must be >= 1")

        if not 0 <= self.config.consciousness.consciousness_threshold <= 1:
            errors.append("consciousness_threshold must be between 0 and 1")

        # Validate semantic config
        if self.config.semantic.embedding_dim < 1:
            errors.append("embedding_dim must be >= 1")

        if self.config.semantic.max_topics < 1:
            errors.append("max_topics must be >= 1")

        # Validate emotion config
        if self.config.emotion.history_window_size < 1:
            errors.append("history_window_size must be >= 1")

        if not 0 <= self.config.emotion.emotion_shift_threshold <= 1:
            errors.append("emotion_shift_threshold must be between 0 and 1")

        # Validate API config
        if not 1024 <= self.config.api.port <= 65535:
            errors.append("port must be between 1024 and 65535")

        if self.config.api.max_request_size_mb < 1:
            errors.append("max_request_size_mb must be >= 1")

        # Validate performance config
        if self.config.performance.worker_threads < 1:
            errors.append("worker_threads must be >= 1")

        return len(errors) == 0, errors


# Global config instance
_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration instance"""
    global _global_config

    if _global_config is None:
        # Look for config file in common locations
        config_locations = [
            'config.json',
            'config.yaml',
            'config.yml',
            os.path.expanduser('~/.ai_context/config.json'),
            '/etc/ai_context/config.json'
        ]

        config_file = None
        for location in config_locations:
            if os.path.exists(location):
                config_file = location
                break

        _global_config = ConfigManager(config_file)

    return _global_config


def initialize_config(config_file: Optional[str] = None):
    """Initialize global configuration"""
    global _global_config
    _global_config = ConfigManager(config_file)
    return _global_config


if __name__ == '__main__':
    # Example usage and testing
    import argparse

    parser = argparse.ArgumentParser(description='Configuration Management')
    parser.add_argument('--print', action='store_true', help='Print current config')
    parser.add_argument('--validate', action='store_true', help='Validate config')
    parser.add_argument('--save', type=str, help='Save config to file')
    parser.add_argument('--load', type=str, help='Load config from file')
    parser.add_argument('--format', choices=['json', 'yaml'], default='json', help='Config format')

    args = parser.parse_args()

    config_manager = get_config()

    if args.load:
        config_manager.load_from_file(args.load)

    if args.print:
        config_manager.print_config()

    if args.validate:
        valid, errors = config_manager.validate_config()
        if valid:
            print("✓ Configuration is valid")
        else:
            print("✗ Configuration has errors:")
            for error in errors:
                print(f"  - {error}")

    if args.save:
        config_manager.save_to_file(args.save, args.format)
