"""
Configuration Manager
Central configuration management for all modules
"""
import streamlit as st
import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        """Initialize configuration manager"""
        self.config_file = Path(config_file)
        self.default_config = self._get_default_config()
        self.config = self.load_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "openai": {
                "api_key": st.secrets.get("OPENAI_API_KEY", ""),
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 500
            },
            "assistant": {
                "name": "NARM Therapy Assistant",
                "instructions": "You are a compassionate NARM therapy assistant.",
                "tools": ["file_search", "code_interpreter"],
                "vector_store_id": "vs_68d4f901de948191bf47c56de33994e8"
            },
            "tts": {
                "model": "tts-1",
                "voice": "alloy",
                "speed": 1.0,
                "format": "mp3"
            },
            "stt": {
                "model": "whisper-1",
                "language": None,
                "temperature": 0
            },
            "chat": {
                "streaming": True,
                "history_limit": 50
            },
            "ui": {
                "theme": "light",
                "show_sidebar": True,
                "enable_voice": True,
                "enable_image": True,
                "autoplay_tts": False
            },
            "session": {
                "storage_file": "sessions.json",
                "auto_save": True
            }
        }

    def load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    loaded = json.load(f)
                    # Merge with defaults
                    return self._deep_merge(self.default_config, loaded)
            except Exception as e:
                logger.error(f"Error loading config: {str(e)}")
        return copy.deepcopy(self.default_config)

    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> bool:
        """Set configuration value by dot notation key"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        return self.save_config()

    def update_section(self, section: str, values: Dict) -> bool:
        """Update entire configuration section"""
        if section in self.config:
            self.config[section].update(values)
            return self.save_config()
        return False

    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        self.config = copy.deepcopy(self.default_config)
        return self.save_config()

    def export_config(self) -> str:
        """Export configuration as JSON string"""
        return json.dumps(self.config, indent=2)

    def import_config(self, json_str: str) -> bool:
        """Import configuration from JSON string"""
        try:
            imported = json.loads(json_str)
            self.config = self._deep_merge(self.default_config, imported)
            return self.save_config()
        except Exception as e:
            logger.error(f"Error importing config: {str(e)}")
            return False

    def _deep_merge(self, base: Dict, overlay: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_module_config(self, module: str) -> Dict:
        """Get configuration for specific module"""
        return self.config.get(module, {})

    def validate_api_key(self) -> bool:
        """Validate OpenAI API key"""
        api_key = self.get("openai.api_key")
        return api_key and api_key.startswith("sk-")

# Global config instance
_config_manager = None

def get_config_manager(config_file: str = None) -> ConfigManager:
    """Get or create config manager instance"""
    global _config_manager
    if _config_manager is None or config_file:
        _config_manager = ConfigManager(config_file or "config.json")
    return _config_manager