#!/usr/bin/env python3
"""
Unified Configuration Manager
============================

This module provides a single entry point for all configuration loading,
with consistent fallback mechanisms and colorful output for better readability.
"""

from typing import Dict, Any
from pathlib import Path
from utils.printer import Printer
from .yaml_handler import YAMLHandler
from .default_configs import create_default_config_files, get_default_configs

# Try to import OmegaConf, fallback to basic config if not available
try:
    from omegaconf import OmegaConf, DictConfig
    OMEGACONF_AVAILABLE = True
    Printer.debug("OmegaConf available - using advanced configuration features")
except ImportError:
    OMEGACONF_AVAILABLE = False
    Printer.warning("OmegaConf not available - using basic configuration fallback")

from .config_classes import (
    PipelineConfig, VIMAConfig, GrounderConfig, SegmentorConfig
)

class ConfigManager:
    """
    Unified configuration manager that handles all config loading with proper fallbacks.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._config_cache: Dict[str, Any] = {}
        
        Printer.debug(f"Initializing ConfigManager with directory: {self.config_dir}")
        
        # Ensure config directory exists and create default configs if needed
        if not self.config_dir.exists():
            Printer.warning(f"Config directory not found: {self.config_dir}")
            Printer.info("Creating default configuration files...")
            create_default_config_files(self.config_dir)
        else:
            # Check if we need to create any missing config files
            self._ensure_config_files_exist()
    
    def _ensure_config_files_exist(self):
        """Ensure all required config files exist, create defaults if missing"""
        required_files = [
            self.config_dir / "graph_config.yaml",
            self.config_dir / "vima" / "vima_config.yaml",
            self.config_dir / "grounder_config.yaml",
            self.config_dir / "segmentor_config.yaml",
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            Printer.warning(f"Missing config files: {[str(f) for f in missing_files]}")
            Printer.info("Creating missing default configuration files...")
            create_default_config_files(self.config_dir)
    
    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load all configurations and return as a structured dictionary.
        
        Returns:
            Dictionary containing all loaded configurations
        """
        Printer.info("🔄 Loading all configurations...")
        
        try:
            configs = {
                "pipeline": self.load_pipeline_config(),
                "vima": self.load_vima_config(),
                "grounder": self.load_grounder_config(),
                "segmentor": self.load_segmentor_config(),
            }
            
            Printer.success("All configurations loaded successfully!")
            return configs
            
        except Exception as e:
            Printer.error(f"Failed to load configurations: {e}")
            Printer.warning("Using default configurations")
            return get_default_configs()
    
    def load_pipeline_config(self) -> PipelineConfig:
        """Load main pipeline configuration"""
        config_path = self.config_dir / "graph_config.yaml"
        
        if not config_path.exists():
            Printer.warning(f"Pipeline config not found: {config_path}")
            Printer.info("Using default pipeline configuration")
            return PipelineConfig()
        
        try:
            if OMEGACONF_AVAILABLE:
                conf = OmegaConf.load(str(config_path))
                config = PipelineConfig(**conf)
            else:
                # Use YAML handler for basic YAML loading
                config_dict = YAMLHandler.load_yaml(config_path)
                config = PipelineConfig(**config_dict)
            
            Printer.success(f"Pipeline config loaded from: {config_path}")
            Printer.config("Pipeline", config.pipeline_name)
            Printer.config("Logging", config.enable_logging)
            Printer.config("Timeout", config.node_timeout)
            
            return config
            
        except Exception as e:
            Printer.error(f"Error loading pipeline config: {e}")
            Printer.info("Using default pipeline configuration")
            return PipelineConfig()
    
    def load_vima_config(self) -> VIMAConfig:
        """Load VIMA configuration"""
        config_path = self.config_dir / "vima" / "vima_config.yaml"
        
        if not config_path.exists():
            Printer.warning(f"VIMA config not found: {config_path}")
            Printer.info("Using default VIMA configuration")
            return VIMAConfig()
        
        try:
            if OMEGACONF_AVAILABLE:
                conf = OmegaConf.load(str(config_path))
                vima_conf = conf.get("vima", {})
                # Extract the env section if it exists, otherwise use the vima section directly
                env_conf = vima_conf.get("env", vima_conf)
                # Filter to only include fields that VIMAConfig expects
                filtered_conf = self._filter_vima_config(env_conf)
                config = VIMAConfig(**filtered_conf)
            else:
                # Use YAML handler for basic YAML loading
                config_dict = YAMLHandler.load_yaml(config_path)
                vima_conf = config_dict.get("vima", {})
                # Extract the env section if it exists, otherwise use the vima section directly
                env_conf = vima_conf.get("env", vima_conf)
                # Filter to only include fields that VIMAConfig expects
                filtered_conf = self._filter_vima_config(env_conf)
                config = VIMAConfig(**filtered_conf)
            
            Printer.success(f"VIMA config loaded from: {config_path}")
            Printer.config("Task", config.task_name)
            Printer.config("Modalities", str(config.modalities))
            Printer.config("Debug", config.debug)
            
            return config
            
        except Exception as e:
            Printer.error(f"Error loading VIMA config: {e}")
            Printer.info("Using default VIMA configuration")
            return VIMAConfig()
    
    def load_grounder_config(self) -> GrounderConfig:
        """Load grounder configuration"""
        config_path = self.config_dir / "grounder_config.yaml"
        
        if not config_path.exists():
            Printer.warning(f"Grounder config not found: {config_path}")
            Printer.info("Using default grounder configuration")
            return GrounderConfig()
        
        try:
            if OMEGACONF_AVAILABLE:
                conf = OmegaConf.load(str(config_path))
                grounder_conf = conf.get("grounder", {})
                config = GrounderConfig(**grounder_conf)
            else:
                # Use YAML handler for basic YAML loading
                config_dict = YAMLHandler.load_yaml(config_path)
                grounder_conf = config_dict.get("grounder", {})
                config = GrounderConfig(**grounder_conf)
            
            Printer.success(f"Grounder config loaded from: {config_path}")
            Printer.config("Mode", config.grounding_mode)
            Printer.config("Device", config.device)
            Printer.config("Auto fallback", config.auto_fallback_to_simple)
            
            return config
            
        except Exception as e:
            Printer.error(f"Error loading grounder config: {e}")
            Printer.info("Using default grounder configuration")
            return GrounderConfig()
    
    def load_segmentor_config(self) -> SegmentorConfig:
        """Load segmentor configuration"""
        config_path = self.config_dir / "segmentor_config.yaml"
        
        if not config_path.exists():
            Printer.warning(f"Segmentor config not found: {config_path}")
            Printer.info("Using default segmentor configuration")
            return SegmentorConfig()
        
        try:
            if OMEGACONF_AVAILABLE:
                conf = OmegaConf.load(str(config_path))
                segmentor_conf = conf.get("segmentor", {})
                config = SegmentorConfig(**segmentor_conf)
            else:
                # Use YAML handler for basic YAML loading
                config_dict = YAMLHandler.load_yaml(config_path)
                segmentor_conf = config_dict.get("segmentor", {})
                config = SegmentorConfig(**segmentor_conf)
            
            Printer.success(f"Segmentor config loaded from: {config_path}")
            Printer.config("Backend", config.backend)
            Printer.config("Device", config.device)
            Printer.config("Points per box", config.points_per_box)
            
            return config
            
        except Exception as e:
            Printer.error(f"Error loading segmentor config: {e}")
            Printer.info("Using default segmentor configuration")
            return SegmentorConfig()
    
    def validate_config(self, config: Any) -> bool:
        """Validate a configuration object"""
        try:
            if OMEGACONF_AVAILABLE and isinstance(config, DictConfig):
                OmegaConf.validate(config)
            
            # Additional custom validation can be added here
            return True
            
        except Exception as e:
            Printer.error(f"Configuration validation failed: {e}")
            return False
    
    def _filter_vima_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Filter VIMA config to only include fields that VIMAConfig expects"""
        from .config_classes import VIMAConfig
        import inspect
        
        # Get the field names from VIMAConfig
        vima_fields = set()
        for field_name, field_info in VIMAConfig.__dataclass_fields__.items():
            vima_fields.add(field_name)
        
        # Filter the config to only include expected fields
        filtered_config = {}
        for key, value in config_dict.items():
            if key in vima_fields:
                filtered_config[key] = value
            else:
                Printer.debug(f"Skipping unknown VIMA config field: {key}")
        
        return filtered_config

    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configurations using OmegaConf if available"""
        if not OMEGACONF_AVAILABLE:
            Printer.warning("OmegaConf not available - using basic dict merge")
            # Basic dict merge
            merged = base_config.copy()
            merged.update(override_config)
            return merged
        
        try:
            base = OmegaConf.create(base_config)
            override = OmegaConf.create(override_config)
            merged = OmegaConf.merge(base, override)
            return OmegaConf.to_container(merged, resolve=True)
            
        except Exception as e:
            Printer.error(f"Error merging configs: {e}")
            return base_config

# Global instance
config_manager = ConfigManager()

# Convenience functions for backward compatibility
def load_all_configs() -> Dict[str, Any]:
    """Load all configurations"""
    return config_manager.load_all_configs()

def load_pipeline_config() -> PipelineConfig:
    """Load pipeline configuration"""
    return config_manager.load_pipeline_config()

def load_vima_config() -> VIMAConfig:
    """Load VIMA configuration"""
    return config_manager.load_vima_config()

def load_grounder_config() -> GrounderConfig:
    """Load grounder configuration"""
    return config_manager.load_grounder_config()

def load_segmentor_config() -> SegmentorConfig:
    """Load segmentor configuration"""
    return config_manager.load_segmentor_config()