#!/usr/bin/env python3
"""
Configuration Loader using OmegaConf
====================================

This module provides structured configuration loading with OmegaConf,
offering type safety, validation, and interpolation capabilities.
"""

import os
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig
from config.config_classes import (
    PipelineConfig, VIMAConfig, GrounderConfig,
    SegmentorConfig
)

class ConfigurationLoader:
    """Loads and validates configuration using OmegaConf"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self._config_cache: Dict[str, Any] = {}

    def load_grounder_config(self) -> GrounderConfig:
        """Load grounder configuration with validation"""
        config_path = os.path.join(self.config_dir, "grounder_config.yaml")

        if not os.path.exists(config_path):
            print(f"⚠️  Grounder config not found: {config_path}, using defaults")
            return GrounderConfig()

        try:
            # Load with OmegaConf
            conf = OmegaConf.load(config_path)

            # Extract grounder section
            grounder_conf = conf.get("grounder", {})

            # Convert to GrounderConfig with validation
            grounder_config = GrounderConfig(**grounder_conf)

            print(f"✅ Grounder config loaded from: {config_path}")
            print(f"   - Mode: {grounder_config.grounding_mode}")
            print(f"   - Device: {grounder_config.device}")
            print(f"   - Auto fallback: {grounder_config.auto_fallback_to_simple}")

            return grounder_config

        except Exception as e:
            print(f"❌ Error loading grounder config: {e}")
            print("   Using default configuration")
            return GrounderConfig()

    def load_segmentor_config(self) -> SegmentorConfig:
        """Load segmentor configuration with validation"""
        config_path = os.path.join(self.config_dir, "segmentor_config.yaml")

        if not os.path.exists(config_path):
            print(f"⚠️  Segmentor config not found: {config_path}, using defaults")
            return SegmentorConfig()

        try:
            # Load with OmegaConf
            conf = OmegaConf.load(config_path)

            # Extract segmentor section
            segmentor_conf = conf.get("segmentor", {})

            # Convert to SegmentorConfig with validation
            segmentor_config = SegmentorConfig(**segmentor_conf)

            print(f"✅ Segmentor config loaded from: {config_path}")
            print(f"   - Backend: {segmentor_config.backend}")
            print(f"   - Device: {segmentor_config.device}")
            print(f"   - Points per box: {segmentor_config.points_per_box}")

            return segmentor_config

        except Exception as e:
            print(f"❌ Error loading segmentor config: {e}")
            print("   Using default configuration")
            return SegmentorConfig()

    def load_vima_config(self) -> VIMAConfig:
        """Load VIMA configuration with validation"""
        config_path = os.path.join(self.config_dir, "vima", "vima_config.yaml")

        if not os.path.exists(config_path):
            print(f"⚠️  VIMA config not found: {config_path}, using defaults")
            return VIMAConfig()

        try:
            # Load with OmegaConf
            conf = OmegaConf.load(config_path)

            # Extract VIMA section
            vima_conf = conf.get("vima", {})

            # Convert to VIMAConfig with validation
            vima_config = VIMAConfig(**vima_conf)

            print(f"✅ VIMA config loaded from: {config_path}")
            print(f"   - Task: {vima_config.task_name}")
            print(f"   - Modalities: {vima_config.modalities}")
            print(f"   - Debug: {vima_config.debug}")

            return vima_config

        except Exception as e:
            print(f"❌ Error loading VIMA config: {e}")
            print("   Using default configuration")
            return VIMAConfig()

    def load_main_config(self) -> PipelineConfig:
        """Load main pipeline configuration with validation"""
        config_path = os.path.join(self.config_dir, "graph_config.yaml")

        if not os.path.exists(config_path):
            print(f"⚠️  Main config not found: {config_path}, using defaults")
            return PipelineConfig()

        try:
            # Load with OmegaConf
            conf = OmegaConf.load(config_path)

            # Convert to PipelineConfig with validation
            pipeline_config = PipelineConfig(**conf)

            print(f"✅ Main config loaded from: {config_path}")
            print(f"   - Pipeline: {pipeline_config.pipeline_name}")
            print(f"   - Logging: {pipeline_config.enable_logging}")
            print(f"   - Timeout: {pipeline_config.node_timeout}")

            return pipeline_config

        except Exception as e:
            print(f"❌ Error loading main config: {e}")
            print("   Using default configuration")
            return PipelineConfig()

    def load_all_configs(self) -> Dict[str, Any]:
        """Load all configurations and return as a structured dict"""
        try:
            configs = {
                "pipeline": self.load_main_config(),
                "grounder": self.load_grounder_config(),
                "segmentor": self.load_segmentor_config(),
                "vima": self.load_vima_config()
            }

            print("\n🎉 All configurations loaded successfully!")
            return configs

        except Exception as e:
            print(f"❌ Error loading configurations: {e}")
            # Return defaults
            return {
                "pipeline": PipelineConfig(),
                "grounder": GrounderConfig(),
                "segmentor": SegmentorConfig(),
                "vima": VIMAConfig()
            }

    def validate_config(self, config: Any) -> bool:
        """Validate a configuration object"""
        try:
            # OmegaConf validation
            if isinstance(config, DictConfig):
                OmegaConf.validate(config)

            # Custom validation logic can be added here
            return True

        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False

    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configurations using OmegaConf"""
        try:
            base = OmegaConf.create(base_config)
            override = OmegaConf.create(override_config)

            # Merge with override taking precedence
            merged = OmegaConf.merge(base, override)

            return OmegaConf.to_container(merged, resolve=True)

        except Exception as e:
            print(f"❌ Error merging configs: {e}")
            return base_config

# Global instance
config_loader = ConfigurationLoader()

# Convenience functions
def load_grounder_config() -> GrounderConfig:
    """Load grounder configuration"""
    return config_loader.load_grounder_config()

def load_segmentor_config() -> SegmentorConfig:
    """Load segmentor configuration"""
    return config_loader.load_segmentor_config()

def load_vima_config() -> VIMAConfig:
    """Load VIMA configuration"""
    return config_loader.load_vima_config()

def load_all_configs() -> Dict[str, Any]:
    """Load all configurations"""
    return config_loader.load_all_configs()
