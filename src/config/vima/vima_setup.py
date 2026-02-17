#!/usr/bin/env python3
"""
VIMA Setup and Configuration Module

This module handles all VIMA-related configuration, initialization, and setup.
It provides a clean interface for creating and configuring VIMA environments.
"""

import sys
import os
from typing import Dict, Any, Optional

# Try to import yaml, fallback to None if not available
try:
    import yaml
except ImportError:
    yaml = None
    print("⚠️  PyYAML not available, will use default configuration")

# Try to import omegaconf, fallback to None if not available
try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = None
    OmegaConf = None
    print("⚠️  OmegaConf not available, will use basic configuration")

# Add VIMABench to path
sys.path.append('../VIMABench')

try:
    from vima_interface import VIMAInterface
except ImportError:
    VIMAInterface = None

class VIMASetup:
    """Handles VIMA environment setup and configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize VIMA setup.

        Args:
            config_path: Path to VIMA configuration file
        """
        if config_path is None:
            # Get the directory where this module is located
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.join(module_dir, "vima_config.yaml")
        else:
            self.config_path = config_path

        self.config = self._load_config()
        print(f"🔍 Config loaded: {self.config}")

        # Ensure config is never None
        if self.config is None:
            print("⚠️  Config loading failed, using default configuration")
            self.config = self._get_default_config()
            print(f"🔍 Default config set: {type(self.config)}")

        self.vima_interface = None

    def _load_config(self) -> Dict[str, Any]:
        """Load VIMA configuration from YAML file."""
        print(f"🔍 Loading VIMA config from: {self.config_path}")
        print(f"🔍 YAML module available: {yaml is not None}")

        try:
            # Check if yaml module is available
            if yaml is None:
                print("⚠️  PyYAML not available, using default configuration")
                return self._get_default_config()

            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    if config is None:
                        print("⚠️  YAML file is empty, using default configuration")
                        return self._get_default_config()
                    print(f"✅ VIMA config loaded from: {self.config_path}")
                    return config
                except Exception as yaml_error:
                    print(f"⚠️  Error parsing YAML file: {yaml_error}")
                    print("   Using default configuration")
                    return self._get_default_config()
            else:
                print(f"⚠️  VIMA config file not found: {self.config_path}")
                print("   Creating config file with default settings...")

                # Try to create the config file with default settings
                try:
                    os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                    with open(self.config_path, 'w') as f:
                        yaml.dump(self._get_default_config(), f, default_flow_style=False, indent=2)
                    print(f"✅ Created default VIMA config at: {self.config_path}")
                    return self._get_default_config()
                except Exception as create_error:
                    print(f"⚠️  Could not create config file: {create_error}")
                    print("   Using in-memory default configuration")
                    return self._get_default_config()

        except Exception as e:
            print(f"❌ Error loading VIMA config: {e}")
            print("   Using default configuration")
            result = self._get_default_config()
            print(f"🔍 Returning default config: {type(result)}")
            return result

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default VIMA configuration."""
        return {
            "vima": {
                "env": {
                    "task_name": "instruction_following/visual_manipulation",
                    "display_debug_window": False,
                    "render_mode": "rgb_array",
                    "camera_config": {
                        "width": 256,
                        "height": 128,
                        "fov": 60
                    }
                },
                "interface": {
                    "observation_modalities": ["rgb"],
                    "include_prompt_assets": True,
                    "include_metadata": True,
                    "action_space": "discrete",
                    "max_action_sequence": 10,
                    "max_retries": 3,
                    "fallback_on_error": True
                }
            }
        }

    def create_vima_interface(self, **kwargs) -> VIMAInterface:
        """
        Create and configure a VIMA interface.

        Args:
            **kwargs: Override configuration parameters

        Returns:
            Configured VIMAInterface instance
        """
        if VIMAInterface is None:
            raise ImportError("VIMAInterface not available. Check VIMABench installation.")

        # Ensure config is available
        if self.config is None:
            print("⚠️  Configuration is None, loading default configuration")
            self.config = self._get_default_config()

        # Get configuration
        config = self.config.get("vima", {})
        env_config = config.get("env", {})

        # Override with kwargs
        env_config.update({k: v for k, v in kwargs.items() if k in env_config})

        # Create VIMA interface with proper camera and rendering settings
        try:
            # Ensure robot arm is hidden for clean observations
            vima_kwargs = {
                "task_name": env_config.get("task_name", "instruction_following/visual_manipulation"),
                "display_debug_window": env_config.get("display_debug_window", False),
                "hide_arm_rgb": env_config.get("hide_arm_rgb", False),  # Always hide robot arm for clean images
                "save_observation_images": True,  # Save images when getting observations
                "observation_image_dir": "observations",  # Directory for saved images
                "camera_config": env_config.get("camera_config", {
                    "width": 1280,
                    "height": 720,
                    "fov": 60
                })
            }

            # Override with any additional kwargs
            vima_kwargs.update({k: v for k, v in kwargs.items() if k not in vima_kwargs})

            self.vima_interface = VIMAInterface(**vima_kwargs)

            print("✅ VIMA interface created successfully")
            print(f"   Task: {env_config.get('task_name', 'instruction_following/visual_manipulation')}")
            print(f"   Display: {'Enabled' if env_config.get('display_debug_window', False) else 'Disabled'}")

            return self.vima_interface

        except Exception as e:
            print(f"❌ Failed to create VIMA interface: {e}")
            raise

    def setup_task(self, task_name: Optional[str] = None) -> bool:
        """
        Set up a specific task in the VIMA environment.

        Args:
            task_name: Task name to set up (uses config default if None)

        Returns:
            True if successful, False otherwise
        """
        if not self.vima_interface:
            print("❌ VIMA interface not created. Call create_vima_interface() first.")
            return False

        try:
            task = task_name or self.config["vima"]["env"]["task_name"]
            self.vima_interface.setup_task(task)
            print(f"✅ Task setup successful: {task}")
            return True
        except Exception as e:
            print(f"❌ Task setup failed: {e}")
            return False

    def get_observation(self) -> Any:
        """
        Get current observation from VIMA environment.

        Returns:
            VIMA observation object
        """
        if not self.vima_interface:
            print("❌ VIMA interface not created. Call create_vima_interface() first.")
            return None

        try:
            observation = self.vima_interface.get_observation()
            print("✅ Observation retrieved successfully")
            return observation
        except Exception as e:
            print(f"❌ Failed to get observation: {e}")
            return None

    def get_prompt_and_assets(self) -> tuple[str, dict]:
        """
        Get task prompt and assets from VIMA environment.

        Returns:
            Tuple of (prompt_string, assets_dict)
        """
        if not self.vima_interface:
            print("❌ VIMA interface not created. Call create_vima_interface() first.")
            return "Default task", {}

        try:
            prompt, assets = self.vima_interface.get_prompt_and_assets()
            print("✅ Prompt and assets retrieved successfully")
            return prompt, assets
        except Exception as e:
            print(f"❌ Failed to get prompt and assets: {e}")
            return "Default task", {}

    def execute_action(self, action: Any) -> bool:
        """
        Execute an action in the VIMA environment.

        Args:
            action: Action to execute

        Returns:
            True if successful, False otherwise
        """
        if not self.vima_interface:
            print("❌ VIMA interface not created. Call create_vima_interface() first.")
            return False

        try:
            self.vima_interface.execute_action(action)
            print("✅ Action executed successfully")
            return True
        except Exception as e:
            print(f"❌ Action execution failed: {e}")
            return False

    def reset_environment(self) -> bool:
        """
        Reset the VIMA environment.

        Returns:
            True if successful, False otherwise
        """
        if not self.vima_interface:
            print("❌ VIMA interface not created. Call create_vima_interface() first.")
            return False

        try:
            self.vima_interface.reset()
            print("✅ Environment reset successfully")
            return True
        except Exception as e:
            print(f"❌ Environment reset failed: {e}")
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current VIMA configuration."""
        return {
            "config_file": self.config_path,
            "task_name": self.config["vima"]["env"].get("task_name"),
            "display_enabled": self.config["vima"]["env"].get("display_debug_window"),
            "observation_modalities": self.config["vima"]["interface"].get("observation_modalities"),
            "interface_created": self.vima_interface is not None
        }

def create_vima_interface_from_config(config_path: Optional[str] = None, **kwargs) -> VIMAInterface:
    """
    Convenience function to create VIMA interface from configuration.

    Args:
        config_path: Path to VIMA configuration file (if None, uses default in vima config dir)
        **kwargs: Override configuration parameters

    Returns:
        Configured VIMAInterface instance
    """
    setup = VIMASetup(config_path)
    return setup.create_vima_interface(**kwargs)

# Example usage
if __name__ == "__main__":
    print("🔧 VIMA Setup Module")
    print("=" * 30)

    # Create VIMA setup
    setup = VIMASetup()

    # Show configuration summary
    summary = setup.get_config_summary()
    print("Configuration Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Create VIMA interface
    try:
        vima_interface = setup.create_vima_interface()
        print("\n✅ VIMA interface created successfully!")
    except Exception as e:
        print(f"\n❌ Failed to create VIMA interface: {e}")
