#!/usr/bin/env python3
"""
Hydra-enabled Runner for Robotic Manipulation Pipeline with VIMA
=================================================================

This script uses Hydra to configure and run the robotic manipulation pipeline
with real-time VIMA environment visualization.

Usage:
    python run_with_hydra.py
    python run_with_hydra.py vima.debug=true
    python run_with_hydra.py vima.display_debug_window=false
    python run_with_hydra.py --config-dir=config --config-name=graph_config

Configuration is loaded from:
- src/config/graph_config.yaml (default)
- Override with --config-dir and --config-name flags
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Hydra imports
try:
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    print("❌ Hydra not available. Install with: pip install hydra-core")
    HYDRA_AVAILABLE = False

# Local imports
from vima_interface import VIMAInterface
from config.config_classes import VIMAConfig, PipelineConfig

def setup_hydra_config_store():
    """Setup Hydra configuration store."""
    cs = ConfigStore.instance()

    # Register configuration schemas
    cs.store(name="vima_config", node=VIMAConfig)
    cs.store(name="pipeline_config", node=PipelineConfig)

    return cs

def run_pipeline_with_config(cfg: DictConfig) -> bool:
    """
    Run the robotic manipulation pipeline with Hydra configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        True if successful, False otherwise
    """
    print("🤖 Starting Robotic Manipulation Pipeline with Hydra Config")
    print("=" * 70)

    try:
        # Extract VIMA configuration
        vima_cfg = cfg.vima if hasattr(cfg, 'vima') else cfg

        print("🔧 VIMA Configuration:")
        print(f"   - Task: {getattr(vima_cfg, 'task_name', 'None')}")
        print(f"   - Modalities: {getattr(vima_cfg, 'modalities', ['rgb'])}")
        print(f"   - Debug: {getattr(vima_cfg, 'debug', False)}")
        print("   - GUI Window: False (DIRECT mode to avoid conflicts)")
        print(f"   - GUI Delay: {getattr(vima_cfg, 'gui_delay', 0.1)}s")

        # Create VIMA interface with Hydra config
        print("\n🏗️  Creating VIMA interface with configuration...")
        print("   Note: Using DIRECT connection mode to avoid GUI conflicts")

        # Create VIMA interface directly with parameters to avoid GUI conflicts
        # Extract parameters from config and force no GUI
        modalities = getattr(vima_cfg, 'modalities', ['rgb'])
        debug = getattr(vima_cfg, 'debug', False)
        action_delay = getattr(vima_cfg, 'action_delay', 0.5)
        task_name = getattr(vima_cfg, 'task_name', 'instruction_following/visual_manipulation')

        # Force DIRECT connection (no GUI) to avoid PyBullet conflicts
        vima_interface = VIMAInterface(
            task_name=task_name,
            modalities=modalities,
            debug=debug,
            display_debug_window=False,  # Force no GUI
            action_delay=action_delay,
            enable_action_logging=True
        )

        if not vima_interface.is_initialized:
            print("❌ Failed to initialize VIMA interface")
            return False

        print("✅ VIMA interface initialized successfully")

        # Get initial observation
        print("\n📡 Getting initial observation from VIMA environment...")
        observation = vima_interface.get_observation()

        print("✅ Initial observation retrieved:")
        print(f"   - RGB image: {observation.rgb_image.size}")
        depth_info = "Not supported" if observation.depth_image is None else observation.depth_image.shape
        print(f"   - Depth: {depth_info}")

        # Get prompt and assets
        print("\n📝 Getting task prompt and assets...")
        prompt, assets = vima_interface.get_prompt_and_assets()
        print(f"✅ Task prompt: {prompt[:100]}...")

        # Demonstrate action execution with visualization
        print("\n🤖 Demonstrating action execution with real-time visualization...")

        # Create a sample action
        import numpy as np
        from vima_interface import VIMAAction

        action = VIMAAction(
            pose0_position=np.array([0.5, 0.0, 0.1]),  # Pick position
            pose0_rotation=np.array([0.0, 0.0, 0.0]),  # Pick rotation
            pose1_position=np.array([0.5, 0.2, 0.1]),  # Place position
            pose1_rotation=np.array([0.0, 0.0, 0.0]),  # Place rotation
            action_metadata={"demo_action": True}
        )

        print("🎯 Executing sample pick-and-place action...")
        print("   Watch the VIMA GUI window for real-time visualization!")

        success, result = vima_interface.execute_action(action)

        if success:
            print(f"✅ Action completed successfully! Reward: {result.get('reward', 'N/A')}")
        else:
            print(f"⚠️  Action completed with issues. Reward: {result.get('reward', 'N/A')}")

        # Execute a sequence of actions
        print("\n🔄 Executing action sequence...")
        actions = [
            VIMAAction(
                pose0_position=np.array([0.5, 0.2, 0.1]),
                pose0_rotation=np.array([0.0, 0.0, 0.0]),
                pose1_position=np.array([0.5, -0.2, 0.1]),
                pose1_rotation=np.array([0.0, 0.0, 0.0]),
            ),
            VIMAAction(
                pose0_position=np.array([0.5, -0.2, 0.1]),
                pose0_rotation=np.array([0.0, 0.0, 0.0]),
                pose1_position=np.array([0.5, 0.0, 0.1]),
                pose1_rotation=np.array([0.0, 0.0, 0.0]),
            )
        ]

        success, results = vima_interface.execute_action_sequence(actions)
        print(f"✅ Action sequence completed. Overall success: {success}")

        # Keep the environment open for a few seconds to see the final state
        print("\n⏳ Keeping environment open for visualization...")
        import time
        time.sleep(3)

        # Cleanup
        print("\n🧹 Cleaning up VIMA interface...")
        vima_interface.close_environment()
        print("✅ Environment closed successfully")

        print("\n🎉 Pipeline demonstration completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_with_hydra_config(config_path: str = "config/graph_config.yaml"):
    """
    Run pipeline using Hydra configuration from file.

    Args:
        config_path: Path to configuration file
    """
    if not HYDRA_AVAILABLE:
        print("❌ Hydra not available. Please install: pip install hydra-core")
        return False

    try:
        # Load configuration from file
        print(f"📄 Loading configuration from: {config_path}")

        # Use OmegaConf to load the config
        cfg = OmegaConf.load(config_path)

        # Run pipeline with loaded config
        return run_pipeline_with_config(cfg)

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False

def main():
    """Main entry point."""
    if not HYDRA_AVAILABLE:
        print("❌ Hydra not available.")
        print("💡 Install Hydra: pip install hydra-core")
        print("💡 Or run basic demo: python run_basic_demo.py")
        return

    print("🚀 Hydra-enabled Robotic Manipulation Pipeline")
    print("=" * 60)

    # Try to load from default config
    config_file = src_path / "config" / "graph_config.yaml"

    if config_file.exists():
        print(f"📁 Using default config: {config_file}")
        success = run_with_hydra_config(str(config_file))
    else:
        print("⚠️  Default config not found, using built-in configuration")
        # Create a default config
        default_cfg = OmegaConf.create({
            "vima": {
                "task_name": None,
                "modalities": ["rgb", "depth"],
                "debug": False,
                "display_debug_window": True,
                "hide_arm_rgb": False,
                "gui_delay": 0.1,
                "action_delay": 0.5,
                "enable_action_logging": True,
                "enable_observation_logging": False
            }
        })
        success = run_pipeline_with_config(default_cfg)

    if success:
        print("\n🎉 Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
