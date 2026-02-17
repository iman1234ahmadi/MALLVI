#!/usr/bin/env python3
"""
Main entry point for the Robotic Manipulation Pipeline with VIMA Integration.
This script integrates VIMA interface with Hydra configuration and runs the
multi-agent pipeline for robotic manipulation tasks.
"""

import sys
import time
import traceback
from typing import Dict, Any
import numpy as np

# Add VIMABench to path
sys.path.append('../VIMABench')

def safe_serialize_for_logging(data: Any) -> Any:
    """
    Safely serialize data for logging by converting numpy arrays and other
    non-JSON-serializable objects to strings or basic Python types.

    Args:
        data: Data to serialize

    Returns:
        JSON-serializable version of the data
    """
    if isinstance(data, np.ndarray):
        return f"ndarray(shape={data.shape}, dtype={data.dtype})"
    elif isinstance(data, dict):
        return {key: safe_serialize_for_logging(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [safe_serialize_for_logging(item) for item in data]
    elif hasattr(data, '__dict__'):
        # Handle objects with __dict__ attribute
        return f"{type(data).__name__}(id={id(data)})"
    else:
        return data

# Try to import Hydra components
try:
    from omegaconf import OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    print("⚠️  OmegaConf not available. Install with: pip install omegaconf")

# Import our components
from vima_interface import VIMAInterface
from graph_setup import app, logger, set_grounder_config, set_segmentor_config, set_actor_config
from config.vima.vima_setup import VIMASetup

def load_hydra_config(config_path: str = "config/graph_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from Hydra YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration
    """
    if not HYDRA_AVAILABLE:
        print("❌ OmegaConf not available. Using default configuration.")
        return get_default_config()

    try:
        print(f"📄 Loading configuration from: {config_path}")
        cfg = OmegaConf.load(config_path)

        # Convert to dictionary for easier access
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        print("✅ Configuration loaded successfully")
        print(f"   - VIMA config: {config_dict['vima']}")

        return config_dict

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        print("🔄 Falling back to default configuration")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration when Hydra is not available."""
    return {
        "pipeline_name": "robotic_manipulation_pipeline",
        "enable_logging": True,
        "log_directory": "logs",
        "vima": {
            "task_name": "instruction_following/visual_manipulation",
            "modalities": ["rgb"],
            "debug": True,
            "display_debug_window": True,
            "hide_arm_rgb": False,
            "gui_delay": 0.1,
            "action_delay": 0.5,
            "camera_config": {
                "width": 1280,
                "height": 720,
                "fov": 60
            }
        },
        "node_timeout": 30.0,
        "max_retries": 3,
        "enable_node_profiling": False,
        "image_processing": {
            "enable_preprocessing": True,
            "normalization": True,
            "resize_images": False,
            "target_size": [640, 480]
        },
        "action_planning": {
            "max_actions_per_task": 10,
            "enable_orientation_control": True,
            "rotation_degrees_limit": 180,
            "position_tolerance": 0.01
        },
        "task": {
            "default_prompt": "Pick up the red block and place it on the table.",
            "enable_multi_object": False,
            "max_prompts_per_task": 5
        },
        "visualization": {
            "enable_realtime_display": True,
            "show_depth_overlay": False,
            "show_grasp_points": True,
            "update_frequency": 10
        },
        "performance": {
            "enable_multithreading": False,
            "max_workers": 4,
            "memory_limit_gb": 8
        }
    }

def create_vima_interface(config: Dict[str, Any]) -> VIMAInterface:
    """
    Create and configure VIMA interface based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured VIMAInterface instance
    """
    vima_config = config.get("vima", {})

    print("🔧 Creating VIMA interface with configuration...")
    print(f"   Task: {vima_config.get('task_name', 'Unknown')}")
    print(f"   Modalities: {vima_config.get('modalities', ['rgb'])}")
    print(f"   GUI: {'Enabled' if vima_config.get('display_debug_window', False) else 'Disabled'}")

    try:
        # Create VIMA setup and interface
        vima_setup = VIMASetup()
        print(f"🔧 VIMA setup created, config path: {vima_setup.config_path}")

        # Create VIMA interface with proper camera and rendering settings
        vima_interface = vima_setup.create_vima_interface(
            task_name=vima_config.get("task_name", "instruction_following/visual_manipulation"),
            display_debug_window=vima_config.get("display_debug_window", False),
            hide_arm_rgb=vima_config.get("hide_arm_rgb", True),  # Hide robot arm
            save_observation_images=vima_config.get("save_observation_images", True),
            observation_image_dir=vima_config.get("observation_image_dir", "observations"),
            camera_config=vima_config.get("camera_config", {
                "width": 1280,
                "height": 720,
                "fov": 60
            })
        )

        print("✅ VIMA interface created successfully")
        logger.flush()  # Ensure VIMA creation logs are written
        return vima_interface

    except Exception as e:
        print(f"❌ Failed to create VIMA interface: {e}")
        print(f"   Config path attempted: {vima_setup.config_path if 'vima_setup' in locals() else 'Unknown'}")
        raise

def run_pipeline_with_vima(config: Dict[str, Any], vima_interface: VIMAInterface) -> bool:
    """
    Run the robotic manipulation pipeline with VIMA integration.

    Args:
        config: Configuration dictionary
        vima_interface: Configured VIMA interface

    Returns:
        True if successful, False otherwise
    """
    try:
        print("\n🤖 Starting Robotic Manipulation Pipeline with VIMA Integration")
        print("=" * 70)

        # Get initial observation from VIMA
        print("📡 Getting initial observation from VIMA environment...")
        initial_observation = vima_interface.get_observation()
        print(f"✅ Observation retrieved: RGB image size {initial_observation.rgb_image.size}")

        # Get task prompt and assets from VIMA
        print("📝 Getting task prompt and assets...")
        vima_prompt, vima_assets = vima_interface.get_prompt_and_assets()
        print(f"✅ Task prompt: {vima_prompt[:100]}...")
        print(f"✅ Assets available: {list(vima_assets.keys())}")

        # Use VIMA prompt if available, otherwise fall back to default
        default_prompt = config.get("task", {}).get("default_prompt", "Perform robotic manipulation task")
        is_default_vima = vima_prompt == "Default robotic manipulation task"
        actual_prompt = vima_prompt if vima_prompt and not is_default_vima else default_prompt

        print(f"📝 Using prompt: {actual_prompt[:100]}...")

        # Create initial state for the graph
        initial_state = {
            "original_prompt": actual_prompt,
            "decomposed_prompts": [],
            "queue": [],
            "current_prompt": None,
            "object_of_interest": None,
            "not_object_of_interest": "",
            "results": {},
            "image": initial_observation.rgb_image,
            "depth_image": initial_observation.depth_image,  # Will be None since VIMA doesn't support depth
            "camera_matrix": initial_observation.camera_matrix,
            "rotation_matrix": initial_observation.rotation_matrix,
            "translation_vector": initial_observation.translation_vector,
            "grounder_output": [],
            "grasp_points": [],
            "grasp_points_3d": [],
            "thinker_output": {},
            "actor_output": {},
            "reflection_output": {},
            "should_terminate": False,
            "vima_prompt_assets": vima_assets,
            "vima_metadata": initial_observation.metadata,
            "vima_interface": vima_interface,
            "oracle_action": {},
        }

        # Set up configurations using OmegaConf-based loader
        try:
            from config.config_loader import load_grounder_config, load_segmentor_config

            # Load grounder and segmentor configurations directly as objects
            grounder_config_obj = load_grounder_config()
            segmentor_config_obj = load_segmentor_config()

            # Use config objects directly (no manual dictionary conversion needed)
            grounder_config = grounder_config_obj
            segmentor_config = segmentor_config_obj

        except Exception as e:
            print(f"⚠️  Error loading configurations: {e}, using defaults")
            # Fallback to default configurations (as objects)
            from config.config_classes import GrounderConfig, SegmentorConfig
            grounder_config = GrounderConfig(grounding_mode="simple", auto_fallback_to_simple=True, device="cuda")
            segmentor_config = SegmentorConfig(backend="box_only", device="cuda")

        set_grounder_config(grounder_config)

        # Handle logging for both config objects and dictionaries
        if hasattr(grounder_config, 'grounding_mode'):
            # Config object
            grounder_mode = grounder_config.grounding_mode
        else:
            # Dictionary
            grounder_mode = grounder_config.get('grounding_mode', 'simple')
        print(f"🔧 Grounder configured with mode: {grounder_mode}")

        set_segmentor_config(segmentor_config)

        # Handle logging for both config objects and dictionaries
        if hasattr(segmentor_config, 'backend'):
            # Config object
            segmentor_backend = segmentor_config.backend
        else:
            # Dictionary
            segmentor_backend = segmentor_config.get('backend', 'box_only')
        print(f"🔧 Segmentor configured with backend: {segmentor_backend}")

        # Handle actor configuration
        if HYDRA_AVAILABLE:
            actor_config = config.get('actor', {})
        else:
            # Default actor configuration (pipeline mode)
            actor_config = {"mode": "pipeline"}

        set_actor_config(actor_config)
        actor_mode = actor_config.get('mode', 'pipeline')
        print(f"🎯 Actor configured with mode: {actor_mode}")

        # Log pipeline start
        logger.log("Pipeline execution started", {
            "config": safe_serialize_for_logging(config),
            "initial_prompt": actual_prompt,
            "vima_assets": list(vima_assets.keys())
        })

        print("\n🔄 Executing multi-agent pipeline...")
        print("   Nodes: decomposer → perceptor → grounder → segmentor → projector → thinker → actor → reflector")

        # Run the graph (suppress automatic output printing)
        step_count = 0
        for step in app.stream(initial_state, output_keys=[]):
            step_count += 1
            node, state = next(iter(step.items()))

            print(f"✅ Step {step_count}: {node} completed")

            # Show reflection output if available
            # if node == "reflector" and state.get('reflection_output'):
            #     current_prompt = state.get('current_prompt', 'Unknown')
            #     reflection = state['reflection_output'].get(current_prompt, {})
            #     if reflection:
            #         task_complete = reflection.get('task_complete', False)
            #         verification = reflection.get('verification_result', 'Unknown')
            #         print(f"   📊 Reflection: Task complete: {task_complete}, Verification: {verification}")

            #         logger.log("Task reflection", {
            #             "task_complete": task_complete,
            #             "verification_result": verification
            #         })

            # if node == "actor":
            #     time.sleep(5)

            # Add small delay for better visualization
            time.sleep(0.1)

        print(f"\n✅ Pipeline completed successfully in {step_count} steps!")
        logger.log("Pipeline execution completed", {"total_steps": step_count})
        logger.flush()  # Ensure all execution logs are written

        return True

    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        logger.log("Pipeline execution failed", {"error": str(e)})
        traceback.print_exc()
        return False

def main():
    """Main entry point for the robotic manipulation pipeline."""
    print("🚀 Robotic Manipulation Pipeline with VIMA Integration")
    print("=" * 60)

    # Load configuration
    config = load_hydra_config()

    # Create VIMA interface
    vima_interface = None
    try:
        vima_interface = create_vima_interface(config)

        # Run the pipeline
        success = run_pipeline_with_vima(config, vima_interface)

        if success:
            print("\n🎉 Pipeline execution completed successfully!")
        else:
            print("\n❌ Pipeline execution failed!")

    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        logger.log("Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.log("Unexpected error", {"error": str(e)})
        traceback.print_exc()
    finally:
        # Cleanup
        if vima_interface:
            print("\n🧹 Cleaning up VIMA interface...")
            try:
                vima_interface.close_environment()
                print("✅ VIMA interface closed successfully")
            except Exception as e:
                print(f"⚠️  Warning: Error closing VIMA interface: {e}")

        # Flush logs
        try:
            logger.flush()
            print("✅ Logs flushed successfully")
        except Exception as e:
            print(f"⚠️  Warning: Error flushing logs: {e}")

    print("\n👋 Pipeline execution finished")

if __name__ == "__main__":
    main()
