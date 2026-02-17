"""
VIMA Interface Module
====================

Comprehensive interface for VIMABench integration with the robotic manipulation pipeline.
Handles environment setup, prompt retrieval, observations, and action execution.
"""
import sys
import os

# Force GPU selection for RTX 3060 (GPU0)
# This ensures PyBullet and CUDA use the dedicated GPU instead of integrated graphics
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYBULLET_GPU_DEVICE"] = "0"

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VIMABench'))

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image
import time
import logging
import traceback
from dataclasses import dataclass
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import VIMA utilities for quaternion conversion
try:
    from vima_bench.tasks.utils import misc_utils as vima_utils
    VIMA_UTILS_AVAILABLE = True
except ImportError:
    VIMA_UTILS_AVAILABLE = False

# Optional Hydra imports
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = Any
    HYDRA_AVAILABLE = False

try:
    from config.config_classes import VIMAConfig
    CONFIG_AVAILABLE = True
except ImportError:
    VIMAConfig = None
    CONFIG_AVAILABLE = False

@dataclass
class VIMAObservation:
    """Container for VIMA environment observations"""
    rgb_image: Image.Image
    depth_image: Optional[np.ndarray]  # VIMA doesn't support depth
    camera_matrix: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    task_prompt: str
    prompt_assets: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class VIMAAction:
    """Container for VIMA actions"""
    pose0_position: np.ndarray  # Pick position
    pose0_rotation: np.ndarray  # Pick rotation (Euler angles)
    pose1_position: np.ndarray  # Place position
    pose1_rotation: np.ndarray  # Place rotation (Euler angles)
    action_metadata: Optional[Dict[str, Any]] = None

@dataclass
class VIMATask:
    """Container for VIMA task information"""
    task_name: str
    prompt: str
    prompt_assets: Dict[str, Any]
    environment_config: Dict[str, Any]

class VIMAInterface:
    """
    Comprehensive interface for VIMABench environment management and interaction.

    This class provides methods to:
    - Initialize and manage VIMABench environments
    - Retrieve task prompts and associated assets
    - Get observations from the environment
    - Execute actions in the environment
    - Handle errors and provide logging
    """

    def __init__(
        self,
        task_name: Optional[str] = None,
        modalities: Optional[List[str]] = None,
        debug: bool = False,
        display_debug_window: bool = False,
        hide_arm_rgb: bool = True,  # Default to True for clean observations
        auto_reset: bool = True,
        config: Optional[Union[VIMAConfig, DictConfig, Dict[str, Any]]] = None,
        gui_delay: float = 0.1,
        action_delay: float = 0.5,
        enable_action_logging: bool = True,
        enable_observation_logging: bool = False,
        save_observation_images: bool = True,
        observation_image_dir: str = "observations",
        camera_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the VIMA interface.

        Args:
            task_name: Name of the VIMA task to use (e.g., "instruction_following/visual_manipulation")
            modalities: List of observation modalities (e.g., ["rgb"])
            debug: Enable debug mode
            display_debug_window: Show debug visualization window
            hide_arm_rgb: Hide robot arm in RGB observations (default: True for clean images)
            auto_reset: Automatically reset environment on initialization
            config: VIMA configuration object (VIMAConfig, DictConfig, or dict)
            gui_delay: Delay between GUI updates in seconds
            action_delay: Delay between actions in seconds
            enable_action_logging: Enable detailed action logging
            enable_observation_logging: Enable detailed observation logging
            save_observation_images: Save images when getting observations (default: True)
            observation_image_dir: Directory to save observation images (default: "observations")
        """
        # Initialize from config if provided
        if config is not None:
            self._initialize_from_config(config)
        else:
            # Use individual parameters
            self.task_name = task_name or "instruction_following/visual_manipulation"
            self.modalities = modalities or ["rgb"]
            self.debug = debug
            self.display_debug_window = display_debug_window
            self.hide_arm_rgb = hide_arm_rgb
            self.auto_reset = auto_reset
            self.gui_delay = gui_delay
            self.action_delay = action_delay
            self.enable_action_logging = enable_action_logging
            self.enable_observation_logging = enable_observation_logging
            self.save_observation_images = save_observation_images
            self.observation_image_dir = observation_image_dir
            self.camera_config = camera_config

        self.env = None
        self.current_task: Optional[VIMATask] = None
        self.vima_task = None  # Store actual VIMA task object for oracle access
        self.is_initialized = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Set logging level based on debug mode
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize environment
        self._initialize_environment()

    def _save_observation_image(self, image: Image.Image, observation_type: str = "observation",
                               suffix: str = "") -> str:
        """Save an observation image to disk.

        Args:
            image: PIL Image to save
            observation_type: Type of observation (e.g., "initial", "action", "final")
            suffix: Optional suffix for filename

        Returns:
            Path to saved image file
        """
        if not self.save_observation_images:
            return ""

        try:
            import os
            from datetime import datetime

            # Create directory if it doesn't exist
            os.makedirs(self.observation_image_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"vima_{observation_type}_{timestamp}{suffix}.png"
            filepath = os.path.join(self.observation_image_dir, filename)

            # Save image
            image.save(filepath)
            self.logger.info(f"💾 Saved observation image: {filepath}")

            return filepath

        except Exception as e:
            self.logger.warning(f"Failed to save observation image: {e}")
            return ""

    def _initialize_from_config(self, config: Union[VIMAConfig, DictConfig, Dict[str, Any]]) -> None:
        """Initialize parameters from configuration object."""
        if CONFIG_AVAILABLE and isinstance(config, VIMAConfig):
            # Native VIMAConfig
            self.task_name = config.task_name or "instruction_following/visual_manipulation"
            self.modalities = config.modalities
            self.debug = config.debug
            self.display_debug_window = config.display_debug_window
            self.hide_arm_rgb = config.hide_arm_rgb
            self.auto_reset = True  # Always auto-reset when using config
            self.gui_delay = config.gui_delay
            self.action_delay = config.action_delay
            self.enable_action_logging = config.enable_action_logging
            self.enable_observation_logging = config.enable_observation_logging
            self.save_observation_images = config.save_observation_images
            self.observation_image_dir = config.observation_image_dir
            self.camera_config = config.get('camera_config', {
                "width": 1280,
                "height": 720,
                "fov": 60
            })
        elif HYDRA_AVAILABLE and isinstance(config, DictConfig):
            # Hydra DictConfig
            self.task_name = config.get('task_name', 'instruction_following/visual_manipulation')
            self.modalities = list(config.get('modalities', ["rgb"]))
            self.debug = config.get('debug', False)
            self.display_debug_window = config.get('display_debug_window', True)
            self.hide_arm_rgb = config.get('hide_arm_rgb', True)  # Default to True for clean images
            self.auto_reset = True
            self.gui_delay = config.get('gui_delay', 0.1)
            self.action_delay = config.get('action_delay', 0.5)
            self.enable_action_logging = config.get('enable_action_logging', True)
            self.enable_observation_logging = config.get('enable_observation_logging', False)
            self.save_observation_images = config.get('save_observation_images', True)
            self.observation_image_dir = config.get('observation_image_dir', 'observations')
            self.camera_config = config.get('camera_config', {
                "width": 1280,
                "height": 720,
                "fov": 60
            })
        elif isinstance(config, dict):
            # Plain dictionary
            self.task_name = config.get('task_name', 'instruction_following/visual_manipulation')
            self.modalities = config.get('modalities', ["rgb"])
            self.debug = config.get('debug', False)
            self.display_debug_window = config.get('display_debug_window', True)
            self.hide_arm_rgb = config.get('hide_arm_rgb', True)  # Default to True for clean images
            self.auto_reset = True
            self.gui_delay = config.get('gui_delay', 0.1)
            self.action_delay = config.get('action_delay', 0.5)
            self.enable_action_logging = config.get('enable_action_logging', True)
            self.enable_observation_logging = config.get('enable_observation_logging', False)
            self.save_observation_images = config.get('save_observation_images', True)
            self.observation_image_dir = config.get('observation_image_dir', 'observations')
            self.camera_config = config.get('camera_config', {
                "width": 1280,
                "height": 720,
                "fov": 60
            })
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

    def _initialize_environment(self) -> bool:
        """Initialize the VIMABench environment and set up the initial task."""
        try:
            self.logger.info("🔧 Initializing VIMABench environment...")

            # Import VIMABench
            try:
                from vima_bench import make
                self.logger.info("✅ Successfully imported vima_bench.make")
            except ImportError as import_error:
                self.logger.error(f"❌ Failed to import VIMABench: {import_error}")
                self._log_import_troubleshooting()
                return False

            # Set up the initial task (this will create the environment)
            if not self.setup_task(self.task_name):
                return False

            self.is_initialized = True
            self.logger.info("🎉 VIMABench environment initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Unexpected error during VIMABench initialization: {e}")
            self.logger.error(f"❌ Unexpected error type: {type(e).__name__}")
            self.logger.error("🔍 Full unexpected error stack trace:")
            self._log_stack_trace()
            return False

    def _log_import_troubleshooting(self):
        """Log detailed troubleshooting information for import errors."""
        self.logger.error("💡 Troubleshooting steps:")
        self.logger.error("   1. Activate virtual environment: source mallvi/bin/activate")
        self.logger.error("   2. Install VIMABench: cd VIMABench && pip install -e .")
        self.logger.error("   3. Check Python path includes VIMABench location")
        self.logger.error("🔍 Full import error stack trace:")
        self._log_stack_trace()

    def _log_stack_trace(self):
        """Log the current stack trace in a formatted way."""
        stack_lines = traceback.format_exc().split('\n')
        for i, line in enumerate(stack_lines):
            if line.strip():
                self.logger.error(f"   [{i:02d}] {line}")

    @contextmanager
    def task_context(self, task_name: Optional[str] = None):
        """
        Context manager for task-specific operations.

        Args:
            task_name: Name of the task to set up
        """
        previous_task = self.current_task

        if task_name:
            self.setup_task(task_name)

        try:
            yield
        finally:
            if previous_task != self.current_task:
                self.current_task = previous_task

    def setup_task(self, task_name: str) -> bool:
        """
        Set up a specific task in the environment.

        Args:
            task_name: Name of the task to set up

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Setting up task: {task_name}")

            # Create new environment for specific task
            from vima_bench import make


            print("========================================")
            print(f"Hide arm RGB: {self.hide_arm_rgb}")
            print("========================================")

            # Prepare environment configuration
            env_kwargs = {
                "task_name": task_name,
                "modalities": self.modalities,
                "debug": self.debug,
                "display_debug_window": self.display_debug_window,
                "hide_arm_rgb": self.hide_arm_rgb,
                "task_kwargs": {"obs_img_size": (self.camera_config["height"], self.camera_config["width"])}
            }

            # Create VIMA environment
            # Note: Camera view selection happens during observation extraction from the "rgb" dict
            self.env = make(**env_kwargs)

            # Try to get the VIMA task object for oracle access
            # VIMA environments typically have the task as an attribute
            if hasattr(self.env, 'task'):
                self.vima_task = self.env.task
                self.logger.info("✅ Found VIMA task object for oracle access")
            elif hasattr(self.env, 'task_name'):
                # Try to get task from task registry if available
                try:
                    from vima_bench import TASKS
                    if self.env.task_name in TASKS:
                        self.vima_task = TASKS[self.env.task_name]
                        self.logger.info("✅ Retrieved VIMA task from task registry")
                    else:
                        self.logger.warning("⚠️  Task name found but not in TASKS registry")
                except ImportError:
                    self.logger.warning("⚠️  Could not import TASKS registry")
            else:
                self.logger.warning("⚠️  Could not find VIMA task object for oracle access")

            # Reset to get initial observation and task info
            try:
                obs, info = self.env.reset()
                self.logger.info("✅ Environment reset successfully")

                # Store last observation for potential reuse
                if hasattr(self.env, '_last_obs'):
                    self.env._last_obs = obs

            except OverflowError as overflow_error:
                self.logger.warning(f"⚠️  Segmentation overflow error during reset: {overflow_error}")
                self.logger.info("🔧 Attempting to continue despite segmentation overflow...")
                # Try to get observation manually if reset fails
                try:
                    if hasattr(self.env, '_get_obs'):
                        obs = self.env._get_obs()
                        self.logger.info("✅ Got observation manually after overflow error")

                        # Store last observation for potential reuse
                        if hasattr(self.env, '_last_obs'):
                            self.env._last_obs = obs

                    else:
                        self.logger.error("❌ Cannot recover from segmentation overflow error")
                        return False
                except Exception as recovery_error:
                    self.logger.error(f"❌ Failed to recover from overflow error: {recovery_error}")
                    return False

            # Extract task information directly from environment
            prompt = getattr(self.env, 'prompt', '')
            prompt_assets = getattr(self.env, 'prompt_assets', {})

            # Inject object information into the prompt
            if prompt_assets:
                self.logger.info(f"🔧 Injecting object information into task setup prompt "
                                f"(assets: {list(prompt_assets.keys())})")
                enhanced_prompt = self._inject_object_info_into_prompt(prompt, prompt_assets)
            else:
                self.logger.info("ℹ️  No prompt assets available for task setup injection")
                enhanced_prompt = prompt

            self.current_task = VIMATask(
                task_name=task_name,
                prompt=enhanced_prompt,
                prompt_assets=prompt_assets,
                environment_config={
                    "modalities": self.modalities,
                    "debug": self.debug,
                    "display_debug_window": self.display_debug_window,
                    "hide_arm_rgb": self.hide_arm_rgb
                }
            )

            self.logger.info(f"Task '{task_name}' set up successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set up task '{task_name}': {e}")
            self.logger.error("🔍 Full task setup stack trace:")
            self._log_stack_trace()
            return False

    def get_prompt_and_assets(self, task_name: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Retrieve the current task prompt and associated assets.

        Args:
            task_name: Optional task name to set up before retrieval

        Returns:
            Tuple of (prompt_string, prompt_assets_dict)
        """
        if task_name and (not self.current_task or self.current_task.task_name != task_name):
            if not self.setup_task(task_name):
                raise RuntimeError(f"Failed to set up task '{task_name}'")

        if not self.env:
            raise RuntimeError("No environment is currently set up")

        # Access prompt and prompt_assets directly from the environment
        prompt = getattr(self.env, 'prompt', '')
        prompt_assets = getattr(self.env, 'prompt_assets', {})

        # Inject object information into the prompt
        if prompt_assets:
            self.logger.info(f"🔧 Injecting object information into prompt (assets: {list(prompt_assets.keys())})")
            enhanced_prompt = self._inject_object_info_into_prompt(prompt, prompt_assets)
        else:
            self.logger.info("ℹ️  No prompt assets available for injection")
            enhanced_prompt = prompt

        return enhanced_prompt, prompt_assets

    def _inject_object_info_into_prompt(self, prompt: str, prompt_assets: Dict[str, Any]) -> str:
        """
        Inject object names and colors from prompt_assets into the prompt text.

        Args:
            prompt: Original prompt text
            prompt_assets: Dictionary containing object information

        Returns:
            Enhanced prompt with object details injected
        """
        if not prompt or not prompt_assets:
            return prompt

        enhanced_prompt = prompt

        # Look for dragged_obj and base_obj in prompt_assets
        for obj_key, obj_info in prompt_assets.items():
            if isinstance(obj_info, dict) and 'obj_info' in obj_info["segm"]:
                obj_details = obj_info["segm"]["obj_info"]
                obj_name = obj_details.get('obj_name', '')
                obj_color = obj_details.get('obj_color', '')

                self.logger.info(f"{obj_name=}")
                self.logger.info(f"{obj_color=}")
                if obj_name and obj_color:
                    # Replace placeholder in prompt with actual object details
                    placeholder = f"{{{obj_key}}}"
                    replacement = f"{obj_name} ({obj_color})"

                    if placeholder in enhanced_prompt:
                        enhanced_prompt = enhanced_prompt.replace(placeholder, replacement)
                        self.logger.info(f"🎯 Injected object info: {placeholder} -> {replacement}")
                    else:
                        # If placeholder not found, append object info to prompt
                        enhanced_prompt += f" The {obj_key} is a {obj_color} {obj_name}."
                        self.logger.info(f"📝 Appended object info: {obj_key} is a {obj_color} {obj_name}")

        return enhanced_prompt

    def _extract_prompt_from_obs(self, obs: Dict[str, Any]) -> str:
        """Extract task prompt from observation dictionary."""
        # VIMA observations typically contain task information
        if "prompt" in obs:
            return obs["prompt"]
        elif "task_description" in obs:
            return obs["task_description"]
        elif "instruction" in obs:
            return obs["instruction"]
        else:
            # Fallback to a default prompt if none found
            return "Perform the robotic manipulation task shown in the environment."

    def _extract_prompt_assets_from_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract prompt assets (objects, constraints, etc.) from observation."""
        assets = {}

        # Extract various types of assets that might be in the observation
        asset_keys = [
            "objects", "object_positions", "target_positions",
            "constraints", "assets", "scene_objects", "task_objects"
        ]

        for key in asset_keys:
            if key in obs:
                assets[key] = obs[key]

        # Also extract any metadata that might be relevant
        if "metadata" in obs:
            assets["metadata"] = obs["metadata"]

        return assets

    def get_observation(self) -> VIMAObservation:
        """
        Get current observation from the environment without resetting.
        This method tries to use cached observations first, then falls back to
        getting the current observation without reset, and only resets as a last resort.

        Returns:
            VIMAObservation containing all observation data
        """
        if not self.env:
            raise RuntimeError("Environment not initialized")

        try:
            # Get observation from environment without resetting
            if hasattr(self.env, '_last_obs') and self.env._last_obs is not None:
                # Use stored observation if available
                obs = self.env._last_obs
                info = {}
            elif hasattr(self.env, '_get_obs'):
                # Try to get current observation without reset
                obs = self.env._get_obs()
                info = {}
            else:
                # Fallback to reset if no other method available
                obs, info = self.env.reset()
                # Store for future use
                if hasattr(self.env, '_last_obs'):
                    self.env._last_obs = obs

            # Extract RGB image
            rgb_image = self._extract_rgb_image(obs)

            # Extract depth image
            depth_image = self._extract_depth_image(obs)

            # Extract camera parameters
            camera_matrix, rotation_matrix, translation_vector = self._extract_camera_params(obs)

            # Extract task information directly from environment
            task_prompt = getattr(self.env, 'prompt', '')
            prompt_assets = getattr(self.env, 'prompt_assets', {})

            # Inject object information into the prompt
            if prompt_assets:
                self.logger.info(f"🔧 Injecting object information into observation prompt "
                                f"(assets: {list(prompt_assets.keys())})")
                enhanced_task_prompt = self._inject_object_info_into_prompt(task_prompt, prompt_assets)
            else:
                self.logger.info("ℹ️  No prompt assets available for observation injection")
                enhanced_task_prompt = task_prompt

            # Save observation image if enabled
            saved_image_path = ""
            if rgb_image and self.save_observation_images:
                saved_image_path = self._save_observation_image(rgb_image, "observation")

            # Create observation object
            observation = VIMAObservation(
                rgb_image=rgb_image,
                depth_image=depth_image,
                camera_matrix=camera_matrix,
                rotation_matrix=rotation_matrix,
                translation_vector=translation_vector,
                task_prompt=enhanced_task_prompt,
                prompt_assets=prompt_assets,
                metadata=info
            )

            # Add saved image path to metadata
            if saved_image_path:
                if observation.metadata is None:
                    observation.metadata = {}
                observation.metadata["saved_image_path"] = saved_image_path

            if self.enable_observation_logging:
                self.logger.info("✅ Observation retrieved successfully")
                self.logger.debug(f"RGB image size: {rgb_image.size if rgb_image else 'None'}")
                depth_status = 'Available' if depth_image is not None else 'None (VIMA doesn\'t support depth)'
                self.logger.debug(f"Depth image: {depth_status}")
                self.logger.debug(f"Task prompt: {enhanced_task_prompt[:100]}...")
                self.logger.debug(f"Prompt assets keys: {list(prompt_assets.keys())}")
                if saved_image_path:
                    self.logger.debug(f"Saved image: {saved_image_path}")

            return observation

        except Exception as e:
            self.logger.error(f"Failed to get observation: {e}")
            # Return a fallback observation
            return self._create_fallback_observation()

    def get_fresh_observation(self) -> VIMAObservation:
        """
        Get a fresh observation by resetting the environment.
        Use this when you want to start a new episode or reset the environment state.
        For getting the current observation without changing state, use get_observation().

        Returns:
            VIMAObservation containing all observation data
        """
        if not self.env:
            raise RuntimeError("Environment not initialized")

        try:
            # Reset environment to get fresh observation
            obs, info = self.env.reset()

            # Store last observation for potential reuse
            if hasattr(self.env, '_last_obs'):
                self.env._last_obs = obs

            # Extract RGB image
            rgb_image = self._extract_rgb_image(obs)

            # Extract depth image
            depth_image = self._extract_depth_image(obs)

            # Extract camera parameters
            camera_matrix, rotation_matrix, translation_vector = self._extract_camera_params(obs)

            # Extract task information directly from environment
            task_prompt = getattr(self.env, 'prompt', '')
            prompt_assets = getattr(self.env, 'prompt_assets', {})

            # Inject object information into the prompt
            if prompt_assets:
                self.logger.info(f"🔧 Injecting object information into fresh observation prompt "
                                f"(assets: {list(prompt_assets.keys())})")
                enhanced_task_prompt = self._inject_object_info_into_prompt(task_prompt, prompt_assets)
            else:
                self.logger.info("ℹ️  No prompt assets available for fresh observation injection")
                enhanced_task_prompt = task_prompt

            # Save observation image if enabled
            saved_image_path = ""
            if rgb_image and self.save_observation_images:
                saved_image_path = self._save_observation_image(rgb_image, "fresh_observation")

            # Create observation object
            observation = VIMAObservation(
                rgb_image=rgb_image,
                depth_image=depth_image,
                camera_matrix=camera_matrix,
                rotation_matrix=rotation_matrix,
                translation_vector=translation_vector,
                task_prompt=enhanced_task_prompt,
                prompt_assets=prompt_assets,
                metadata=info
            )

            # Add saved image path to metadata
            if saved_image_path:
                if observation.metadata is None:
                    observation.metadata = {}
                observation.metadata["saved_image_path"] = saved_image_path

            if self.enable_observation_logging:
                self.logger.info("✅ Fresh observation retrieved successfully")
                self.logger.debug(f"RGB image size: {rgb_image.size if rgb_image else 'None'}")
                depth_status = 'Available' if depth_image is not None else 'None (VIMA doesn\'t support depth)'
                self.logger.debug(f"Depth image: {depth_status}")
                self.logger.debug(f"Task prompt: {enhanced_task_prompt[:100]}...")
                self.logger.debug(f"Prompt assets keys: {list(prompt_assets.keys())}")
                if saved_image_path:
                    self.logger.debug(f"Saved image: {saved_image_path}")

            return observation

        except Exception as e:
            self.logger.error(f"Failed to get fresh observation: {e}")
            # Return a fallback observation
            return self._create_fallback_observation()

    def _extract_rgb_image(self, obs: Dict[str, Any]) -> Image.Image:
        """Extract RGB image from observation using top camera view."""
        if "rgb" in obs:
            rgb_data = obs["rgb"]
            if isinstance(rgb_data, dict):
                # VIMA provides multiple camera views: "front", "top", etc.
                # Use "top" view for better observation of manipulation tasks
                if "top" in rgb_data:
                    # VIMA format: obs["rgb"]["top"] is (3, H, W) array
                    rgb_array = rgb_data["top"]
                    self.logger.debug("Using top camera view for observation")
                elif "front" in rgb_data:
                    # Fallback to front view if top is not available
                    rgb_array = rgb_data["front"]
                    self.logger.debug("Using front camera view (top not available)")
                else:
                    # Use the first available camera view
                    first_view = next(iter(rgb_data.keys()))
                    rgb_array = rgb_data[first_view]
                    self.logger.debug(f"Using {first_view} camera view")

                if isinstance(rgb_array, np.ndarray):
                    # Convert from (3, H, W) to (H, W, 3) and then to PIL Image
                    rgb_array = np.transpose(rgb_array, (1, 2, 0))
                    return Image.fromarray(rgb_array)
            elif isinstance(rgb_data, np.ndarray):
                # Direct numpy array
                if rgb_data.ndim == 3 and rgb_data.shape[0] == 3:
                    # (3, H, W) format
                    rgb_array = np.transpose(rgb_data, (1, 2, 0))
                    return Image.fromarray(rgb_array)
                elif rgb_data.ndim == 3 and rgb_data.shape[2] == 3:
                    # (H, W, 3) format
                    return Image.fromarray(rgb_data)
                else:
                    # Grayscale or other format
                    return Image.fromarray(rgb_data)

        # Fallback: create a dummy RGB image
        self.logger.warning("No RGB image found in observation, creating dummy")
        return Image.new('RGB', (640, 480), color='gray')

    def _extract_depth_image(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract depth image from observation. VIMA doesn't support depth."""
        # VIMA doesn't support depth modality
        self.logger.info("VIMA doesn't support depth modality - returning None")
        return None

    def _extract_camera_params(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract camera parameters from observation."""
        # Default camera parameters (can be overridden by actual observation data)
        camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
        rotation_matrix = np.eye(3)
        translation_vector = np.array([0, 0, 0])

        # Try to extract from observation if available
        if "camera_matrix" in obs:
            camera_matrix = np.array(obs["camera_matrix"])
        if "rotation_matrix" in obs:
            rotation_matrix = np.array(obs["rotation_matrix"])
        if "translation_vector" in obs:
            translation_vector = np.array(obs["translation_vector"])

        return camera_matrix, rotation_matrix, translation_vector

    def _create_fallback_observation(self) -> VIMAObservation:
        """Create a fallback observation when the real one fails."""
        self.logger.warning("Creating fallback observation due to error")

        # Try to get prompt and prompt_assets from environment if available
        task_prompt = "Task information unavailable"
        prompt_assets = {}
        if self.env:
            task_prompt = getattr(self.env, 'prompt', task_prompt)
            prompt_assets = getattr(self.env, 'prompt_assets', prompt_assets)

    def euler_to_quaternion(self, euler_angles: np.ndarray) -> List[float]:
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion format.

        Args:
            euler_angles: Numpy array with [roll, pitch, yaw] in radians

        Returns:
            List of [x, y, z, w] quaternion components
        """
        try:
            # Use VIMA's quaternion conversion if available
            if VIMA_UTILS_AVAILABLE:
                # VIMA expects (roll, pitch, yaw) format
                quaternion = vima_utils.euler_to_quaternion(euler_angles)
                return quaternion.tolist()
            else:
                # Fallback to scipy if VIMA utils not available
                try:
                    from scipy.spatial.transform import Rotation as R
                    rot = R.from_euler('xyz', euler_angles)
                    quaternion = rot.as_quat()  # Returns [x, y, z, w]
                    return quaternion.tolist()
                except ImportError:
                    # Manual quaternion conversion as last resort
                    roll, pitch, yaw = euler_angles
                    cr = np.cos(roll * 0.5)
                    sr = np.sin(roll * 0.5)
                    cp = np.cos(pitch * 0.5)
                    sp = np.sin(pitch * 0.5)
                    cy = np.cos(yaw * 0.5)
                    sy = np.sin(yaw * 0.5)

                    w = cr * cp * cy + sr * sp * sy
                    x = sr * cp * cy - cr * sp * sy
                    y = cr * sp * cy + sr * cp * sy
                    z = cr * cp * sy - sr * sp * cy

                    return [x, y, z, w]
        except Exception as e:
            self.logger.warning(f"Failed to convert Euler to quaternion: {e}, using identity quaternion")
            return [0.0, 0.0, 0.0, 1.0]  # Identity quaternion

    def get_oracle_action(self, observation: Dict[str, Any]) -> Optional[Any]:
        """
        Get the oracle action for the current VIMA task.

        Args:
            observation: Current observation from the environment

        Returns:
            Oracle action if available, None otherwise
        """
        if not self.vima_task:
            self.logger.warning("⚠️  No VIMA task available for oracle action")
            return None

        try:
            # Get oracle function as described by user
            oracle_fn = self.vima_task.oracle(self.env)
            oracle_action = oracle_fn.act(observation)

            self.logger.info("✅ Retrieved oracle action from VIMA task")
            return oracle_action

        except Exception as e:
            self.logger.error(f"❌ Failed to get oracle action: {e}")
            return None

        return VIMAObservation(
            rgb_image=Image.new('RGB', (640, 480), color='gray'),
            depth_image=None,  # VIMA doesn't support depth
            camera_matrix=np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]),
            rotation_matrix=np.eye(3),
            translation_vector=np.array([0, 0, 0]),
            task_prompt=task_prompt,
            prompt_assets=prompt_assets,
            metadata={"error": "Fallback observation created due to error"}
        )

    def convert_actor_action_to_vima_format(self, actor_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert actor action dict to VIMA action format.

        Args:
            actor_action: Dict with format {'pose0': {'position': [...], 'rotation': [...]}, 'pose1': {...}}

        Returns:
            Dict in VIMA format with quaternions and proper numpy arrays
        """
        try:
            # Extract positions and rotations
            pose0_pos = actor_action['pose0']['position']
            pose0_rot = actor_action['pose0']['rotation']
            pose1_pos = actor_action['pose1']['position']
            pose1_rot = actor_action['pose1']['rotation']

            # Convert to numpy arrays with correct dtypes
            pose0_position = np.array(pose0_pos[:2], dtype=np.float32)  # Only x, y coordinates
            pose1_position = np.array(pose1_pos[:2], dtype=np.float32)  # Only x, y coordinates

            # Convert Euler angles to quaternions
            if VIMA_UTILS_AVAILABLE:
                pose0_rotation = np.array(vima_utils.eulerXYZ_to_quatXYZW(pose0_rot), dtype=np.float32)
                pose1_rotation = np.array(vima_utils.eulerXYZ_to_quatXYZW(pose1_rot), dtype=np.float32)
            else:
                # Fallback: convert Euler to quaternion manually
                # For zero rotation, quaternion is [0, 0, 0, 1]
                pose0_rotation = np.array([0., 0., 0., 1.], dtype=np.float32)
                pose1_rotation = np.array([0., 0., 0., 1.], dtype=np.float32)
                self.logger.warning("VIMA utils not available, using identity quaternions")

            action = {
                'pose0_position': pose0_position,
                'pose0_rotation': pose0_rotation,
                'pose1_position': pose1_position,
                'pose1_rotation': pose1_rotation
            }

            print("========================================")
            print(f"Action: {action}")
            print("========================================")

            return action

        except Exception as e:
            self.logger.error(f"Failed to convert actor action to VIMA format: {e}")
            raise

    def execute_action(self, action: Union[VIMAAction, Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a single action in the VIMA environment.

        Args:
            action: VIMAAction object or dict containing the action to execute

        Returns:
            Tuple of (success, result_info)
        """
        if not self.env:
            raise RuntimeError("Environment not initialized")

        try:
            # Handle different action formats
            if isinstance(action, dict):
                # Convert actor dict format to VIMA format
                if 'pose0' in action and 'pose1' in action:
                    vima_action = self.convert_actor_action_to_vima_format(action)
                else:
                    # Assume it's already in VIMA format
                    vima_action = action
            else:
                # Convert VIMAAction object to VIMA dict format with quaternions
                # Convert Euler angles (roll, pitch, yaw) to quaternions
                pose0_quaternion = self.euler_to_quaternion(action.pose0_rotation)
                pose1_quaternion = self.euler_to_quaternion(action.pose1_rotation)

                vima_action = {
                    "pose0_position": action.pose0_position.tolist()[:2],  # Only x, y coordinates
                    "pose0_rotation": pose0_quaternion,
                    "pose1_position": action.pose1_position.tolist()[:2],  # Only x, y coordinates
                    "pose1_rotation": pose1_quaternion
                }

                if self.enable_action_logging:
                    self.logger.debug(f"Converted VIMAAction to dict format: {vima_action}")

            if self.enable_action_logging:
                self.logger.info("Executing action in VIMA environment")
                if isinstance(action, dict):
                    self.logger.debug(f"Action format: dict")
                else:
                    self.logger.debug(f"Action details: pick={action.pose0_position}, place={action.pose1_position}")

            # Execute action
            try:
                obs, reward, done, truncated, info = self.env.step(vima_action)
            except OverflowError as overflow_error:
                self.logger.warning(f"⚠️  Segmentation overflow error during action execution: {overflow_error}")
                # Create dummy observation data since we can't get real observation
                obs = {"rgb": {"front": np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8)}}
                reward = 0.0
                done = False
                truncated = False
                info = {"overflow_error": str(overflow_error)}
                self.logger.info("🔧 Continuing with dummy observation data due to segmentation overflow")

            # Prepare result info
            result_info = {
                "success": reward > 0 or done,
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "info": info,
                "observation": obs
            }

            # Store last observation for potential reuse
            if hasattr(self.env, '_last_obs'):
                self.env._last_obs = obs

            if self.enable_action_logging:
                self.logger.info(f"Action executed successfully. Reward: {reward}")

            # Add GUI delay for real-time visualization
            if hasattr(self, 'gui_delay') and self.display_debug_window:
                time.sleep(self.gui_delay)

            return result_info["success"], result_info

        except Exception as e:
            self.logger.error(f"Failed to execute action: {e}")
            return False, {"error": str(e)}

    def execute_action_sequence(self, actions: List[Union[VIMAAction, Dict[str, Any]]]) -> \
            Tuple[bool, List[Dict[str, Any]]]:
        """
        Execute a sequence of actions in the VIMA environment.

        Args:
            actions: List of VIMAAction objects or dicts to execute

        Returns:
            Tuple of (overall_success, list_of_results)
        """
        if not self.env:
            raise RuntimeError("Environment not initialized")

        results = []
        overall_success = True

        # Use configured action delay, default to 0.5 if not set
        action_delay = getattr(self, 'action_delay', 0.5)

        for i, action in enumerate(actions):
            if self.enable_action_logging:
                self.logger.info(f"Executing action {i+1}/{len(actions)}")

            success, result = self.execute_action(action)
            results.append(result)

            if not success:
                overall_success = False
                if self.enable_action_logging:
                    self.logger.warning(f"Action {i+1} failed, but continuing sequence")

            # Configurable delay between actions (skip delay after last action)
            if i < len(actions) - 1:
                time.sleep(action_delay)

        return overall_success, results

    def reset_environment(self) -> bool:
        """
        Reset the environment to initial state.

        Returns:
            True if successful, False otherwise
        """
        if not self.env:
            self.logger.error("Environment not initialized")
            return False

        try:
            obs, info = self.env.reset()

            # Store last observation for potential reuse
            if hasattr(self.env, '_last_obs'):
                self.env._last_obs = obs

            self.logger.info("Environment reset successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset environment: {e}")
            return False

    def close_environment(self):
        """Close the VIMA environment and clean up resources."""
        if self.env:
            try:
                self.env.close()
                self.logger.info("VIMA environment closed successfully")
            except Exception:
                pass
            finally:
                self.env = None
                self.current_task = None
                self.is_initialized = False
