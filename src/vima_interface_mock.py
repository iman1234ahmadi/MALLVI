#!/usr/bin/env python3
"""
Mock VIMA Interface for Testing
==============================

This is a mock version of VIMAInterface that works without PyBullet/VIMABench
for testing the clean architecture when dependencies are missing.
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class VIMAObservation:
    """Mock observation class"""
    rgb_image: Image.Image
    depth_image: Optional[Image.Image] = None
    camera_matrix: np.ndarray = None
    rotation_matrix: np.ndarray = None
    translation_vector: np.ndarray = None
    metadata: Dict[str, Any] = None

class MockVIMAInterface:
    """Mock VIMA interface that works without PyBullet"""
    
    def __init__(self, **kwargs):
        """Initialize mock VIMA interface"""
        self.task_name = kwargs.get("task_name", "instruction_following/visual_manipulation")
        self.modalities = kwargs.get("modalities", ["rgb"])
        self.debug = kwargs.get("debug", False)
        self.display_debug_window = kwargs.get("display_debug_window", False)
        self.hide_arm_rgb = kwargs.get("hide_arm_rgb", True)
        self.gui_delay = kwargs.get("gui_delay", 0.1)
        self.action_delay = kwargs.get("action_delay", 0.5)
        
        # Mock environment - always initialized
        self.env = "mock_environment"
        
        print("🎭 Mock VIMA interface initialized (PyBullet not available)")
    
    def get_observation(self) -> VIMAObservation:
        """Get a mock observation"""
        # Create a mock RGB image
        mock_image = Image.new('RGB', (640, 480), color='lightblue')
        
        # Create mock camera matrices
        camera_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
        rotation_matrix = np.eye(3, dtype=np.float32)
        translation_vector = np.array([0, 0, 0], dtype=np.float32)
        
        return VIMAObservation(
            rgb_image=mock_image,
            depth_image=None,
            camera_matrix=camera_matrix,
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector,
            metadata={"mock": True, "task": self.task_name}
        )
    
    def get_prompt_and_assets(self) -> Tuple[str, Dict[str, Any]]:
        """Get mock prompt and assets"""
        prompt = f"Mock task: {self.task_name}"
        assets = {
            "objects": ["red_block", "blue_cube"],
            "scene": "mock_scene"
        }
        return prompt, assets
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """Mock action execution"""
        print(f"🎭 Mock executing action: {action}")
        return True
    
    def close_environment(self):
        """Mock environment cleanup"""
        print("🎭 Mock VIMA environment closed")

# Replace the real VIMAInterface with mock when PyBullet is not available
try:
    from vima_interface import VIMAInterface
    VIMA_AVAILABLE = True
except ImportError:
    VIMAInterface = MockVIMAInterface
    VIMA_AVAILABLE = False
    print("🎭 Using mock VIMA interface (PyBullet not available)")
