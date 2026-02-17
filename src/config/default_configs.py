#!/usr/bin/env python3
"""
Default Configuration Generator
==============================

This module generates default configuration files when they don't exist.
"""

from pathlib import Path
from typing import Dict, Any
from utils.printer import Printer
from .yaml_handler import YAMLHandler

def generate_default_pipeline_config() -> Dict[str, Any]:
    """Generate default pipeline configuration"""
    return {
        "pipeline_name": "robotic_manipulation_pipeline",
        "enable_logging": True,
        "log_directory": "logs",
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

def generate_default_vima_config() -> Dict[str, Any]:
    """Generate default VIMA configuration"""
    return {
        "vima": {
            "task_name": "instruction_following/visual_manipulation",
            "modalities": ["rgb"],
            "debug": False,
            "display_debug_window": True,
            "show_gui": True,
            "hide_arm_rgb": True,
            "gui_delay": 0.1,
            "action_delay": 0.5,
            "camera_config": {
                "width": 1280,
                "height": 720,
                "fov": 60
            }
        }
    }

def generate_default_grounder_config() -> Dict[str, Any]:
    """Generate default grounder configuration"""
    return {
        "grounder": {
            "grounding_mode": "simple",
            "device": "cuda",
            "auto_fallback_to_simple": True,
            "box_threshold": 0.3,
            "text_threshold": 0.25,
            "max_attempts": 20,
            "min_box_threshold": 0.1,
            "min_text_threshold": 0.1,
            "max_box_threshold": 0.8,
            "max_text_threshold": 0.8,
            "box_reduction_factor": 0.9,
            "text_reduction_factor": 0.9,
            "box_increase_factor": 1.1,
            "text_increase_factor": 1.1
        }
    }

def generate_default_segmentor_config() -> Dict[str, Any]:
    """Generate default segmentor configuration"""
    return {
        "segmentor": {
            "backend": "box_only",
            "device": "cuda",
            "points_per_box": 1,
            "min_area": 10,
            "dt_suppress_radius": 8,
            "axis_order": "xy",
            "point_mode": "auto",
            "sam_model_type": "sam",
            "sam_checkpoint": None,
            "sam_config": None
        }
    }

def create_default_config_files(config_dir: Path):
    """
    Create default configuration files if they don't exist.
    
    Args:
        config_dir: Directory to create config files in
    """
    Printer.info("Creating default configuration files...")
    
    # Ensure config directory exists
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create vima subdirectory
    vima_dir = config_dir / "vima"
    vima_dir.mkdir(exist_ok=True)
    
    # Generate and save default configs
    configs_to_create = [
        (config_dir / "graph_config.yaml", generate_default_pipeline_config()),
        (vima_dir / "vima_config.yaml", generate_default_vima_config()),
        (config_dir / "grounder_config.yaml", generate_default_grounder_config()),
        (config_dir / "segmentor_config.yaml", generate_default_segmentor_config()),
    ]
    
    for file_path, config_data in configs_to_create:
        if not file_path.exists():
            try:
                YAMLHandler.save_yaml(config_data, file_path)
                Printer.success(f"Created default config: {file_path}")
            except Exception as e:
                Printer.error(f"Failed to create {file_path}: {e}")
        else:
            Printer.debug(f"Config already exists: {file_path}")

def get_default_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get all default configurations as a dictionary.
    
    Returns:
        Dictionary containing all default configurations
    """
    return {
        "pipeline": generate_default_pipeline_config(),
        "vima": generate_default_vima_config()["vima"],
        "grounder": generate_default_grounder_config()["grounder"],
        "segmentor": generate_default_segmentor_config()["segmentor"],
    }
