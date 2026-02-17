#!/usr/bin/env python3
"""
Main Entry Point - Clean Architecture
====================================

This script orchestrates the robotic manipulation pipeline with clean separation of concerns.
It loads configuration, creates VIMA interface, sets up the graph, and runs the pipeline.
"""

import sys
import time
import traceback
from typing import Dict, Any
import numpy as np

# Add VIMABench to path
sys.path.append('../VIMABench')

from utils.printer import Printer

# Import our clean modules
from config.config_manager import config_manager
from config.vima_manager import VIMAManager
from graph_manager import GraphManager

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

def load_configuration() -> Dict[str, Any]:
    """
    Load all configurations using the unified config manager.
    
    Returns:
        Dictionary containing all loaded configurations
    """
    Printer.header("Loading Configuration")
    
    try:
        configs = config_manager.load_all_configs()
        Printer.success("Configuration loaded successfully")
        return configs
    except Exception as e:
        Printer.error(f"Failed to load configuration: {e}")
        raise

def create_vima_interface(configs: Dict[str, Any]):
    """
    Create VIMA interface using the VIMA manager.
    
    Args:
        configs: Configuration dictionary
        
    Returns:
        VIMAManager instance
    """
    Printer.header("Creating VIMA Interface")
    
    try:
        vima_config = configs["vima"]
        vima_manager = VIMAManager(vima_config)
        
        if not vima_manager.is_available():
            Printer.warning("VIMA not available - some functionality may be limited")
            return vima_manager
        
        # Create the VIMA interface
        vima_interface = vima_manager.create_vima_interface()
        Printer.success("VIMA interface created successfully")
        
        return vima_manager
        
    except Exception as e:
        Printer.error(f"Failed to create VIMA interface: {e}")
        raise

def create_graph_pipeline(configs: Dict[str, Any]):
    """
    Create the graph pipeline using the graph manager.
    
    Args:
        configs: Configuration dictionary
        
    Returns:
        GraphManager instance
    """
    Printer.header("Creating Graph Pipeline")
    
    try:
        # Extract configurations
        grounder_config = configs["grounder"]
        segmentor_config = configs["segmentor"]
        actor_config = {"mode": "pipeline"}  # Default actor config
        
        # Create graph manager
        graph_manager = GraphManager(
            grounder_config=grounder_config,
            segmentor_config=segmentor_config,
            actor_config=actor_config
        )
        
        Printer.success("Graph pipeline created successfully")
        return graph_manager
        
    except Exception as e:
        Printer.error(f"Failed to create graph pipeline: {e}")
        raise

def run_pipeline(configs: Dict[str, Any], vima_manager: VIMAManager, graph_manager: GraphManager) -> bool:
    """
    Run the robotic manipulation pipeline.
    
    Args:
        configs: Configuration dictionary
        vima_manager: VIMA manager instance
        graph_manager: Graph manager instance
        
    Returns:
        True if successful, False otherwise
    """
    Printer.header("Running Pipeline")
    
    try:
        # Get VIMA interface
        vima_interface = vima_manager.get_vima_interface()
        if not vima_interface:
            Printer.error("VIMA interface not available")
            return False
        
        # Check if VIMA environment is properly initialized
        if not hasattr(vima_interface, 'env') or vima_interface.env is None:
            Printer.error("VIMA environment not initialized - cannot proceed with pipeline")
            Printer.info("Please install VIMABench dependencies to use VIMA functionality")
            return False
        
        # Get initial observation from VIMA
        Printer.info("📡 Getting initial observation from VIMA environment...")
        initial_observation = vima_interface.get_observation()
        Printer.success(f"Observation retrieved: RGB image size {initial_observation.rgb_image.size}")
        
        # Get task prompt and assets from VIMA
        Printer.info("📝 Getting task prompt and assets...")
        vima_prompt, vima_assets = vima_interface.get_prompt_and_assets()
        Printer.success(f"Task prompt: {vima_prompt[:100]}...")
        Printer.success(f"Assets available: {list(vima_assets.keys())}")
        
        # Use VIMA prompt if available, otherwise fall back to default
        default_prompt = configs["pipeline"].task.default_prompt
        is_default_vima = vima_prompt == "Default robotic manipulation task"
        actual_prompt = vima_prompt if vima_prompt and not is_default_vima else default_prompt
        
        Printer.info(f"📝 Using prompt: {actual_prompt[:100]}...")
        
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
            "depth_image": initial_observation.depth_image,
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
        
        # Log pipeline start
        logger = graph_manager.get_logger()
        logger.log("Pipeline execution started", {
            "config": safe_serialize_for_logging(configs),
            "initial_prompt": actual_prompt,
            "vima_assets": list(vima_assets.keys())
        })
        
        Printer.info("🔄 Executing multi-agent pipeline...")
        Printer.info("   Nodes: decomposer → perceptor → grounder → segmentor → projector → thinker → actor → reflector")
        
        # Run the graph
        app = graph_manager.get_app()
        step_count = 0
        
        for step in app.stream(initial_state, output_keys=[]):
            step_count += 1
            node, state = next(iter(step.items()))
            
            Printer.step(f"{node} completed", step_count)
            
            # Add small delay for better visualization
            time.sleep(0.1)
        
        Printer.success(f"Pipeline completed successfully in {step_count} steps!")
        logger.log("Pipeline execution completed", {"total_steps": step_count})
        logger.flush()
        
        return True
        
    except Exception as e:
        Printer.error(f"Pipeline execution failed: {e}")
        logger = graph_manager.get_logger()
        logger.log("Pipeline execution failed", {"error": str(e)})
        traceback.print_exc()
        return False

def main():
    """Main entry point for the robotic manipulation pipeline."""
    Printer.header("Robotic Manipulation Pipeline with VIMA Integration")
    
    vima_manager = None
    graph_manager = None
    
    try:
        # Load configuration
        configs = load_configuration()
        
        # Create VIMA interface
        vima_manager = create_vima_interface(configs)
        
        # Create graph pipeline
        graph_manager = create_graph_pipeline(configs)
        
        # Run the pipeline
        success = run_pipeline(configs, vima_manager, graph_manager)
        
        if success:
            Printer.success("🎉 Pipeline execution completed successfully!")
        else:
            Printer.error("❌ Pipeline execution failed!")
            
    except KeyboardInterrupt:
        Printer.warning("⚠️ Pipeline interrupted by user")
        if graph_manager:
            graph_manager.get_logger().log("Pipeline interrupted by user")
    except Exception as e:
        Printer.error(f"❌ Unexpected error: {e}")
        if graph_manager:
            graph_manager.get_logger().log("Unexpected error", {"error": str(e)})
        traceback.print_exc()
    finally:
        # Cleanup
        if vima_manager:
            Printer.info("🧹 Cleaning up VIMA interface...")
            vima_manager.close_vima_interface()
        
        if graph_manager:
            Printer.info("🧹 Flushing logs...")
            try:
                graph_manager.get_logger().flush()
                Printer.success("Logs flushed successfully")
            except Exception as e:
                Printer.warning(f"Warning: Error flushing logs: {e}")
    
    Printer.info("👋 Pipeline execution finished")

if __name__ == "__main__":
    main()
