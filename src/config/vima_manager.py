#!/usr/bin/env python3
"""
VIMA Manager
============

This module handles VIMA-specific setup and configuration.
It provides a clean interface for creating and configuring VIMA environments.
"""

import sys
from typing import Optional

# Add VIMABench to path
sys.path.append('../VIMABench')

from utils.printer import Printer

# Try to import VIMA interface
try:
    from vima_interface import VIMAInterface
    VIMA_AVAILABLE = True
    Printer.debug("VIMA interface available")
except ImportError:
    VIMA_AVAILABLE = False
    Printer.warning("VIMA interface not available - VIMA functionality will be limited")

from .config_classes import VIMAConfig

class VIMAManager:
    """
    Manages VIMA environment setup and configuration.
    """
    
    def __init__(self, config: Optional[VIMAConfig] = None):
        """
        Initialize VIMA manager.
        
        Args:
            config: VIMA configuration object
        """
        self.config = config or VIMAConfig()
        self.vima_interface: Optional[VIMAInterface] = None
        
        Printer.debug("Initializing VIMA Manager")
        Printer.config("Task", self.config.task_name)
        Printer.config("Modalities", str(self.config.modalities))
        Printer.config("Debug", str(self.config.debug))
    
    def create_vima_interface(self, **kwargs) -> VIMAInterface:
        """
        Create and configure VIMA interface.
        
        Args:
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured VIMAInterface instance
            
        Raises:
            RuntimeError: If VIMA is not available or creation fails
        """
        if not VIMA_AVAILABLE:
            raise RuntimeError("VIMA interface not available. Please ensure VIMABench is properly installed.")
        
        Printer.info("🤖 Creating VIMA interface...")
        
        try:
            # Merge config with kwargs
            vima_config = {
                "task_name": self.config.task_name,
                "modalities": self.config.modalities,
                "debug": self.config.debug,
                "display_debug_window": self.config.display_debug_window,
                "hide_arm_rgb": self.config.hide_arm_rgb,
                "gui_delay": self.config.gui_delay,
                "action_delay": self.config.action_delay,
                "camera_config": {
                    "width": self.config.camera_width,
                    "height": self.config.camera_height,
                    "fov": self.config.camera_fov
                },
                **kwargs
            }
            
            Printer.config("Task", vima_config['task_name'])
            Printer.config("Modalities", str(vima_config['modalities']))
            Printer.config("GUI", 'Enabled' if vima_config['display_debug_window'] else 'Disabled')
            Printer.config("Hide arm", str(vima_config['hide_arm_rgb']))
            
            # Create VIMA interface
            self.vima_interface = VIMAInterface(**vima_config)
            
            # Check if environment was properly initialized
            if not hasattr(self.vima_interface, 'env') or self.vima_interface.env is None:
                Printer.warning("VIMA environment not initialized - this may be due to missing dependencies")
                Printer.info("To fix this, install VIMABench dependencies:")
                Printer.info("  1. Install gymnasium: pip install gymnasium")
                Printer.info("  2. Install VIMABench: cd VIMABench && pip install -e .")
                Printer.warning("Pipeline will continue but VIMA functionality will be limited")
            else:
                Printer.success("VIMA interface created successfully")
            
            return self.vima_interface
            
        except Exception as e:
            Printer.error(f"Failed to create VIMA interface: {e}")
            raise RuntimeError(f"VIMA interface creation failed: {e}") from e
    
    def get_vima_interface(self) -> Optional[VIMAInterface]:
        """
        Get the current VIMA interface.
        
        Returns:
            VIMAInterface instance if available, None otherwise
        """
        return self.vima_interface
    
    def close_vima_interface(self):
        """Close the VIMA interface and clean up resources"""
        if self.vima_interface:
            Printer.info("🧹 Closing VIMA interface...")
            try:
                self.vima_interface.close_environment()
                Printer.success("VIMA interface closed successfully")
            except Exception as e:
                Printer.warning(f"Warning: Error closing VIMA interface: {e}")
            finally:
                self.vima_interface = None
        else:
            Printer.debug("No VIMA interface to close")
    
    def is_available(self) -> bool:
        """
        Check if VIMA is available.
        
        Returns:
            True if VIMA is available, False otherwise
        """
        return VIMA_AVAILABLE
    
    def get_config(self) -> VIMAConfig:
        """
        Get the current VIMA configuration.
        
        Returns:
            VIMAConfig object
        """
        return self.config
    
    def update_config(self, config: VIMAConfig):
        """
        Update the VIMA configuration.
        
        Args:
            config: New VIMA configuration
        """
        self.config = config
        Printer.debug("VIMA configuration updated")
        Printer.config("Task", config.task_name)
        Printer.config("Modalities", str(config.modalities))
        Printer.config("Debug", str(config.debug))

# Convenience function for backward compatibility
def create_vima_manager(config: Optional[VIMAConfig] = None) -> VIMAManager:
    """
    Create a VIMA manager instance.
    
    Args:
        config: VIMA configuration object
        
    Returns:
        VIMAManager instance
    """
    return VIMAManager(config)
