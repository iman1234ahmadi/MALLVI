#!/usr/bin/env python3
"""
YAML Handler
============

This module handles YAML file operations with proper fallbacks and default generation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from utils.printer import Printer

# Try to import yaml, fallback to None if not available
try:
    import yaml
    YAML_AVAILABLE = True
    Printer.debug("PyYAML available - using YAML features")
except ImportError:
    YAML_AVAILABLE = False
    Printer.warning("PyYAML not available - YAML functionality will be limited")

class YAMLHandler:
    """Handles YAML file operations with fallbacks"""
    
    @staticmethod
    def load_yaml(file_path: Path) -> Dict[str, Any]:
        """
        Load a YAML file with proper error handling.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Dictionary containing YAML data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not available - cannot load YAML files")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return data or {}
        except yaml.YAMLError as e:
            Printer.error(f"YAML parsing error in {file_path}: {e}")
            raise
        except Exception as e:
            Printer.error(f"Error reading YAML file {file_path}: {e}")
            raise
    
    @staticmethod
    def save_yaml(data: Dict[str, Any], file_path: Path, create_dirs: bool = True):
        """
        Save data to a YAML file.
        
        Args:
            data: Data to save
            file_path: Path to save YAML file
            create_dirs: Whether to create parent directories
        """
        if not YAML_AVAILABLE:
            Printer.error("PyYAML not available - cannot save YAML files")
            return
        
        try:
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2, sort_keys=False)
            
            Printer.success(f"YAML file saved: {file_path}")
        except Exception as e:
            Printer.error(f"Error saving YAML file {file_path}: {e}")
            raise
    
    @staticmethod
    def is_yaml_available() -> bool:
        """Check if YAML functionality is available"""
        return YAML_AVAILABLE
