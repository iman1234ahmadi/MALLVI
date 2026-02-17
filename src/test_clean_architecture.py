#!/usr/bin/env python3
"""
Test Clean Architecture
======================

This script tests the new clean architecture to ensure all modules work together
without circular dependencies and with proper fallback mechanisms.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_unified_printer():
    """Test the unified printer functionality"""
    print("🧪 Testing unified printer...")
    
    from utils.printer import Printer
    
    Printer.success("Unified printer test successful!")
    Printer.warning("This is a warning test")
    Printer.error("This is an error test")
    Printer.info("This is an info test")
    Printer.debug("This is a debug test")
    Printer.header("This is a header test")
    Printer.step("This is a step test", 1)
    Printer.config("Test Key", "Test Value")
    
    print("✅ Unified printer test passed!")

def test_yaml_handler():
    """Test the YAML handler functionality"""
    print("🧪 Testing YAML handler...")
    
    from config.yaml_handler import YAMLHandler
    
    # Test YAML availability check
    is_available = YAMLHandler.is_yaml_available()
    print(f"YAML available: {is_available}")
    
    print("✅ YAML handler test passed!")

def test_default_configs():
    """Test the default config generation"""
    print("🧪 Testing default config generation...")
    
    from config.default_configs import get_default_configs
    
    configs = get_default_configs()
    
    # Check that all expected configs are present
    expected_keys = ["pipeline", "vima", "grounder", "segmentor"]
    for key in expected_keys:
        assert key in configs, f"Missing config key: {key}"
        print(f"  ✓ {key} config present")
    
    print("✅ Default config generation test passed!")

def test_config_manager():
    """Test the unified config manager"""
    print("🧪 Testing config manager...")
    
    from config.config_manager import config_manager
    
    # Test loading all configs
    configs = config_manager.load_all_configs()
    
    # Check that all expected configs are loaded
    expected_keys = ["pipeline", "vima", "grounder", "segmentor"]
    for key in expected_keys:
        assert key in configs, f"Missing config key: {key}"
        print(f"  ✓ {key} config loaded")
    
    print("✅ Config manager test passed!")

def test_vima_manager():
    """Test the VIMA manager"""
    print("🧪 Testing VIMA manager...")
    
    from config.vima_manager import VIMAManager
    from config.config_classes import VIMAConfig
    
    # Test creating VIMA manager
    vima_config = VIMAConfig()
    vima_manager = VIMAManager(vima_config)
    
    # Test availability check
    is_available = vima_manager.is_available()
    print(f"VIMA available: {is_available}")
    
    # Test config getter
    config = vima_manager.get_config()
    print(f"VIMA config type: {type(config)}")
    
    print("✅ VIMA manager test passed!")

def test_graph_manager():
    """Test the graph manager"""
    print("🧪 Testing graph manager...")
    
    from graph_manager import GraphManager
    from config.config_classes import GrounderConfig, SegmentorConfig
    
    # Test creating graph manager
    grounder_config = GrounderConfig()
    segmentor_config = SegmentorConfig()
    actor_config = {"mode": "pipeline"}
    
    graph_manager = GraphManager(
        grounder_config=grounder_config,
        segmentor_config=segmentor_config,
        actor_config=actor_config
    )
    
    # Test getting the app
    app = graph_manager.get_app()
    print(f"Graph app type: {type(app)}")
    
    # Test getting logger
    logger = graph_manager.get_logger()
    print(f"Graph logger type: {type(logger)}")
    
    print("✅ Graph manager test passed!")

def test_graph_setup_backward_compatibility():
    """Test that graph_setup.py maintains backward compatibility"""
    print("🧪 Testing graph_setup backward compatibility...")
    
    from graph_setup import app, logger, set_grounder_config, set_segmentor_config, set_actor_config
    from config.config_classes import GrounderConfig, SegmentorConfig
    
    # Test that the expected functions exist
    assert callable(set_grounder_config), "set_grounder_config should be callable"
    assert callable(set_segmentor_config), "set_segmentor_config should be callable"
    assert callable(set_actor_config), "set_actor_config should be callable"
    
    # Test setting configs
    grounder_config = GrounderConfig()
    segmentor_config = SegmentorConfig()
    actor_config = {"mode": "pipeline"}
    
    set_grounder_config(grounder_config)
    set_segmentor_config(segmentor_config)
    set_actor_config(actor_config)
    
    # Test that app and logger are available
    assert app is not None, "app should be available"
    assert logger is not None, "logger should be available"
    
    print("✅ Graph setup backward compatibility test passed!")

def test_no_circular_dependencies():
    """Test that there are no circular dependencies"""
    print("🧪 Testing for circular dependencies...")
    
    # Test importing all modules
    try:
        from utils.printer import Printer
        from config.yaml_handler import YAMLHandler
        from config.default_configs import get_default_configs
        from config.config_manager import config_manager
        from config.vima_manager import VIMAManager
        from graph_manager import GraphManager
        from graph_setup import app, logger
        print("✅ All modules imported successfully - no circular dependencies!")
    except ImportError as e:
        print(f"❌ Import error detected: {e}")
        raise

def main():
    """Run all tests"""
    print("🚀 Starting Clean Architecture Tests")
    print("=" * 50)
    
    try:
        test_unified_printer()
        print()
        
        test_yaml_handler()
        print()
        
        test_default_configs()
        print()
        
        test_config_manager()
        print()
        
        test_vima_manager()
        print()
        
        test_graph_manager()
        print()
        
        test_graph_setup_backward_compatibility()
        print()
        
        test_no_circular_dependencies()
        print()
        
        print("🎉 All tests passed! Clean architecture is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
