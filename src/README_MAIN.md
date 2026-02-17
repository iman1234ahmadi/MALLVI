# Main.py - Robotic Manipulation Pipeline with VIMA Integration

This document explains how to use the new `main.py` file that integrates the VIMA interface with Hydra configuration and runs the multi-agent robotic manipulation pipeline.

## 🚀 Overview

The `main.py` file serves as the main entry point for the robotic manipulation pipeline. It:

1. **Loads configuration** from Hydra YAML files (or uses defaults)
2. **Creates and configures** the VIMA interface based on configuration
3. **Integrates with the graph setup** to provide VIMA interface to all nodes
4. **Executes the multi-agent pipeline** for robotic manipulation tasks
5. **Provides comprehensive logging** and error handling

## 📁 File Structure

```
src/
├── main.py                          # Main entry point
├── graph_setup.py                   # LangGraph pipeline definition
├── vima_interface.py                # VIMA environment interface
├── config/
│   ├── graph_config.yaml           # Main pipeline configuration
│   ├── grounder_config.yaml        # Grounder node configuration
│   ├── segmentor_config.yaml       # Segmentor node configuration
│   └── vima/                       # VIMA configuration and setup
│       ├── __init__.py
│       ├── vima_config.yaml        # VIMA environment configuration
│       ├── vima_setup.py           # VIMA setup and management
│       └── README.md               # VIMA configuration documentation
├── nodes/                          # Pipeline nodes
├── state.py                        # Graph state definition
└── test_main_integration.py        # Integration test script
```

## 🔧 Prerequisites

1. **VIMABench**: Must be installed and accessible
2. **Python dependencies**: All required packages from requirements files
3. **Virtual environment**: Activate the mallvi virtual environment

```bash
# Activate virtual environment
source mallvi/bin/activate

# Install OmegaConf for Hydra support (optional but recommended)
pip install omegaconf
```

## 🎯 Usage

### Basic Usage

```bash
# Run the pipeline with default configuration
python src/main.py

# Run with specific configuration file
python src/main.py --config config/graph_config.yaml
```

### Configuration

The pipeline uses a modular configuration system with separate components:

#### Main Pipeline Configuration (`config/graph_config.yaml`)
```yaml
# Grounder Configuration
grounder:
  grounding_mode: "simple"
  auto_fallback_to_simple: true
  device: "cuda"

# Segmentor Configuration  
segmentor:
  backend: "box_only"
  point_mode: "auto"
  device: "cuda"
```

#### VIMA Configuration (`config/vima/vima_config.yaml`)
```yaml
vima:
  env:
    task_name: "instruction_following/visual_manipulation"
    display_debug_window: false
    render_mode: "rgb_array"
  interface:
    observation_modalities: ["rgb"]
    include_prompt_assets: true
    max_retries: 3
```

#### Pipeline Configuration
```yaml
pipeline_name: "robotic_manipulation_pipeline"
enable_logging: true
log_directory: "logs"
node_timeout: 30.0
max_retries: 3
```

#### Action Planning Configuration
```yaml
action_planning:
  max_actions_per_task: 10
  enable_orientation_control: true
  rotation_degrees_limit: 180
  position_tolerance: 0.01
```

## 🔄 Pipeline Flow

The pipeline executes the following sequence:

1. **Configuration Loading**: Load Hydra config and VIMA config separately
2. **VIMA Setup**: Initialize VIMA environment using dedicated setup module
3. **VIMA Interface Creation**: Create VIMA interface with proper configuration
4. **Initial Observation**: Get initial state from VIMA environment
5. **Multi-Agent Execution**:
   - **Decomposer**: Break down the task into subtasks
   - **Perceptor**: Process visual information

## 🏗️ VIMA Configuration Architecture

The VIMA configuration is now completely separated from the main graph setup:

### Benefits of Separation
- **Clean Architecture**: VIMA configuration is independent of graph logic
- **Modularity**: Can be used independently of the main pipeline
- **Configuration Management**: Centralized VIMA settings in dedicated files
- **Reusability**: Importable by other modules and projects
- **Maintainability**: Easier to modify VIMA settings without affecting graph setup

### VIMA Setup Module
The `config/vima/vima_setup.py` module provides:
- Configuration loading from YAML files
- VIMAInterface creation with proper configuration
- Environment management (task setup, observation retrieval)
- Comprehensive error handling and fallback options

### Usage in Main Pipeline
```python
from config.vima.vima_setup import VIMASetup

# Create VIMA setup and interface
vima_setup = VIMASetup()
vima_interface = vima_setup.create_vima_interface(
    task_name=config.get("vima", {}).get("task_name"),
    display_debug_window=config.get("vima", {}).get("display_debug_window", False)
)
```
   - **Grounder**: Ground language to visual concepts
   - **Segmentor**: Segment objects in the scene
   - **Projector**: Project 2D to 3D coordinates
   - **Thinker**: Plan actions based on task and scene
   - **Actor**: Execute actions in VIMA environment
   - **Reflector**: Evaluate task completion
6. **Cleanup**: Close VIMA interface and flush logs

## 🧪 Testing

### Integration Test

Run the integration test to verify everything works:

```bash
python src/test_main_integration.py
```

This test verifies:
- All imports work correctly
- Configuration loading functions
- VIMA interface creation
- Graph setup integration

### Manual Testing

1. **Test VIMA Interface**:
   ```bash
   python src/test_vima_debug.py
   ```

2. **Test Object Injection**:
   ```bash
   python src/test_object_injection.py
   ```

## 📊 Logging

The pipeline provides comprehensive logging:

- **Console output**: Real-time progress and status
- **File logging**: Detailed logs saved to `logs/` directory
- **Graph logging**: Step-by-step execution tracking
- **VIMA logging**: Environment and action logging

## 🚨 Error Handling

The pipeline includes robust error handling:

- **Configuration errors**: Falls back to default configuration
- **VIMA errors**: Graceful degradation with fallback observations
- **Node errors**: Logging and error reporting
- **Cleanup**: Ensures resources are properly released

## 🔧 Customization

### Adding New Configuration Options

1. Update `config/graph_config.yaml`
2. Add corresponding fields in `get_default_config()`
3. Use the configuration in relevant functions

### Modifying Pipeline Behavior

1. Update node logic in `src/nodes/`
2. Modify graph structure in `graph_setup.py`
3. Adjust state management in `state.py`

### Adding New VIMA Tasks

1. Update task configuration in YAML files
2. Ensure VIMA environment supports the task
3. Test with the integration test

## 📝 Example Output

```
🚀 Robotic Manipulation Pipeline with VIMA Integration
============================================================
📄 Loading configuration from: config/graph_config.yaml
✅ Configuration loaded successfully

🔧 Creating VIMA interface with configuration...
   Task: instruction_following/visual_manipulation
   Modalities: ['rgb']
   GUI: Enabled
✅ VIMA interface created successfully

🤖 Starting Robotic Manipulation Pipeline with VIMA Integration
======================================================================
📡 Getting initial observation from VIMA environment...
✅ Observation retrieved: RGB image size (640, 480)
📝 Getting task prompt and assets...
✅ Task prompt: Put the cross (red and blue stripe) into the container (green and blue polka dot).
✅ Assets available: ['base_obj', 'dragged_obj_1']

🔄 Executing multi-agent pipeline...
   Nodes: decomposer → perceptor → grounder → segmentor → projector → thinker → actor → reflector
   ✅ Step 1: decomposer completed
   ✅ Step 2: perceptor completed
   ✅ Step 3: grounder completed
   ✅ Step 4: segmentor completed
   ✅ Step 5: projector completed
   ✅ Step 6: thinker completed
   ✅ Step 7: actor completed
   ✅ Step 8: reflector completed

✅ Pipeline completed successfully in 8 steps!

🎉 Pipeline execution completed successfully!

🧹 Cleaning up VIMA interface...
✅ VIMA interface closed successfully
✅ Logs flushed successfully

👋 Pipeline execution finished
```

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **VIMA Errors**: Check VIMABench installation and configuration
3. **Configuration Errors**: Verify YAML syntax and file paths
4. **GUI Issues**: Ensure display settings are correct for your system

### Debug Mode

Enable debug mode in configuration:

```yaml
vima:
  debug: true
  enable_action_logging: true
  enable_observation_logging: true
```

### Log Analysis

Check log files in `logs/` directory for detailed error information and execution traces.

## 🔮 Future Enhancements

- **Multi-task support**: Handle multiple VIMA tasks in sequence
- **Real-time monitoring**: Web-based dashboard for pipeline monitoring
- **Advanced error recovery**: Automatic retry and recovery mechanisms
- **Performance optimization**: Parallel execution and caching
- **Extended VIMA support**: Additional VIMA tasks and modalities

## 📚 Related Documentation

- [VIMA Interface Documentation](VIMA_INTERFACE_README.md)
- [Graph Setup Documentation](README.md)
- [Node Implementation Details](nodes/README.md)
- [Configuration Reference](config/README.md)
