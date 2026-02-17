# VIMA Interface Documentation

## Overview

This document describes the VIMA interface that has been integrated into the robotic manipulation pipeline. The interface provides a comprehensive way to interact with VIMABench environments, retrieve task prompts and assets, get observations, and execute actions.

## Architecture

The VIMA interface consists of several key components:

1. **VIMAInterface Class**: Main interface for VIMABench interaction
2. **VIMAObservation Dataclass**: Container for environment observations
3. **VIMAAction Dataclass**: Container for robot actions
4. **VIMATask Dataclass**: Container for task information
5. **Integration with Graph Setup**: Functions to retrieve VIMA data for the pipeline

## Files Modified/Created

### New Files
- `src/vima_interface.py` - Main VIMA interface implementation
- `src/test_vima_integration.py` - Test script for the interface

### Modified Files
- `src/nodes/actor.py` - Updated to use the new VIMA interface
- `src/graph_setup.py` - Added VIMA integration functions
- `src/state.py` - Added VIMA-specific state fields

## VIMAInterface Class

### Initialization
```python
from vima_interface import VIMAInterface

vima = VIMAInterface(
    modalities=["rgb", "depth"],      # Observation modalities
    debug=False,                      # Enable debug mode
    display_debug_window=False,       # Show debug window
    hide_arm_rgb=False               # Hide robot arm in RGB
)
```

### Key Methods

#### 1. Setup Task
```python
success = vima.setup_task("pick_place")  # Set up specific task
```

#### 2. Get Prompt and Assets
```python
prompt, assets = vima.get_prompt_and_assets()
# Returns: (task_prompt_string, assets_dict)
```

#### 3. Get Observation
```python
observation = vima.get_observation()
# Returns: VIMAObservation with RGB, depth, camera params, etc.
```

#### 4. Execute Action
```python
from vima_interface import VIMAAction
import numpy as np

action = VIMAAction(
    pose0_position=np.array([0.5, 0.0, 0.1]),    # Pick position
    pose0_rotation=np.array([0.0, 0.0, 0.0]),    # Pick rotation
    pose1_position=np.array([0.5, 0.2, 0.1]),    # Place position
    pose1_rotation=np.array([0.0, 0.0, 0.0]),    # Place rotation
)

success, result = vima.execute_action(action)
```

#### 5. Execute Action Sequence
```python
actions = [action1, action2, action3]
success, results = vima.execute_action_sequence(actions)
```

#### 6. Environment Management
```python
vima.reset_environment()    # Reset to initial state
vima.close_environment()    # Clean up resources
```

## Data Structures

### VIMAObservation
```python
@dataclass
class VIMAObservation:
    rgb_image: Image.Image              # RGB observation
    depth_image: np.ndarray             # Depth map
    camera_matrix: np.ndarray           # 3x3 camera matrix
    rotation_matrix: np.ndarray         # 3x3 rotation matrix
    translation_vector: np.ndarray      # 3x1 translation vector
    task_prompt: str                    # Current task prompt
    prompt_assets: Dict[str, Any]       # Task-related assets
    metadata: Dict[str, Any]            # Additional metadata
```

### VIMAAction
```python
@dataclass
class VIMAAction:
    pose0_position: np.ndarray          # Pick position [x, y, z]
    pose0_rotation: np.ndarray          # Pick rotation [roll, pitch, yaw]
    pose1_position: np.ndarray          # Place position [x, y, z]
    pose1_rotation: np.ndarray          # Place rotation [roll, pitch, yaw]
    action_metadata: Optional[Dict] = None  # Additional action info
```

## Graph Integration

The VIMA interface is integrated into the graph setup with helper functions:

### Helper Functions in graph_setup.py

```python
# Get observation from VIMA
observation = get_vima_observation(task_name="pick_place")

# Get prompt and assets
prompt, assets = get_vima_prompt_and_assets(task_name="pick_place")
```

### State Fields Added

New fields in GraphState:
- `vima_prompt_assets`: Assets from VIMA task
- `vima_metadata`: Metadata from VIMA environment
- `vima_interface`: Shared VIMA interface instance (passed to all nodes)

### Shared Environment Architecture

**Important**: The VIMA interface uses a shared environment architecture:

1. **Single Environment**: One VIMA environment is created in `graph_setup.py`
2. **Shared Instance**: The same interface instance is passed to all nodes through the state
3. **Consistent State**: All nodes (perceptor, actor, etc.) work with the same environment
4. **Centralized Cleanup**: Environment cleanup is handled by `graph_setup.py`

This ensures:
- The actor executes actions in the same environment used for perception
- No duplicate environments are created
- Consistent environment state across the entire pipeline
- Proper resource management

## Usage Examples

### Basic Usage
```python
from vima_interface import VIMAInterface

# Initialize
vima = VIMAInterface()

# Get current task info
prompt, assets = vima.get_prompt_and_assets()
print(f"Task: {prompt}")

# Get observation
obs = vima.get_observation()
print(f"Image size: {obs.rgb_image.size}")

# Execute an action
action = VIMAAction(
    pose0_position=np.array([0.5, 0.0, 0.1]),
    pose0_rotation=np.array([0.0, 0.0, 0.0]),
    pose1_position=np.array([0.5, 0.2, 0.1]),
    pose1_rotation=np.array([0.0, 0.0, 0.0]),
)

success, result = vima.execute_action(action)
print(f"Action success: {success}, Reward: {result.get('reward')}")

# Cleanup
vima.close_environment()
```

### Pipeline Integration
```python
# In graph_setup.py - Single shared environment
from vima_interface import VIMAInterface

vima_interface = VIMAInterface()  # One instance for the entire pipeline
observation = get_vima_observation()
prompt, assets = get_vima_prompt_and_assets()

# Use in initial state - pass the same interface to all nodes
initial_state = {
    "original_prompt": prompt,
    "image": observation.rgb_image,
    "depth_image": observation.depth_image,
    # ... other fields
    "vima_prompt_assets": assets,
    "vima_metadata": observation.metadata,
    "vima_interface": vima_interface  # Shared across all nodes
}

# In actor_node.py - Use the shared interface
def actor_node(state: GraphState) -> dict:
    vima_interface = state.get('vima_interface')  # Get shared instance

    # Execute actions in the same environment used for perception
    success, result = vima_interface.execute_action(action)

    # Don't close the interface - graph_setup.py handles cleanup
    return {"actor_output": output}
```

## GUI and Real-Time Visualization

The VIMA interface now supports real-time GUI visualization of the robotic environment:

### Quick Start with GUI
```bash
# Run basic demo with real-time visualization
python run_basic_demo.py

# Choose from:
# 1. Automated demo (recommended)
# 2. Interactive demo (control the robot yourself)
# 3. Quick test
```

### Hydra Configuration
```bash
# Install Hydra
pip install hydra-core omegaconf

# Run with Hydra configuration
python run_with_hydra.py

# Or specify custom config
python run_with_hydra.py --config-dir=config --config-name=graph_config
```

### Configuration Features
- **Real-time GUI**: See the robot move in a separate window
- **Configurable delays**: Adjust timing for smooth visualization
- **Action logging**: Detailed logs of all actions and results
- **Flexible configuration**: Use YAML files or Python dictionaries

### Demo Scripts

1. **`run_basic_demo.py`**: Basic GUI demo without Hydra
2. **`run_with_hydra.py`**: Full Hydra configuration demo
3. **`test_vima_integration.py`**: Comprehensive integration tests

## Testing

Run the integration test:
```bash
cd src
python test_vima_integration.py
```

This will test:
- VIMA interface initialization
- Observation retrieval
- Prompt and assets retrieval
- Action execution
- Task setup functionality

## Error Handling

The interface includes comprehensive error handling:

1. **Import Errors**: Graceful fallback if VIMABench is not available
2. **Environment Errors**: Proper cleanup on failures
3. **Action Errors**: Detailed logging of failed actions
4. **Observation Errors**: Fallback to dummy data when needed

## Best Practices

1. **Always cleanup**: Call `close_environment()` when done
2. **Use context managers**: For task-specific operations
3. **Check return values**: Verify success of operations
4. **Handle exceptions**: Use try-catch blocks for robustness
5. **Log operations**: Enable logging for debugging

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure VIMABench is properly installed
   ```bash
   pip install vima-bench
   ```

2. **Environment Not Found**: Check VIMA task names are valid

3. **Action Failure**: Verify action coordinates are within workspace

4. **Memory Issues**: Close environments when not needed

### Debug Mode

Enable debug mode for detailed logging:
```python
vima = VIMAInterface(debug=True, display_debug_window=True)
```

## Future Enhancements

Potential improvements:
- Support for additional observation modalities
- Batch action execution
- Task-specific action validation
- Integration with more VIMA tasks
- Performance optimization for real-time applications
