# VIMA Configuration and Setup

This directory contains the VIMA (Visual Manipulation) configuration and setup system, completely separated from the main graph setup.

## Structure

```
src/config/vima/
├── __init__.py              # Package initialization
├── vima_config.yaml         # VIMA configuration file
├── vima_setup.py            # VIMA setup and management module
└── README.md                # This file
```

## Components

### 1. VIMA Configuration (`vima_config.yaml`)

The main configuration file that controls all aspects of the VIMA environment:

- **Environment Configuration**: Task settings, display options, camera settings
- **Interface Configuration**: Observation modalities, action settings, error handling
- **Performance Settings**: Caching, retry logic, logging options

### 2. VIMA Setup Module (`vima_setup.py`)

A Python module that handles:

- **Configuration Loading**: YAML file parsing with fallback defaults
- **Interface Creation**: VIMAInterface instantiation with proper configuration
- **Environment Management**: Task setup, observation retrieval, action execution
- **Error Handling**: Robust error handling with informative messages

## Usage

### Basic Usage

```python
from config.vima.vima_setup import VIMASetup

# Create VIMA setup
setup = VIMASetup()

# Create VIMA interface
vima_interface = setup.create_vima_interface()

# Set up task
setup.setup_task()

# Get observation
observation = setup.get_observation()
```

### With Custom Configuration

```python
from config.vima.vima_setup import VIMASetup

# Create VIMA setup with custom config path
setup = VIMASetup("path/to/custom_vima_config.yaml")

# Create interface with overrides
vima_interface = setup.create_vima_interface(
    task_name="custom_task",
    display_debug_window=False
)
```

### Convenience Function

```python
from config.vima.vima_setup import create_vima_interface_from_config

# Create interface directly from config
vima_interface = create_vima_interface_from_config(
    "path/to/config.yaml",
    task_name="custom_task"
)
```

## Configuration Options

### Environment Settings

- `task_name`: VIMA task to use (default: "instruction_following/visual_manipulation")
- `display_debug_window`: Enable/disable PyBullet GUI (default: false)
- `render_mode`: Rendering mode (default: "rgb_array")
- `camera_config`: Camera parameters (width, height, FOV)

### Interface Settings

- `observation_modalities`: Supported observation types (default: ["rgb"])
- `include_prompt_assets`: Include prompt assets in observations
- `action_space`: Action space type (default: "discrete")
- `max_retries`: Maximum retry attempts for operations
- `fallback_on_error`: Enable fallback behavior on errors

## Example Configurations

### Debug Configuration
```yaml
vima:
  env:
    display_debug_window: false
    render_mode: "rgb_array"
  interface:
    observation_modalities: ["rgb"]
    log_observations: true
    log_actions: true
```

### Production Configuration
```yaml
vima:
  env:
    task_name: "instruction_following/visual_manipulation"
    display_debug_window: false
    render_mode: "rgb_array"
    camera_config:
      width: 512
      height: 256
      fov: 90
  interface:
    observation_modalities: ["rgb"]
    include_prompt_assets: true
    include_metadata: true
    action_space: "discrete"
    max_action_sequence: 20
    max_retries: 5
    cache_observations: true
    cache_size: 200
```

## Benefits of Separation

1. **Clean Architecture**: VIMA configuration is completely separate from graph logic
2. **Modularity**: Can be used independently of the main pipeline
3. **Configuration Management**: Centralized VIMA configuration in one place
4. **Reusability**: Can be imported and used by other modules
5. **Maintainability**: Easier to modify VIMA settings without affecting graph setup

## Testing

Run the test script to verify the VIMA setup:

```bash
cd src
python test_vima_setup.py
```

This will test:
- Configuration loading
- VIMA setup creation
- Interface instantiation (if VIMABench is available)
- Task setup and observation retrieval

## Error Handling

The VIMA setup module includes comprehensive error handling:

- **Import Errors**: Graceful handling when VIMABench is not available
- **Configuration Errors**: Fallback to default configuration
- **Runtime Errors**: Informative error messages and recovery options
- **Validation**: Configuration parameter validation and sanitization

## Integration with Main Pipeline

The VIMA setup is integrated into the main pipeline through `main.py`:

```python
# Create VIMA interface using setup module
vima_setup = VIMASetup()
vima_interface = vima_setup.create_vima_interface(
    task_name=config.get("vima", {}).get("task_name"),
    display_debug_window=config.get("vima", {}).get("display_debug_window", False)
)
```

This provides a clean separation of concerns while maintaining full functionality.
