# VIMA GUI Setup and Usage Guide
==================================

This guide will help you set up and run the VIMA environment with real-time GUI visualization.

## Quick Start (3 minutes)

### 1. Install Dependencies
```bash
# Install Hydra (optional, for advanced configuration)
pip install hydra-core omegaconf

# Install VIMA Bench (required)
# Download and install from: https://github.com/vimalabs/VIMABench
cd VIMABench
pip install -e .
```

### 2. Run the Demo
```bash
cd src

# Choose your demo:
python run_basic_demo.py    # Interactive GUI demo
python run_with_hydra.py    # Hydra-configured demo
```

### 3. What You'll See
- **Separate GUI Window**: Real-time visualization of the robot
- **Console Output**: Detailed logs of actions and results
- **Smooth Animation**: Robot movements with configurable timing

## Detailed Setup

### Prerequisites

1. **Python 3.8+**
2. **VIMABench** (robotics environment)
3. **PyBullet** (physics simulation)
4. **Hydra** (optional, for configuration management)

### Installation Steps

#### Step 1: Install VIMABench
```bash
# Clone the repository
git clone https://github.com/vimalabs/VIMABench.git
cd VIMABench

# Install in development mode
pip install -e .
```

#### Step 2: Install Additional Dependencies
```bash
pip install hydra-core omegaconf pillow numpy pyyaml
```

#### Step 3: Verify Installation
```bash
python -c "from vima_bench import make; print('VIMA installed successfully!')"
```

## Running the Demos

### Basic GUI Demo
```bash
cd src
python run_basic_demo.py
```

**Options:**
1. **Automated Demo**: Watch pre-programmed actions
2. **Interactive Demo**: Control the robot yourself
3. **Quick Test**: Basic functionality test

### Hydra Configuration Demo
```bash
cd src
python run_with_hydra.py
```

**Configuration Options:**
```bash
# Use different config files
python run_with_hydra.py --config-dir=config --config-name=graph_config

# Override specific settings
python run_with_hydra.py vima.debug=true vima.action_delay=1.0
```

### Integration Test
```bash
cd src
python test_vima_integration.py
```

## Configuration Guide

### YAML Configuration
Edit `src/config/graph_config.yaml`:

```yaml
vima:
  display_debug_window: true    # Enable GUI window
  gui_delay: 0.1               # Delay between GUI updates
  action_delay: 0.5            # Delay between actions
  enable_action_logging: true  # Log all actions
```

### Python Configuration
```python
config = {
    "display_debug_window": True,
    "gui_delay": 0.1,
    "action_delay": 0.5,
    "enable_action_logging": True
}

vima = VIMAInterface(config=config)
```

## GUI Features

### Real-Time Visualization
- **Robot Movement**: See the robot arm move in real-time
- **Object Manipulation**: Watch objects being picked and placed
- **Environment State**: View the complete scene

### Interactive Controls
- **Mouse**: Rotate and zoom the view
- **Keyboard**: Control robot (in interactive mode)
- **Window**: Resize and reposition as needed

### Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `display_debug_window` | Show GUI window | `true` |
| `gui_delay` | Delay between GUI updates (seconds) | `0.1` |
| `action_delay` | Delay between actions (seconds) | `0.5` |
| `show_gui` | Enable GUI visualization | `true` |
| `hide_arm_rgb` | Hide robot arm in RGB observations | `false` |

## Troubleshooting

### Common Issues

#### 1. GUI Window Not Appearing
```bash
# Ensure GUI is enabled in config
config = {"display_debug_window": True, "show_gui": True}
```

#### 2. Slow Performance
```python
# Increase delays for smoother visualization
config = {
    "gui_delay": 0.2,
    "action_delay": 1.0
}
```

#### 3. Import Errors
```bash
# Install missing dependencies
pip install hydra-core omegaconf

# Verify VIMA installation
python -c "import vima_bench; print('OK')"
```

#### 4. Memory Issues
```python
# Enable cleanup
vima.close_environment()

# Or use context manager
with VIMAInterface(config=config) as vima:
    # Your code here
    pass
```

## Advanced Usage

### Custom Action Sequences
```python
from vima_interface import VIMAAction
import numpy as np

# Define custom actions
actions = [
    VIMAAction(
        pose0_position=np.array([0.5, 0.0, 0.1]),
        pose0_rotation=np.array([0.0, 0.0, 0.0]),
        pose1_position=np.array([0.5, 0.2, 0.1]),
        pose1_rotation=np.array([0.0, 0.0, 0.0]),
    )
]

# Execute with GUI visualization
success, results = vima.execute_action_sequence(actions)
```

### Configuration Overrides
```python
# Override specific settings at runtime
config = {
    "display_debug_window": True,
    "action_delay": 2.0,  # Slower for detailed observation
    "debug": True         # Enable debug logging
}

vima = VIMAInterface(config=config)
```

## Performance Tips

1. **Adjust GUI Delays**: Increase `gui_delay` for smoother animation
2. **Disable Unneeded Logging**: Set `enable_observation_logging: false`
3. **Use Appropriate Action Delays**: Balance speed vs. visibility
4. **Close Environments**: Always call `close_environment()` when done

## Next Steps

1. **Run the Basic Demo**: Get familiar with the GUI
2. **Try Interactive Mode**: Learn the controls
3. **Customize Configuration**: Adjust settings for your needs
4. **Integrate with Pipeline**: Use in your robotic manipulation pipeline

Happy robot programming! 🤖✨
