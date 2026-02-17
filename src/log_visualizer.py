"""
Legacy Log Visualizer - DEPRECATED
===================================

This is the old log visualizer. Please use the new modular version instead:

    python src/run_visualizer.py

The new version features:
- 🎨 Modern Bootstrap UI with responsive design
- 📊 Interactive timeline visualization with Plotly
- 🔍 Detailed log inspection with syntax-highlighted JSON
- 📈 Performance metrics and statistics
- 🚀 Fast, modular architecture
- ✅ Fixed tab rendering issues
"""

import warnings
import sys
from pathlib import Path

warnings.warn(
    "This log_visualizer.py is deprecated. Use 'python src/run_visualizer.py' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Add the src directory to Python path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Import new visualizer
from visualizer import create_app


def main():
    """Legacy entry point - redirects to new visualizer."""
    print("⚠️  DEPRECATED: This file is deprecated!")
    print("🔄 Redirecting to new modular visualizer...")
    print("💡 In the future, use: python src/run_visualizer.py")
    print()

    app = create_app(debug=True)
    app.run(debug=True, host='0.0.0.0', port=8050)


if __name__ == '__main__':
    main()
