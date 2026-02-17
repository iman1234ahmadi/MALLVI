#!/usr/bin/env python3
"""
Robotics Pipeline Log Visualizer v2.0
=====================================

A modern, interactive web application for analyzing robotics pipeline execution logs.

Features:
- 🎨 Modern Bootstrap UI with responsive design
- 📊 Interactive timeline visualization with Plotly
- 🔍 Detailed log inspection with syntax-highlighted JSON
- 📈 Performance metrics and statistics
- 🚀 Fast, modular architecture

Usage:
    python src/run_visualizer.py

Then navigate to: http://localhost:8050
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from visualizer import create_app


def main():
    """Main entry point."""
    print("🤖 Robotics Pipeline Log Visualizer v2.0")
    print("=" * 50)
    print("🚀 Starting application...")

    # Create and run the app
    app = create_app(debug=True)

    print("✅ Application ready!")
    print("📊 Open your browser and navigate to: http://localhost:8050")
    print("📁 Upload a .log file from the logs/ directory to get started")
    print("💡 Press Ctrl+C to stop the server")
    print()

    try:
        app.run(debug=True, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
