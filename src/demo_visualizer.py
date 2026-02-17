#!/usr/bin/env python3
"""
Demo script for the Robotics Pipeline Log Visualizer
===================================================

This script demonstrates the new visualizer with a sample log file.
"""

import sys
import webbrowser
import time
from pathlib import Path
from threading import Timer

# Add the src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from visualizer import create_app


def open_browser():
    """Open browser after a short delay."""
    time.sleep(2)  # Wait for server to start
    webbrowser.open('http://localhost:8050')


def main():
    """Run the demo."""
    print("🎬 Robotics Pipeline Log Visualizer Demo")
    print("=" * 50)

    # Check if sample logs exist
    logs_dir = src_path / "logs"
    if not logs_dir.exists():
        print("⚠️  No logs directory found")
        print("📁 Create some log files first by running your pipeline")
    else:
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            print(f"📊 Found {len(log_files)} log files in logs/ directory")
            print("💡 You can upload any of these files in the web interface")
        else:
            print("⚠️  No .log files found in logs/ directory")

    print("\n🚀 Starting demo server...")
    print("📊 The visualizer will open in your browser automatically")
    print("💡 Press Ctrl+C to stop the server")
    print()

    # Create and configure app
    app = create_app(debug=False)  # Disable debug for demo

    # Open browser after delay
    Timer(2.0, open_browser).start()

    try:
        app.run(debug=False, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        print("\n👋 Demo stopped. Thanks for trying the visualizer!")
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
