#!/usr/bin/env python3
"""
Unified Printer Utility
======================

This module provides a single, consistent printing interface with colors and emojis
for better readability across the entire codebase.
"""

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class Printer:
    """Unified printer with consistent styling"""
    
    @staticmethod
    def colored(message: str, color: str = Colors.WHITE, emoji: str = ""):
        """Print a colored message with optional emoji"""
        print(f"{color}{emoji} {message}{Colors.END}")
    
    @staticmethod
    def success(message: str):
        """Print a success message in green"""
        Printer.colored(message, Colors.GREEN, "✅")
    
    @staticmethod
    def warning(message: str):
        """Print a warning message in yellow"""
        Printer.colored(message, Colors.YELLOW, "⚠️")
    
    @staticmethod
    def error(message: str):
        """Print an error message in red"""
        Printer.colored(message, Colors.RED, "❌")
    
    @staticmethod
    def info(message: str):
        """Print an info message in blue"""
        Printer.colored(message, Colors.BLUE, "ℹ️")
    
    @staticmethod
    def debug(message: str):
        """Print a debug message in cyan"""
        Printer.colored(message, Colors.CYAN, "🔧")
    
    @staticmethod
    def header(message: str):
        """Print a header message"""
        Printer.colored(f"\n{message}", Colors.BOLD + Colors.CYAN, "🚀")
        Printer.colored("=" * len(message), Colors.CYAN)
    
    @staticmethod
    def step(message: str, step_num: int = None):
        """Print a step message"""
        if step_num:
            Printer.colored(f"Step {step_num}: {message}", Colors.MAGENTA, "🔄")
        else:
            Printer.colored(message, Colors.MAGENTA, "🔄")
    
    @staticmethod
    def config(key: str, value: str):
        """Print a configuration key-value pair"""
        Printer.colored(f"  - {key}: {value}", Colors.CYAN, "⚙️")

# Convenience functions for backward compatibility
def print_colored(message: str, color: str = Colors.WHITE, emoji: str = ""):
    """Print a colored message with optional emoji"""
    Printer.colored(message, color, emoji)

def print_success(message: str):
    """Print a success message in green"""
    Printer.success(message)

def print_warning(message: str):
    """Print a warning message in yellow"""
    Printer.warning(message)

def print_error(message: str):
    """Print an error message in red"""
    Printer.error(message)

def print_info(message: str):
    """Print an info message in blue"""
    Printer.info(message)

def print_debug(message: str):
    """Print a debug message in cyan"""
    Printer.debug(message)

def print_header(message: str):
    """Print a header message"""
    Printer.header(message)
