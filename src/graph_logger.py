import json
import os
import atexit
from datetime import datetime
from typing import Dict, Optional
from numpy import ndarray
from PIL.Image import Image

class GraphLogger:
    def __init__(self, folder: str = "logs", log_file: str = "graph_execution.log", console_log: bool = True):
        """
        Initialize GraphLogger with folder and log file parameters.

        :param folder: Directory where log files will be stored
        :param log_file: Name of the log file
        :param console_log: Whether to print logs to console
        """
        self.folder = folder
        self.log_file = log_file
        self.console_log = console_log
        self._log_buffer = []
        self._log_file_handle = None  # Persistent file handle

        # Create log directory if it doesn't exist
        self._ensure_log_dir_exists()

        # Register safe_close at exit
        atexit.register(self.safe_close)

    def _ensure_log_dir_exists(self):
        """Create log directory if it doesn't exist"""
        try:
            os.makedirs(self.folder, exist_ok=True)
            if self.console_log:
                print(f"Log directory ensured: {os.path.abspath(self.folder)}")
        except OSError as e:
            try:
                print(f"Error creating log directory {self.folder}: {e}")
            except Exception:
                # Avoid errors during shutdown
                pass

    def _get_log_file_handle(self):
        """Get or create a persistent file handle with unique filename"""
        if self._log_file_handle is None or self._log_file_handle.closed:
            # Create unique filename to prevent overwrites
            base, ext = os.path.splitext(self.log_file)
            timestamp = datetime.now().strftime("%H%M%S%f")
            unique_file = f"{base}_{timestamp}{ext}"
            full_path = os.path.join(self.folder, unique_file)

            try:
                self._log_file_handle = open(full_path, "a", encoding="utf-8")
                if self.console_log:
                    print(f"Log file created: {full_path}")
            except OSError as e:
                if self.console_log:
                    print(f"Error opening log file: {e}")
                # Fallback to stdout
                self._log_file_handle = None

        return self._log_file_handle

    def log(self, message: str, data: Optional[Dict] = None):
        """Log message with optional data payload"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "data": data or {}
        }

        # Add to buffer (don't write to file immediately to avoid duplication)
        self._log_buffer.append(entry)

        # Write to console if enabled (immediate)
        if self.console_log:
            try:
                print(f"[{entry['timestamp']}] {message}")
                if data:
                    safe_data = self._safe_serialize_for_json(data)
                    print(json.dumps(safe_data, indent=2))
            except Exception as e:
                # Avoid errors during shutdown
                print(f"Console logging error: {e}")
                pass

        # Flush buffer to file if it gets large enough
        if len(self._log_buffer) >= 10:
            self.safe_flush()

    def _safe_serialize_for_json(self, obj):
        """Safely serialize object for JSON, handling numpy arrays and other non-serializable types"""
        if isinstance(obj, ndarray):
            return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
        elif isinstance(obj, Image):
            return f"Image(size={obj.size}, mode={obj.mode})"
        elif isinstance(obj, dict):
            return {key: self._safe_serialize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._safe_serialize_for_json(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return f"{type(obj).__name__}(id={id(obj)})"
        else:
            return obj

    def _write_entry_to_file(self, entry: dict):
        """Immediately write a single entry to the log file"""
        try:
            f = self._get_log_file_handle()
            if f:
                # Safely serialize the entry before writing
                safe_entry = self._safe_serialize_for_json(entry)
                f.write(json.dumps(safe_entry) + "\n")
                f.flush()  # Force write to disk
        except Exception as e:
            if self.console_log:
                try:
                    print(f"Log write error: {e}")
                except Exception:
                    pass

    def safe_flush(self):
        """Ensure all buffered entries are written to file"""
        if not self._log_buffer:
            return

        try:
            # Write all buffered entries to file
            for entry in self._log_buffer:
                self._write_entry_to_file(entry)
        except Exception as e:
            if self.console_log:
                try:
                    print(f"Flush error: {e}")
                except Exception:
                    pass
        finally:
            # Clear buffer even if some writes failed
            self._log_buffer = []

    def safe_close(self):
        """Properly close the log file"""
        try:
            self.safe_flush()
            if self._log_file_handle and not self._log_file_handle.closed:
                self._log_file_handle.close()
        except Exception as e:
            if self.console_log:
                try:
                    print(f"Close error: {e}")
                except Exception:
                    pass
        finally:
            self._log_file_handle = None

    def flush(self):
        """Public flush method"""
        self.safe_flush()

    def summarize_state(self, state: Dict) -> Dict:
        """Create a human-readable summary of the state"""
        summary = {}
        for key, value in state.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = value
            elif isinstance(value, list):
                if len(value) > 3:
                    summary[key] = f"List[{len(value)} items]"
                else:
                    summary[key] = [self.summarize_state(item) if isinstance(item, dict) else item for item in value]
            elif isinstance(value, dict):
                if len(value) > 3:
                    summary[key] = f"Dict[{len(value)} keys]"
                else:
                    summary[key] = {k: self.summarize_state(v) if isinstance(v, dict) else v for k, v in value.items()}
            elif isinstance(value, ndarray):
                summary[key] = f"ndarray{value.shape} ({value.dtype})"
            elif isinstance(value, Image):
                summary[key] = f"Image{value.size} ({value.mode})"
            else:
                try:
                    # Try to get a simple string representation
                    summary[key] = str(value)
                except Exception:
                    summary[key] = f"<{type(value).__name__} object>"
        return summary

    def log_state(self, message: str, state: Dict):
        """Log message with state summary"""
        self.log(message, {"state": self.summarize_state(state)})

    def log_state_diff(self, before: Dict, after: Dict):
        """Log state differences between before and after"""
        diff = {}
        for key in set(before.keys()) | set(after.keys()):
            if key not in before:
                diff[key] = {"action": "added", "value": self.summarize_state({key: after[key]})[key]}
            elif key not in after:
                diff[key] = {"action": "removed", "prev_value": self.summarize_state({key: before[key]})[key]}
            elif before[key] != after[key]:
                diff[key] = {
                    "action": "changed",
                    "prev_value": self.summarize_state({key: before[key]})[key],
                    "new_value": self.summarize_state({key: after[key]})[key]
                }
        return diff

    def node_callback(self, node_name: str):
        """Create callback function for a specific node"""
        def callback(state_before: Dict, state_after: Dict):
            self.log(
                f"Node completed: {node_name}",
                {
                    "node": node_name,
                    "state_before": self.summarize_state(state_before),
                    "state_after": self.summarize_state(state_after),
                    "state_diff": self.log_state_diff(state_before, state_after)
                }
            )
        return callback

    def __del__(self):
        """Final close during destruction"""
        self.safe_close()
