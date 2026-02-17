"""Data processing utilities for log parsing and event extraction."""

import json
import pandas as pd
from datetime import datetime
import re
from typing import List, Dict, Tuple, Any


class LogProcessor:
    """Processes log files and extracts events for visualization."""

    def __init__(self):
        self.logs = []
        self.events_df = None
        self.node_durations = {}
        self.node_stats = {}

    def parse_logs(self, log_lines: List[str]) -> List[Dict[str, Any]]:
        """Parse log lines into structured data."""
        logs = []
        for i, line in enumerate(log_lines):
            if not line.strip():
                continue

            try:
                # Handle trailing commas and other JSON issues
                cleaned_line = re.sub(r',\s*}$', '}', line.strip())
                cleaned_line = re.sub(r',\s*]$', ']', cleaned_line)

                log = json.loads(cleaned_line)
                log['index'] = i
                logs.append(log)

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {i}: {e}")
                print(f"Problematic line: {line[:100]}...")
                continue

        self.logs = logs
        return logs

    def create_events_dataframe(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Create events DataFrame from parsed logs."""
        events = []
        node_starts = {}
        node_durations = {}
        node_stats = {
            'total_nodes': 0,
            'successful_nodes': 0,
            'failed_nodes': 0,
            'total_duration': 0
        }

        graph_start_time = None

        for log in self.logs:
            try:
                # Parse timestamp
                ts_str = log['timestamp'].rstrip('Z')
                if '+' not in ts_str and 'Z' not in ts_str:
                    ts = datetime.fromisoformat(ts_str)
                else:
                    # Handle timezone info
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

                msg = log['message']
                data = log.get('data', {})

                # Process different event types
                if msg == "Graph execution started":
                    graph_start_time = ts
                    events.append({
                        'event': 'Graph Start',
                        'start': ts,
                        'end': ts,
                        'type': 'graph_event',
                        'node': None,
                        'details': 'Pipeline execution started',
                        'log_index': log['index'],
                        'duration': 0,
                        'message': msg
                    })

                elif msg == "Graph execution completed":
                    if graph_start_time:
                        total_duration = (ts - graph_start_time).total_seconds()
                        node_stats['total_duration'] = total_duration

                    events.append({
                        'event': 'Graph End',
                        'start': ts,
                        'end': ts,
                        'type': 'graph_event',
                        'node': None,
                        'details': f'Pipeline completed in {node_stats["total_duration"]:.2f}s',
                        'log_index': log['index'],
                        'duration': 0,
                        'message': msg
                    })

                elif "Node started:" in msg:
                    node_name = msg.split("Node started: ")[1].strip()
                    node_starts[node_name] = ts
                    node_stats['total_nodes'] += 1

                    events.append({
                        'event': f"{node_name} Start",
                        'start': ts,
                        'end': ts,
                        'type': 'node_start',
                        'node': node_name,
                        'details': f"{node_name.title()} node execution started",
                        'log_index': log['index'],
                        'duration': 0,
                        'message': msg
                    })

                elif "Node completed:" in msg:
                    node_name = msg.split("Node completed: ")[1].strip()
                    start_time = node_starts.get(node_name, ts)
                    duration = (ts - start_time).total_seconds()

                    node_durations[node_name] = duration
                    node_stats['successful_nodes'] += 1

                    events.append({
                        'event': f"{node_name} End",
                        'start': start_time,
                        'end': ts,
                        'type': 'node_end',
                        'node': node_name,
                        'details': f"{node_name.title()} completed successfully in {duration:.3f}s",
                        'log_index': log['index'],
                        'duration': duration,
                        'message': msg
                    })

                # Skip Graph step completed events as they clutter the timeline
                elif msg == "Graph step completed":
                    # These events are not useful for visualization - skip them entirely
                    continue

                elif "Task Complete:" in msg or "Verification:" in msg:
                    # Handle reflector outputs - these should be system events, not node-specific
                    events.append({
                        'event': 'Task Verification',
                        'start': ts,
                        'end': ts,
                        'type': 'system',
                        'node': None,  # Changed from 'reflector' to None to put in System Events section
                        'details': msg,
                        'log_index': log['index'],
                        'duration': 0,
                        'message': msg
                    })

                else:
                    # Generic system messages
                    events.append({
                        'event': msg[:50] + ('...' if len(msg) > 50 else ''),
                        'start': ts,
                        'end': ts,
                        'type': 'system',
                        'node': None,
                        'details': f"{msg}\n{json.dumps(data, indent=2) if data else 'No additional data'}",
                        'log_index': log['index'],
                        'duration': 0,
                        'message': msg
                    })

            except Exception as e:
                print(f"Error processing log entry: {e}")
                continue

        self.events_df = pd.DataFrame(events)
        self.node_durations = node_durations
        self.node_stats = node_stats

        return self.events_df, node_durations

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the log data."""
        if not self.logs:
            return {}

        first_log = self.logs[0] if self.logs else {}
        last_log = self.logs[-1] if self.logs else {}

        return {
            'total_logs': len(self.logs),
            'first_timestamp': first_log.get('timestamp', 'N/A'),
            'last_timestamp': last_log.get('timestamp', 'N/A'),
            'node_stats': self.node_stats,
            'unique_nodes': len(self.node_durations),
            'avg_node_duration': (sum(self.node_durations.values()) / len(self.node_durations)
                                 if self.node_durations else 0)
        }

    def get_node_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed node performance metrics."""
        performance = {}

        for node, duration in self.node_durations.items():
            # Count occurrences of this node in events
            node_events = (self.events_df[self.events_df['node'] == node]
                          if self.events_df is not None else pd.DataFrame())

            performance[node] = {
                'duration': duration,
                'executions': len(node_events[node_events['type'] == 'node_end']),
                'avg_duration': duration,  # For now, assuming single execution
                'status': 'success'  # Could be enhanced to detect failures
            }

        return performance
