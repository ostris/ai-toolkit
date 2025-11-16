"""
Log file parser for extracting warnings, errors, and training issues.
Parses output/{job_name}/log.txt to identify problems.
"""

import re
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class LogParser:
    """
    Parses training log files to extract errors, warnings, and issues.
    """

    # Patterns for different log levels and issues
    ERROR_PATTERNS = [
        (r'ERROR|Error|error:', 'error'),
        (r'RuntimeError:', 'runtime_error'),
        (r'CUDA out of memory|OOM|OutOfMemoryError', 'oom_error'),
        (r'CUDA error|cudaErrorNoDevice', 'cuda_error'),
        (r'Traceback \(most recent call last\)', 'traceback'),
        (r'Exception:|exception:', 'exception'),
    ]

    WARNING_PATTERNS = [
        (r'WARNING|Warning|warn:', 'warning'),
        (r'UserWarning:', 'user_warning'),
        (r'DeprecationWarning:', 'deprecation_warning'),
        (r'FutureWarning:', 'future_warning'),
    ]

    # Specific issue patterns
    OOM_PATTERNS = [
        r'CUDA out of memory',
        r'OutOfMemoryError',
        r'RuntimeError:.*out of memory',
        r'torch\.cuda\.OutOfMemoryError',
        r'Tried to allocate .* GiB',
    ]

    CUDA_ERROR_PATTERNS = [
        r'CUDA error:',
        r'cudaErrorNoDevice',
        r'CUDA initialization error',
        r'CUDA driver error',
    ]

    CONFIG_ISSUE_PATTERNS = [
        r'Invalid configuration',
        r'Missing required',
        r'Configuration error',
        r'KeyError:',
        r'TypeError:.*argument',
    ]

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.oom_events: List[Dict[str, Any]] = []

    def parse(self) -> Dict[str, Any]:
        """
        Parse the log file and extract all issues.

        Returns:
            Dictionary with errors, warnings, and summary statistics.
        """
        if not os.path.exists(self.log_path):
            return {
                'errors': [],
                'warnings': [],
                'oom_events': [],
                'error_count': 0,
                'warning_count': 0,
                'oom_count': 0,
            }

        self.errors = []
        self.warnings = []
        self.oom_events = []

        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            current_block = []
            in_traceback = False

            for i, line in enumerate(lines):
                # Track tracebacks
                if 'Traceback (most recent call last)' in line:
                    in_traceback = True
                    current_block = [line]
                    continue

                if in_traceback:
                    current_block.append(line)
                    # End of traceback (line starts with actual error)
                    if line.strip() and not line.startswith(' ') and ':' in line:
                        in_traceback = False
                        self._process_traceback(current_block, i - len(current_block) + 1)
                        current_block = []
                    continue

                # Check for OOM events (high priority)
                for pattern in self.OOM_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        self._add_oom_event(line, i + 1)
                        break

                # Check for errors
                for pattern, error_type in self.ERROR_PATTERNS:
                    if re.search(pattern, line):
                        self._add_error(line, i + 1, error_type)
                        break

                # Check for warnings
                for pattern, warning_type in self.WARNING_PATTERNS:
                    if re.search(pattern, line):
                        self._add_warning(line, i + 1, warning_type)
                        break

        except Exception as e:
            self.errors.append({
                'line_number': 0,
                'message': f'Failed to parse log file: {str(e)}',
                'type': 'parse_error',
                'timestamp': None,
            })

        return {
            'errors': self.errors,
            'warnings': self.warnings,
            'oom_events': self.oom_events,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'oom_count': len(self.oom_events),
        }

    def _process_traceback(self, lines: List[str], start_line: int):
        """Process a complete traceback block."""
        if not lines:
            return

        traceback_text = ''.join(lines)

        # Check if it's an OOM error
        is_oom = any(re.search(p, traceback_text, re.IGNORECASE) for p in self.OOM_PATTERNS)

        if is_oom:
            self._add_oom_event(traceback_text, start_line, is_traceback=True)
        else:
            # Extract the actual error message (last line)
            error_message = lines[-1].strip() if lines else 'Unknown error'
            self._add_error(
                error_message,
                start_line,
                'traceback',
                full_traceback=traceback_text
            )

    def _add_error(
        self,
        line: str,
        line_number: int,
        error_type: str,
        full_traceback: Optional[str] = None
    ):
        """Add an error to the list."""
        timestamp = self._extract_timestamp(line)

        error_entry = {
            'line_number': line_number,
            'message': line.strip()[:500],  # Truncate long messages
            'type': error_type,
            'timestamp': timestamp,
        }

        if full_traceback:
            error_entry['traceback'] = full_traceback[:2000]  # Limit traceback size

        self.errors.append(error_entry)

    def _add_warning(self, line: str, line_number: int, warning_type: str):
        """Add a warning to the list."""
        timestamp = self._extract_timestamp(line)

        self.warnings.append({
            'line_number': line_number,
            'message': line.strip()[:500],
            'type': warning_type,
            'timestamp': timestamp,
        })

    def _add_oom_event(self, text: str, line_number: int, is_traceback: bool = False):
        """Add an OOM event to the list."""
        timestamp = self._extract_timestamp(text)

        # Try to extract memory info from the error
        memory_info = self._extract_memory_from_oom(text)

        self.oom_events.append({
            'line_number': line_number,
            'message': text.strip()[:1000],
            'timestamp': timestamp,
            'is_traceback': is_traceback,
            'memory_info': memory_info,
        })

    def _extract_timestamp(self, line: str) -> Optional[float]:
        """
        Try to extract a timestamp from a log line.
        Common formats: ISO 8601, epoch, custom formats.
        """
        # Try ISO 8601 format: 2024-01-15T10:30:45
        iso_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})'
        match = re.search(iso_pattern, line)
        if match:
            try:
                dt = datetime.fromisoformat(match.group(1))
                return dt.timestamp()
            except Exception:
                pass

        # Try timestamp with date and time: 2024-01-15 10:30:45
        dt_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'
        match = re.search(dt_pattern, line)
        if match:
            try:
                dt = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                return dt.timestamp()
            except Exception:
                pass

        # Try epoch timestamp in brackets: [1234567890.123]
        epoch_pattern = r'\[(\d{10,13}(?:\.\d+)?)\]'
        match = re.search(epoch_pattern, line)
        if match:
            try:
                timestamp = float(match.group(1))
                # Convert milliseconds to seconds if needed
                if timestamp > 1e12:
                    timestamp = timestamp / 1000
                return timestamp
            except Exception:
                pass

        return None

    def _extract_memory_from_oom(self, text: str) -> Dict[str, Any]:
        """Extract memory information from OOM error message."""
        info = {}

        # Try to extract attempted allocation size
        # Pattern: "Tried to allocate X.XX GiB"
        alloc_pattern = r'Tried to allocate\s+([\d.]+)\s*(GiB|MiB|GB|MB)'
        match = re.search(alloc_pattern, text)
        if match:
            size = float(match.group(1))
            unit = match.group(2)
            if unit in ('MiB', 'MB'):
                size = size / 1024
            info['attempted_allocation_gb'] = size

        # Try to extract GPU memory info
        # Pattern: "X.XX GiB total capacity"
        total_pattern = r'([\d.]+)\s*(GiB|GB)\s+total capacity'
        match = re.search(total_pattern, text)
        if match:
            info['gpu_total_gb'] = float(match.group(1))

        # Pattern: "X.XX GiB already allocated"
        allocated_pattern = r'([\d.]+)\s*(GiB|GB)\s+already allocated'
        match = re.search(allocated_pattern, text)
        if match:
            info['gpu_allocated_gb'] = float(match.group(1))

        return info

    def get_summary(self) -> str:
        """Get a human-readable summary of the log analysis."""
        if not self.errors and not self.warnings and not self.oom_events:
            return "Log analysis complete: No errors or warnings found."

        summary_parts = []

        if self.oom_events:
            summary_parts.append(f"Found {len(self.oom_events)} out-of-memory error(s)")

        if self.errors:
            summary_parts.append(f"Found {len(self.errors)} error(s)")

        if self.warnings:
            summary_parts.append(f"Found {len(self.warnings)} warning(s)")

        return "Log analysis: " + ", ".join(summary_parts)


def parse_training_log(log_path: str) -> Dict[str, Any]:
    """
    Convenience function to parse a training log file.

    Args:
        log_path: Path to the log file

    Returns:
        Dictionary with parsed errors, warnings, and OOM events
    """
    parser = LogParser(log_path)
    return parser.parse()
