import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


@dataclass
class LogEntry:
    """Represents a single log entry."""

    timestamp: datetime
    logger: str
    level: str
    message: str
    raw_line: str


class LogReader:
    """Utility class for reading and parsing log files."""

    def __init__(self, log_file_path: Path):
        """Initialize the log reader with a log file path."""
        self.log_file_path = Path(log_file_path)
        # Pattern to match Python logging format: timestamp - logger - level - message
        self.log_pattern = re.compile(
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (.+?) - (\w+) - (.+)"
        )

    def read_last_entries(self, count: int = 10) -> List[LogEntry]:
        """Read the last N entries from the log file."""
        if not self.log_file_path.exists():
            return []

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            # Parse log entries from the end
            entries = []
            for line in reversed(lines[-count * 2:]):  # Read more lines in case some don't match
                entry = self._parse_log_line(line.strip())
                if entry:
                    entries.append(entry)
                if len(entries) >= count:
                    break

            # Return entries in chronological order (oldest first of the last N)
            return list(reversed(entries))

        except Exception as e:
            print(f"Error reading log file: {e}")
            return []

    def read_entries_by_level(self, level: str, count: int = 50) -> List[LogEntry]:
        """Read entries filtered by log level."""
        if not self.log_file_path.exists():
            return []

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            entries = []
            for line in reversed(lines):
                entry = self._parse_log_line(line.strip())
                if entry and entry.level.upper() == level.upper():
                    entries.append(entry)
                if len(entries) >= count:
                    break

            return list(reversed(entries))

        except Exception as e:
            print(f"Error reading log file: {e}")
            return []

    def read_entries_since(self, since: datetime, count: int = 100) -> List[LogEntry]:
        """Read entries since a specific datetime."""
        if not self.log_file_path.exists():
            return []

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            entries = []
            for line in reversed(lines):
                entry = self._parse_log_line(line.strip())
                if entry:
                    if entry.timestamp >= since:
                        entries.append(entry)
                    else:
                        # Stop reading if we've gone past the since time
                        break
                if len(entries) >= count:
                    break

            return list(reversed(entries))

        except Exception as e:
            print(f"Error reading log file: {e}")
            return []

    def get_log_stats(self) -> Dict[str, int]:
        """Get statistics about the log file."""
        if not self.log_file_path.exists():
            return {}

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            stats = {
                "total_lines": len(lines),
                "INFO": 0,
                "WARNING": 0,
                "ERROR": 0,
                "DEBUG": 0,
                "CRITICAL": 0,
            }

            for line in lines:
                entry = self._parse_log_line(line.strip())
                if entry:
                    level = entry.level.upper()
                    if level in stats:
                        stats[level] += 1

            return stats

        except Exception as e:
            print(f"Error reading log file: {e}")
            return {}

    def _parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a single log line into a LogEntry object."""
        if not line:
            return None

        match = self.log_pattern.match(line)
        if not match:
            return None

        try:
            timestamp_str, logger, level, message = match.groups()
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

            return LogEntry(
                timestamp=timestamp, logger=logger, level=level, message=message, raw_line=line
            )
        except ValueError:
            return None


class MultiLogReader:
    """Reader for multiple log files."""

    def __init__(self, log_directory: Path):
        """Initialize with a directory containing log files."""
        self.log_directory = Path(log_directory)

    def get_available_log_files(self) -> List[Path]:
        """Get list of available log files."""
        if not self.log_directory.exists():
            return []

        return list(self.log_directory.glob("*.log"))

    def read_from_all_files(self, count: int = 10) -> List[LogEntry]:
        """Read entries from all log files and merge them chronologically."""
        all_entries = []

        for log_file in self.get_available_log_files():
            reader = LogReader(log_file)
            entries = reader.read_last_entries(count)
            all_entries.extend(entries)

        # Sort by timestamp
        all_entries.sort(key=lambda x: x.timestamp, reverse=True)

        return all_entries[:count]
