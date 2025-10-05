from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    message: str


class LogReader:
    def __init__(self, log_file_path: Path):
        self.log_file_path = Path(log_file_path)

    def read_last_entries(self, num_entries: int = 20) -> List[LogEntry]:
        entries = []
        if not self.log_file_path.exists():
            return entries

        with open(self.log_file_path, "r") as f:
            lines = f.readlines()[-num_entries:]

        for line in lines:
            entry = self._parse_log_line(line)
            if entry:
                entries.append(entry)

        return entries

    def read_entries_by_level(
        self, level: str, num_entries: int = 20
    ) -> List[LogEntry]:
        entries = []
        if not self.log_file_path.exists():
            return entries

        with open(self.log_file_path, "r") as f:
            for line in reversed(f.readlines()):
                entry = self._parse_log_line(line)
                if entry and entry.level == level:
                    entries.append(entry)
                    if len(entries) >= num_entries:
                        break

        return list(reversed(entries))

    def get_log_stats(self) -> dict:
        if not self.log_file_path.exists():
            return {}

        stats = {"total_lines": 0}
        with open(self.log_file_path, "r") as f:
            for line in f:
                stats["total_lines"] += 1
                entry = self._parse_log_line(line)
                if entry:
                    stats[entry.level] = stats.get(entry.level, 0) + 1

        return stats

    @staticmethod
    def _parse_log_line(line: str) -> LogEntry | None:
        try:
            # Expected format from logging config: [timestamp] - LEVEL - message
            parts = line.strip().split(" - ")
            if len(parts) >= 3:
                timestamp_str = parts[0].strip("[]")
                timestamp = datetime.fromisoformat(timestamp_str)
                level = parts[1].strip()
                message = " - ".join(parts[2:])
                return LogEntry(timestamp, level, message)
        except Exception:
            return None

        return None
