"""SQLite persistence helpers for recognised text history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3
import threading


@dataclass(frozen=True, slots=True)
class HistoryItem:
    """A single recognised text history row."""

    id: int
    text: str
    source: str
    ts: datetime

    @property
    def formatted_ts(self) -> str:
        """Return the timestamp as 'May 17, 2026 · 09:41'."""

        return self.ts.strftime("%b %d, %Y · %H:%M")


class HistoryDatabase:
    """Small SQLite repository for speech and sign recognition history."""

    def __init__(self, db_path: Path) -> None:
        """Create a database helper bound to the given SQLite path."""

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        """Create the history schema when it does not already exist."""

        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL CHECK(source IN ('speech', 'sign')),
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_text ON history(text)"
            )
            self._connection.commit()

    def add_history(self, text: str, source: str) -> int:
        """Insert recognised text and return its database id."""

        if source not in {"speech", "sign"}:
            raise ValueError("source must be 'speech' or 'sign'")
        with self._lock:
            cursor = self._connection.execute(
                "INSERT INTO history(text, source) VALUES (?, ?)",
                (text.strip(), source),
            )
            self._connection.commit()
            return int(cursor.lastrowid)

    def list_history(self, search: str = "", limit: int = 200) -> list[HistoryItem]:
        """Return recent history rows, optionally filtered by recognised text."""

        needle = search.strip()
        with self._lock:
            if needle:
                rows = self._connection.execute(
                    """
                    SELECT id, text, source, ts
                    FROM history
                    WHERE text LIKE ?
                    ORDER BY datetime(ts) DESC, id DESC
                    LIMIT ?
                    """,
                    (f"%{needle}%", limit),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """
                    SELECT id, text, source, ts
                    FROM history
                    ORDER BY datetime(ts) DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def clear_history(self) -> None:
        """Delete all recognised text history."""

        with self._lock:
            self._connection.execute("DELETE FROM history")
            self._connection.commit()

    def close(self) -> None:
        """Close the SQLite connection."""

        with self._lock:
            self._connection.close()

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> HistoryItem:
        """Convert a SQLite row into a typed history item."""

        raw_ts = str(row["ts"])
        try:
            ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S")
        return HistoryItem(
            id=int(row["id"]),
            text=str(row["text"]),
            source=str(row["source"]),
            ts=ts,
        )


__all__ = ["HistoryDatabase", "HistoryItem"]
