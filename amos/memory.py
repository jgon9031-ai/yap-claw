"""Memory layer for AMOS — SQLite-backed experience storage with simple retrieval."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from amos.models import Experience


class MemoryLayer:
    """Persistent memory that records and retrieves past query experiences.

    Uses SQLite with FTS5 full-text search for efficient keyword-based
    similarity retrieval without heavy ML dependencies.
    """

    def __init__(self, db_path: str = "~/.amos/memory.db") -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the experiences table and FTS5 virtual table."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                target TEXT NOT NULL,
                model_used TEXT NOT NULL,
                success INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts
            USING fts5(query_text, content=experiences, content_rowid=id);

            -- Triggers to keep FTS index in sync
            CREATE TRIGGER IF NOT EXISTS experiences_ai AFTER INSERT ON experiences BEGIN
                INSERT INTO experiences_fts(rowid, query_text)
                VALUES (new.id, new.query_text);
            END;

            CREATE TRIGGER IF NOT EXISTS experiences_ad AFTER DELETE ON experiences BEGIN
                INSERT INTO experiences_fts(experiences_fts, rowid, query_text)
                VALUES ('delete', old.id, old.query_text);
            END;
        """)
        self._conn.commit()
        self._migrate_mood_column()

    def record(
        self,
        query_text: str,
        target: str,
        model_used: str,
        success: bool,
        latency_ms: int,
        mood: str | None = None,
    ) -> None:
        """Save an experience to the memory layer."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO experiences (query_text, target, model_used, success, latency_ms, timestamp, mood)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (query_text, target, model_used, int(success), latency_ms, now, mood),
        )
        self._conn.commit()

    def retrieve_similar(self, query_text: str, top_k: int = 5) -> list[Experience]:
        """Retrieve similar past experiences using FTS5 keyword matching.

        Falls back to LIKE-based search if FTS match returns nothing.
        """
        # Try FTS5 first
        rows = self._conn.execute(
            """SELECT e.* FROM experiences e
               JOIN experiences_fts f ON e.id = f.rowid
               WHERE experiences_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (self._fts_query(query_text), top_k),
        ).fetchall()

        # Fallback: LIKE-based search on individual words
        if not rows:
            words = query_text.split()[:5]  # Use first 5 words
            conditions = " OR ".join(["query_text LIKE ?"] * len(words))
            params = [f"%{w}%" for w in words]
            params.append(top_k)
            rows = self._conn.execute(
                f"""SELECT * FROM experiences
                    WHERE {conditions}
                    ORDER BY timestamp DESC
                    LIMIT ?""",
                params,
            ).fetchall()

        return [self._row_to_experience(r) for r in rows]

    def get_local_failure_rate(self, query_text: str) -> float:
        """Check failure rate for similar queries routed to local models.

        Returns a float between 0.0 (no failures) and 1.0 (all failed).
        Returns 0.0 if no similar experiences exist.
        """
        similar = self.retrieve_similar(query_text, top_k=10)
        local_experiences = [e for e in similar if e.target == "local"]
        if not local_experiences:
            return 0.0
        failures = sum(1 for e in local_experiences if not e.success)
        return failures / len(local_experiences)

    def get_recent(self, limit: int = 50) -> list[Experience]:
        """Retrieve the most recent experiences ordered by timestamp descending."""
        rows = self._conn.execute(
            "SELECT * FROM experiences ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_experience(r) for r in rows]

    def count_since(self, since_id: int) -> int:
        """Count records with id > since_id."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM experiences WHERE id > ?", (since_id,)
        ).fetchone()
        return row[0]

    def max_id(self) -> int:
        """Return the maximum experience id, or 0 if empty."""
        row = self._conn.execute("SELECT MAX(id) FROM experiences").fetchone()
        return row[0] or 0

    def stats(self) -> dict:
        """Return summary statistics about the memory layer."""
        total = self._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

        targets = self._conn.execute(
            """SELECT target,
                      COUNT(*) as total,
                      SUM(success) as successes,
                      AVG(latency_ms) as avg_latency
               FROM experiences
               GROUP BY target"""
        ).fetchall()

        per_target = {}
        for row in targets:
            t = row["total"]
            s = row["successes"]
            per_target[row["target"]] = {
                "total": t,
                "success_rate": s / t if t > 0 else 0.0,
                "avg_latency_ms": round(row["avg_latency"]),
            }

        return {"total_records": total, "per_target": per_target}

    def mood_breakdown(self) -> dict[str, int]:
        """Return {mood: count} breakdown from the experiences table."""
        rows = self._conn.execute(
            """SELECT COALESCE(mood, 'neutral') as mood_val, COUNT(*) as cnt
               FROM experiences
               GROUP BY mood_val
               ORDER BY cnt DESC"""
        ).fetchall()
        return {r["mood_val"]: r["cnt"] for r in rows}

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _migrate_mood_column(self) -> None:
        """Add mood column if it doesn't exist (backward-compatible migration)."""
        cursor = self._conn.execute("PRAGMA table_info(experiences)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "mood" not in columns:
            self._conn.execute("ALTER TABLE experiences ADD COLUMN mood TEXT")
            self._conn.commit()

    @staticmethod
    def _fts_query(text: str) -> str:
        """Convert natural text into an FTS5 OR query."""
        words = text.split()
        # Escape special FTS5 characters and join with OR
        safe = [w.replace('"', '""') for w in words if len(w) > 1]
        if not safe:
            return text
        return " OR ".join(f'"{w}"' for w in safe[:10])

    @staticmethod
    def _row_to_experience(row: sqlite3.Row) -> Experience:
        mood_val = None
        try:
            mood_val = row["mood"]
        except (IndexError, KeyError):
            pass
        return Experience(
            query_text=row["query_text"],
            target=row["target"],
            model_used=row["model_used"],
            success=bool(row["success"]),
            latency_ms=row["latency_ms"],
            mood=mood_val,
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )
