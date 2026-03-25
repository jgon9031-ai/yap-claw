"""
NEST Seed Script — Mood-Tagged Experience Importer
===================================================
Seeds the NEST memory layer with mood-tagged experiences for the MOOD layer.

Run:
    python3 scripts/seed_mood_patterns.py [--db ~/.amos/memory.db] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from amos.memory import MemoryLayer


# ─────────────────────────────────────────────────────────────────────────────
# Mood-tagged patterns
# Each entry: (query_text, target, success, mood, count)
# ─────────────────────────────────────────────────────────────────────────────

MOOD_TAGGED: list[tuple[str, str, bool, str, int]] = [
    # frustrated
    ("실패에서 배우는게 없니", "cloud", True, "frustrated", 1),
    ("또 실패네", "local", False, "frustrated", 3),
    # satisfied
    ("응 좋아", "local", True, "satisfied", 8),
    ("응 그래", "local", True, "satisfied", 6),
    ("멋지게 만들어줘", "cloud", True, "satisfied", 2),
    # curious
    ("가능한가", "cloud", True, "curious", 4),
    ("어떻게 구축 가능할지", "cloud", True, "curious", 3),
    ("방법이 있을까", "cloud", True, "curious", 4),
    # urgent
    ("바로 실행해", "local", True, "urgent", 4),
    ("지금 확인해줘", "local", True, "urgent", 5),
    # neutral
    ("76.5", "local", True, "neutral", 8),
    ("점심 라면 저녁 떡볶이", "local", True, "neutral", 5),
    ("응", "local", True, "neutral", 10),
]


def seed(db_path: str, dry_run: bool = False) -> None:
    mem = MemoryLayer(db_path=db_path)
    total = 0

    print(f"{'[DRY RUN] ' if dry_run else ''}Seeding mood-tagged patterns at {db_path}")
    print(f"Patterns to import: {len(MOOD_TAGGED)}")

    for query_text, target, success, mood, count in MOOD_TAGGED:
        model = "qwen2.5-coder:7b" if target == "local" else "gpt-4o"
        latency = 200 if target == "local" else 1800
        for _ in range(count):
            if not dry_run:
                mem.record(
                    query_text=query_text,
                    target=target,
                    model_used=model,
                    success=success,
                    latency_ms=latency,
                    mood=mood,
                )
            total += 1

    print(f"\n{'Would import' if dry_run else 'Imported'}: {total} mood-tagged records")

    if not dry_run:
        breakdown = mem.mood_breakdown()
        print("\nMood breakdown:")
        for mood_val, cnt in breakdown.items():
            print(f"  {mood_val:<12}: {cnt}")

    mem.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed NEST with mood-tagged patterns")
    parser.add_argument("--db", default="~/.amos/memory.db", help="NEST DB path")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()
    seed(db_path=args.db, dry_run=args.dry_run)
