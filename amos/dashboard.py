"""Routing statistics dashboard for AMOS — plain-text CLI output using only stdlib."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

from amos.memory import MemoryLayer


# Spark-line block characters (U+2581..U+2588), 8 levels
_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _spark(values: list[float]) -> str:
    """Render a list of floats as a Unicode spark-line."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0
    return "".join(
        _SPARK_CHARS[min(int((v - lo) / span * 7) + 1, 8)] for v in values
    )


def _bar(ratio: float, width: int = 10) -> str:
    """Render a ratio (0..1) as a filled/empty bar."""
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


class RoutingDashboard:
    """Generates plain-text routing statistics from the AMOS memory layer."""

    def __init__(self, memory: MemoryLayer) -> None:
        self._memory = memory

    def summary(self) -> str:
        """Return a formatted text summary of routing statistics."""
        raw = self._gather_stats()
        lines: list[str] = []
        sep = "═" * 43

        lines.append(sep)
        lines.append("  AMOS Routing Dashboard")
        lines.append(sep)

        total = raw["total"]
        local_n = raw["local_total"]
        cloud_n = raw["cloud_total"]
        local_pct = (local_n / total * 100) if total else 0.0
        cloud_pct = (cloud_n / total * 100) if total else 0.0

        lines.append(f"  Total queries:   {total:>5}")
        lines.append(f"  Local queries:   {local_n:>5} ({local_pct:.1f}%)")
        lines.append(f"  Cloud queries:   {cloud_n:>5} ({cloud_pct:.1f}%)")
        lines.append("")

        # Success rates
        lines.append("  Success rates:")
        lr = raw["local_success_rate"]
        cr = raw["cloud_success_rate"]
        lines.append(f"    Local:  {lr * 100:5.1f}%  {_bar(lr)}")
        lines.append(f"    Cloud:  {cr * 100:5.1f}%  {_bar(cr)}")
        lines.append("")

        # Avg latency
        lines.append("  Avg latency:")
        lines.append(f"    Local:  {raw['local_avg_latency']:>5}ms")
        lines.append(f"    Cloud:  {raw['cloud_avg_latency']:>5}ms")
        lines.append("")

        # Top routing reasons
        if raw["reasons"]:
            lines.append("  Top routing reasons:")
            for reason, count in raw["reasons"]:
                lines.append(f"    {reason:<16}: {count} queries")
            lines.append("")

        # Recent trend (last 24h)
        if raw["local_trend"] or raw["cloud_trend"]:
            lines.append("  Recent trend (last 24h):")
            if raw["local_trend"]:
                lines.append(f"    Local  {_spark(raw['local_trend'])}")
            if raw["cloud_trend"]:
                lines.append(f"    Cloud  {_spark(raw['cloud_trend'])}")

        lines.append(sep)
        return "\n".join(lines)

    def print(self) -> None:
        """Print the dashboard summary to stdout."""
        print(self.summary())

    def export_json(self) -> dict:
        """Return raw statistics as a dict for programmatic use."""
        raw = self._gather_stats()
        return {
            "total_queries": raw["total"],
            "local": {
                "count": raw["local_total"],
                "success_rate": round(raw["local_success_rate"], 4),
                "avg_latency_ms": raw["local_avg_latency"],
            },
            "cloud": {
                "count": raw["cloud_total"],
                "success_rate": round(raw["cloud_success_rate"], 4),
                "avg_latency_ms": raw["cloud_avg_latency"],
            },
            "routing_reasons": dict(raw["reasons"]),
            "trends": {
                "local_24h": raw["local_trend"],
                "cloud_24h": raw["cloud_trend"],
            },
        }

    def mood_breakdown(self) -> str:
        """Return a formatted text summary of mood distribution."""
        raw = self._memory.mood_breakdown()
        if not raw:
            return "  No mood data recorded yet."

        total = sum(raw.values())
        lines: list[str] = ["  Mood distribution:"]

        for mood_val, count in raw.items():
            pct = count / total * 100
            bar_width = max(1, round(pct / 5))  # ~1 char per 5%
            bar = "\u2588" * bar_width if pct >= 3 else "\u2591"
            lines.append(f"    {mood_val:<12}: {count:>3} ({pct:4.0f}%)  {bar}")

        return "\n".join(lines)

    def _gather_stats(self) -> dict:
        """Query the memory layer and compute dashboard data."""
        conn = self._memory._conn

        # Total counts per target
        rows = conn.execute(
            "SELECT target, COUNT(*) as cnt FROM experiences GROUP BY target"
        ).fetchall()
        counts = {r["target"]: r["cnt"] for r in rows}
        local_total = counts.get("local", 0)
        cloud_total = counts.get("cloud", 0)
        total = local_total + cloud_total

        # Success rates per target
        rows = conn.execute(
            """SELECT target,
                      SUM(success) as successes,
                      COUNT(*) as cnt
               FROM experiences GROUP BY target"""
        ).fetchall()
        rates = {}
        avg_latencies = {}
        for r in rows:
            rates[r["target"]] = r["successes"] / r["cnt"] if r["cnt"] else 0.0

        # Avg latency per target
        rows = conn.execute(
            "SELECT target, AVG(latency_ms) as avg_lat FROM experiences GROUP BY target"
        ).fetchall()
        for r in rows:
            avg_latencies[r["target"]] = round(r["avg_lat"]) if r["avg_lat"] else 0

        # Routing reasons — extract from query_text patterns stored in memory
        # We use the router reason keywords embedded in the routing decision cascade
        # Since we don't store reason directly, we approximate from the data:
        # count entries by looking at patterns in the experience data
        reason_rows = conn.execute(
            """SELECT
                 CASE
                   WHEN query_text LIKE '%personal%' OR query_text LIKE '%private%'
                        OR query_text LIKE '%나의%' OR query_text LIKE '%내%'
                        OR query_text LIKE '%체중%' OR query_text LIKE '%비밀번호%'
                     THEN 'privacy'
                   WHEN target = 'cloud' AND success = 0
                     THEN 'past_failure'
                   WHEN LENGTH(query_text) > 200
                        OR query_text LIKE '%analyze%'
                        OR query_text LIKE '%compare%'
                        OR query_text LIKE '%explain%'
                        OR query_text LIKE '%research%'
                     THEN 'complexity'
                   ELSE 'default'
                 END as reason,
                 COUNT(*) as cnt
               FROM experiences
               GROUP BY reason
               ORDER BY cnt DESC"""
        ).fetchall()
        reasons = [(r["reason"], r["cnt"]) for r in reason_rows]

        # 24h trend — bucket into 18 bins of ~80 minutes each
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=24)).isoformat()
        trend_rows = conn.execute(
            """SELECT target, timestamp FROM experiences
               WHERE timestamp >= ? ORDER BY timestamp""",
            (cutoff,),
        ).fetchall()

        num_bins = 18
        local_bins = [0.0] * num_bins
        cloud_bins = [0.0] * num_bins
        if trend_rows:
            bin_seconds = 24 * 3600 / num_bins
            for r in trend_rows:
                ts = datetime.fromisoformat(r["timestamp"])
                age_seconds = (now - ts).total_seconds()
                idx = max(0, min(num_bins - 1, int((24 * 3600 - age_seconds) / bin_seconds)))
                if r["target"] == "local":
                    local_bins[idx] += 1
                else:
                    cloud_bins[idx] += 1

        return {
            "total": total,
            "local_total": local_total,
            "cloud_total": cloud_total,
            "local_success_rate": rates.get("local", 0.0),
            "cloud_success_rate": rates.get("cloud", 0.0),
            "local_avg_latency": avg_latencies.get("local", 0),
            "cloud_avg_latency": avg_latencies.get("cloud", 0),
            "reasons": reasons,
            "local_trend": local_bins if any(v > 0 for v in local_bins) else [],
            "cloud_trend": cloud_bins if any(v > 0 for v in cloud_bins) else [],
        }
