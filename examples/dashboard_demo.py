#!/usr/bin/env python3
"""Demo: seed fake routing data and print the AMOS dashboard.

Usage:
    python examples/dashboard_demo.py
"""

from __future__ import annotations

import random

from amos.dashboard import RoutingDashboard
from amos.memory import MemoryLayer


def seed_fake_data(memory: MemoryLayer, count: int = 142) -> None:
    """Seed the memory layer with realistic fake routing data."""
    query_templates = {
        "local": [
            "print hello world in python",
            "내 체중 기록을 보여줘",
            "fix the typo in line 42",
            "write a function to reverse a string",
            "translate 'hello' to Korean",
            "my personal notes summary",
            "calculate 15 * 23",
            "list files in current directory",
        ],
        "cloud": [
            "analyze and compare React vs Vue architecture patterns",
            "explain the CAP theorem and its implications for microservices",
            "research best practices for distributed caching",
            "summarize the differences between REST and GraphQL",
            "compare LSTM and Transformer models for time-series forecasting",
        ],
    }

    local_count = int(count * 0.63)
    cloud_count = count - local_count

    for i in range(local_count):
        tpl = random.choice(query_templates["local"])
        memory.record(
            query_text=f"{tpl} (variation {i})",
            target="local",
            model_used="qwen2.5-coder:7b",
            success=random.random() < 0.84,
            latency_ms=random.randint(150, 500),
        )

    for i in range(cloud_count):
        tpl = random.choice(query_templates["cloud"])
        memory.record(
            query_text=f"{tpl} (variation {i})",
            target="cloud",
            model_used="gpt-4o-mini",
            success=random.random() < 0.96,
            latency_ms=random.randint(1200, 2500),
        )


def main() -> None:
    # Use in-memory DB for demo
    memory = MemoryLayer(db_path=":memory:")
    seed_fake_data(memory, count=142)

    dashboard = RoutingDashboard(memory)

    print()
    dashboard.print()
    print()

    # Also show JSON export
    print("JSON export (first few fields):")
    data = dashboard.export_json()
    print(f"  total_queries: {data['total_queries']}")
    print(f"  local:  {data['local']}")
    print(f"  cloud:  {data['cloud']}")
    print(f"  reasons: {data['routing_reasons']}")

    memory.close()


if __name__ == "__main__":
    main()
