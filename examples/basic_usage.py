#!/usr/bin/env python3
"""Basic usage example for the AMOS harness.

Demonstrates routing decisions for personal, complex, and simple queries.
Requires a running Ollama instance for local model calls.

Usage:
    python examples/basic_usage.py
"""

import logging

from amos import AMOS

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")


def main() -> None:
    # Create AMOS with default config (Ollama local, GPT cloud)
    amos = AMOS(config={
        "memory_db": "~/.amos/example_memory.db",
        "local_model": "qwen2.5-coder:7b",
        "cloud_model": "gpt-4o-mini",
    })

    queries = [
        # Personal query — should route to local
        ("내 체중 기록을 분석해줘", {"type": "personal"}),
        # Complex query — should route to cloud
        (
            "Compare and analyze the architectural differences between "
            "microservices and monolithic systems. Explain the trade-offs "
            "in detail for a high-traffic e-commerce platform.",
            {"type": "research"},
        ),
        # Simple query — should route to local
        ("What is 2 + 2?", {"type": "simple"}),
    ]

    print("=" * 60)
    print("AMOS v0.1 — Basic Usage Demo")
    print("=" * 60)

    for text, meta in queries:
        print(f"\nQuery: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"Metadata: {meta}")

        try:
            response = amos.run(text, metadata=meta)
            decision = response.routing_decision

            print(f"  -> Target:     {decision.target}")
            print(f"  -> Reason:     {decision.reason}")
            print(f"  -> Confidence: {decision.confidence:.2f}")
            print(f"  -> Model:      {response.model_used}")
            print(f"  -> Latency:    {response.latency_ms}ms")
            print(f"  -> Response:   {response.text[:120]}...")
        except Exception as exc:
            print(f"  -> Error: {exc}")

    # Print memory stats
    print("\n" + "=" * 60)
    print("Memory Stats:")
    stats = amos.stats()
    print(f"  Total records: {stats['total_records']}")
    for target, info in stats.get("per_target", {}).items():
        print(f"  {target}: {info['total']} queries, "
              f"{info['success_rate']:.0%} success, "
              f"avg {info['avg_latency_ms']}ms")
    print("=" * 60)

    amos.close()


if __name__ == "__main__":
    main()
