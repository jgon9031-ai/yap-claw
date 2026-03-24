# YAP-CLAW

**Yappy Agent Picking Cloud or Local Automatically & Wisely**

> *formerly AMOS (Autonomous Meta-Orchestration System)*

AMOS is a lightweight hybrid AI harness that intelligently routes queries between local (Ollama) and cloud (Claude/GPT) models. It records outcomes in a persistent memory layer and uses past experience to improve routing decisions over time — a practical implementation inspired by the arXiv paper ["Why AI systems do not learn and what to do about it"](https://arxiv.org/abs/2603.15381).

## Architecture

```
                         ┌─────────────────┐
                         │    User Query    │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │   System M      │
                         │   (Router)      │
                         │                 │
                         │ 1. Privacy      │
                         │ 2. Past failure │
                         │ 3. Complexity   │
                         └───┬─────────┬───┘
                             │         │
                    local    │         │   cloud
                 ┌───────────▼──┐  ┌───▼───────────┐
                 │   Ollama     │  │  Claude / GPT  │
                 │  (on-device) │  │   (API call)   │
                 └───────┬──────┘  └───┬────────────┘
                         │             │
                         └──────┬──────┘
                                │
                       ┌────────▼────────┐
                       │  Memory Layer   │
                       │   (SQLite +     │
                       │    FTS5)        │
                       └────────┬────────┘
                                │
                       ┌────────▼────────┐
                       │    Response     │
                       └─────────────────┘
```

### Routing Logic (System M)

1. **Privacy check** — queries containing personal keywords stay local
2. **Past failure check** — if local models fail often on similar queries, escalate to cloud
3. **Complexity heuristic** — long, multi-question, keyword-heavy queries route to cloud
4. **Default** — prefer local for cost and latency

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Ensure Ollama is running with a model
ollama pull qwen2.5-coder:7b

# Set cloud API key (optional, only needed for cloud routing)
export OPENAI_API_KEY=sk-...

# Run example
python examples/basic_usage.py

# Run tests
pytest
```

## Usage

```python
from amos import AMOS

amos = AMOS(config={
    "local_model": "qwen2.5-coder:7b",
    "cloud_model": "gpt-4o-mini",
})

# Personal query -> routes to local automatically
response = amos.run("내 체중 기록을 분석해줘")
print(response.routing_decision.target)  # "local"

# Complex query -> routes to cloud
response = amos.run("Analyze and compare microservices vs monolithic architectures")
print(response.routing_decision.target)  # "cloud"

# Provide feedback to improve future routing
amos.feedback(response, success=True)

# Check memory stats
print(amos.stats())
```

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `memory_db` | `~/.amos/memory.db` | SQLite database path |
| `local_base_url` | `http://localhost:11434/v1` | Ollama API endpoint |
| `local_model` | `qwen2.5-coder:7b` | Local model name |
| `cloud_model` | `gpt-4o-mini` | Cloud model name |
| `cloud_base_url` | *(OpenAI default)* | Cloud API endpoint |
| `cloud_api_key` | *(from env)* | API key (or set `OPENAI_API_KEY`) |

## Self-Improving Loop

AMOS v0.2 implements a three-layer self-improving loop that progressively optimizes routing decisions:

```
┌─────────────────────────────────────────────────────────┐
│                   AMOS Self-Improving Loop               │
│                                                         │
│  Layer 1: Memory (raw experience)                       │
│  ┌───────────────────────────────────────┐              │
│  │ SQLite + FTS5: every query outcome    │              │
│  │ recorded with target, success, latency│              │
│  └──────────────┬────────────────────────┘              │
│                 │                                        │
│  Layer 2: SessionAnalyzer (Hermes-inspired)             │
│  ┌──────────────▼────────────────────────┐              │
│  │ Every N interactions, reviews recent  │              │
│  │ experiences to extract patterns:      │              │
│  │ - which categories fail locally       │              │
│  │ - which are fast/slow per target      │              │
│  │ - preferred routing by category       │              │
│  │ Outputs: routing_hints -> Router      │              │
│  └──────────────┬────────────────────────┘              │
│                 │                                        │
│  Layer 3: SkillHealthTracker (cognee-inspired)          │
│  ┌──────────────▼────────────────────────┐              │
│  │ observe() -> inspect() -> amend() ->  │              │
│  │ evaluate()                            │              │
│  │ - tracks per-category failure rates   │              │
│  │ - flags issues when rate > threshold  │              │
│  │ - suggests routing amendments         │              │
│  │ - evaluates if amendments improve     │              │
│  │ - rolls back if not improved          │              │
│  └───────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### How It Works

1. **Memory** records every query outcome (target, model, success, latency)
2. **SessionAnalyzer** (runs in a background thread every N queries) reviews recent history and generates routing hints — e.g., "code queries work better on cloud"
3. **SkillHealthTracker** watches each query outcome in real-time. When a category's failure rate crosses a threshold, it suggests a routing amendment and later evaluates whether it actually helped

Both subsystems feed routing hints into the Router, which applies them before its standard decision cascade.

### Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `analyze_every` | `10` | Queries between SessionAnalyzer runs |
| `failure_threshold` | `0.4` | Failure rate that triggers a HealthIssue |

### Inspirations

- **SessionAnalyzer**: [NousResearch Hermes Agent](https://github.com/NousResearch/hermes-function-calling) — periodic self-reflection on interaction patterns
- **SkillHealthTracker**: [cognee-skills](https://github.com/topoteretes/cognee) — observe/inspect/amend/evaluate loop for continuous skill improvement

## v0.3 Features

### Real Ollama Integration with Error Handling

AMOS v0.3 adds production-grade error handling for the local Ollama executor:

- **Connection errors** (Ollama not running) raise `LocalUnavailableError`
- **Model not found** (not pulled) raises `ModelNotFoundError`
- **Configurable timeout** (default 30s) with automatic retry (1 retry on timeout)
- **Auto-fallback**: when local is unavailable, the harness silently falls back to cloud without recording a routing failure (infrastructure issue, not a routing mistake)

```python
from amos import AMOS, LocalUnavailableError, ModelNotFoundError

amos = AMOS(config={
    "local_timeout": 30.0,       # seconds before timeout
    "local_max_retries": 1,      # retry once on timeout
})

# If Ollama is down, run() auto-falls back to cloud
response = amos.run("print hello world in python")
```

#### New Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `local_timeout` | `30.0` | Seconds before local request times out |
| `local_max_retries` | `1` | Number of retries on timeout |
| `cloud_timeout` | `60.0` | Seconds before cloud request times out |

#### Running Integration Tests

```bash
# Unit tests (no Ollama needed)
pytest

# Integration tests (requires running Ollama with qwen2.5-coder:7b)
pytest -m integration
```

### Routing Statistics Dashboard

A plain-text CLI dashboard for visualizing routing statistics:

```python
from amos import AMOS

amos = AMOS()
# ... after running some queries ...

# Get dashboard
dash = amos.dashboard()
dash.print()
```

Example output:

```
═══════════════════════════════════════════
  AMOS Routing Dashboard
═══════════════════════════════════════════
  Total queries:     142
  Local queries:      89 (62.7%)
  Cloud queries:      53 (37.3%)

  Success rates:
    Local:  84.3%  ████████░░
    Cloud:  96.2%  █████████░

  Avg latency:
    Local:   312ms
    Cloud:  1842ms

  Top routing reasons:
    privacy         : 34 queries
    complexity      : 28 queries
    past_failure    : 15 queries
    default         : 65 queries

  Recent trend (last 24h):
    Local  ▁▂▃▄▂▃▅▄▃▂▄▃▅▆▄▃▂▁
    Cloud  ▃▂▁▂▃▄▂▃▄▃▂▁▂▃▂▃▄▅
═══════════════════════════════════════════
```

Programmatic access:

```python
data = dash.export_json()
print(data["local"]["success_rate"])  # 0.843
```

Run the demo:

```bash
python examples/dashboard_demo.py
```

## Paper Reference

> "Why AI systems do not learn and what to do about it"
> arXiv:2603.15381
> https://arxiv.org/abs/2603.15381
