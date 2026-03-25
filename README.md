<div align="center">

<img src="assets/banner.jpg" alt="YAP-CLAW Banner" width="100%"/>

**Yappy Agent Picking Cloud or Local Automatically & Wisely**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-80%20passed-brightgreen?logo=pytest)](tests/)
[![Version](https://img.shields.io/badge/version-0.3.0-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/Inspired%20by-arXiv%202603.15381-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.15381)
[![GitHub](https://img.shields.io/badge/GitHub-yap--claw-181717?logo=github)](https://github.com/jgon9031-ai/yap-claw)

*Drop a query. YAP-CLAW figures out where to send it — local or cloud — and gets smarter every time.* 🐾

</div>

---

## What is YAP-CLAW?

YAP-CLAW is a **self-improving hybrid AI routing harness** that intelligently dispatches queries between on-device (**System B** / Ollama) and cloud (**System A** / Claude, GPT) models. Unlike static routers, YAP-CLAW remembers past successes and failures — and adapts its routing decisions over time without retraining.

Inspired by the cognitive science paper **["Why AI systems don't learn and what to do about it"](https://arxiv.org/abs/2603.15381)** (META FAIR, NYU, UC Berkeley, 2026):

| Paper Concept | YAP-CLAW Implementation | Role |
|---|---|---|
| **System A** — Learning from observation | ☁️ Cloud models (Claude / GPT / Gemini) | Handles complex reasoning, latest knowledge |
| **System B** — Learning from action | 🖥️ Local models via Ollama | Handles private data, fast & cost-free inference |
| **System M** — Meta-controller | 🧠 YAP-CLAW Router | Decides which system handles each query |
| Persistent memory | 💾 SQLite + FTS5 Experience Store | Records outcomes, improves routing over time |

> **Key insight:** System A and System B are not alternatives — they are **complementary agents** in an A/B routing architecture. The Router (System M) dynamically assigns each query to the best agent based on privacy, complexity, and historical performance.

---

## Architecture

```
              ╔═══════════════════════════════════════════════╗
              ║              YAP-CLAW Harness                 ║
              ║                                               ║
  query  ───► ║  ┌───────────────────────────────────────┐   ║
              ║  │          🧠 System M  (Router)         │   ║
              ║  │                                       │   ║
              ║  │  1. 🔒 Privacy check                  │   ║
              ║  │  2. 📉 Past local failure rate        │   ║
              ║  │  3. 🧮 Complexity heuristic score     │   ║
              ║  │  4. 💾 Memory hints (experience)      │   ║
              ║  └──────────────┬────────────────────────┘   ║
              ║                 │                             ║
              ║        ┌────────┴────────┐                   ║
              ║        ▼                 ▼                   ║
              ║  ┌───────────────┐ ┌───────────────┐        ║
              ║  │  🖥️  Sys B    │ │  ☁️  Sys A    │        ║
              ║  │  (Local)      │ │  (Cloud)      │        ║
              ║  │               │ │               │        ║
              ║  │  • Ollama     │ │  • OpenAI GPT │        ║
              ║  │  • qwen2.5    │ │  • Claude     │        ║
              ║  │  • llama3     │ │  • Gemini     │        ║
              ║  │  • mistral    │ │  • Any OAI-   │        ║
              ║  │  • phi3       │ │    compatible │        ║
              ║  └───────┬───────┘ └───────┬───────┘        ║
              ║          └────────┬─────────┘               ║
              ║                   ▼                          ║
              ║  ┌────────────────────────────────────┐     ║
              ║  │         💾 Memory Layer             │     ║
              ║  │   SQLite + FTS5 experience store   │     ║
              ║  │   Records: query · target · result │     ║
              ║  │   (Router gets smarter over time📈)│     ║
              ║  └────────────────────────────────────┘     ║
              ╚═══════════════════════════════════════════════╝
```

---

## 🛠️ Prerequisites

### System B — Local Agent (Ollama)

Install [Ollama](https://ollama.com) to enable on-device inference:

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull a coding model (recommended for general use)
ollama pull qwen2.5-coder:7b

# Or lighter alternatives
ollama pull llama3.2:3b       # Fast, general purpose
ollama pull phi3:mini          # Very lightweight (3.8B)
ollama pull mistral:7b         # Strong reasoning

# Verify
ollama list
ollama run qwen2.5-coder:7b "Say hello"
```

> Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1` by default.

### System A — Cloud Agent

Set your preferred cloud provider API key:

```bash
# Option 1: OpenAI
export OPENAI_API_KEY="sk-..."

# Option 2: Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Option 3: Google Gemini
export GEMINI_API_KEY="AIza..."

# Option 4: Any OpenAI-compatible endpoint
export OPENAI_BASE_URL="https://your-provider.com/v1"
export OPENAI_API_KEY="your-key"
```

### Router (System M) — No extra setup

The Router runs entirely in Python with no additional dependencies beyond `openai` and `pydantic`. It coordinates System A and System B automatically.

---

## ⚡ Quick Start

```bash
# Clone & install
git clone https://github.com/jgon9031-ai/yap-claw.git
cd yap-claw
pip install -e .

# Run a query (Router auto-selects System A or B)
python3 -c "
from amos import AMOS

yc = AMOS()
response = yc.run('What is the capital of France?')
print(response.text)
print('Routed to:', response.routing_decision.target)   # 'local' or 'cloud'
print('Reason:', response.routing_decision.reason)
"
```

---

## 🧠 Routing Logic (System M)

The Router evaluates every query through 4 sequential rules:

```python
# Rule 1: Privacy — sensitive queries always go to System B (local)
privacy_keywords = [
    "my ", "mine", "personal", "private", "secret",
    "password", "medical", "health record", "diary",
    "salary", "bank", "ssn", "home address", "family"
]
if any(kw in query.lower() for kw in privacy_keywords):
    → System B / local  (data never leaves the device)

# Rule 2: Past failure — escalate to System A if local keeps failing
if local_failure_rate(similar_queries) > 60%:
    → System A / cloud

# Rule 3: Complexity — analytical/research queries go to System A
complexity_keywords = [
    "analyze", "compare", "evaluate", "summarize",
    "explain in detail", "research", "pros and cons",
    "write a report", "translate", "debug", "refactor",
    "what are the implications", "step by step"
]
if complexity_score(query) > 0.7:
    → System A / cloud

# Rule 4: Default — prefer System B (cost-efficient, low latency)
    → System B / local
```

---

## 🧪 A/B Testing: System A vs System B

YAP-CLAW is designed for rigorous **A/B routing experiments**. Compare any two agents head-to-head:

```python
from amos import AMOS
from amos.executor import LocalExecutor, CloudExecutor

yc = AMOS(config={
    "local": {"model": "qwen2.5-coder:7b"},   # System B
    "cloud": {"model": "gpt-4o-mini"},          # System A
})

# Force System B
resp_b = yc.run("Summarize this text", force_target="local")

# Force System A  
resp_a = yc.run("Summarize this text", force_target="cloud")

# Let the Router decide (A/B auto-routing)
resp_auto = yc.run("Summarize this text")

print(f"System B: {resp_b.text[:80]}  [{resp_b.latency_ms}ms]")
print(f"System A: {resp_a.text[:80]}  [{resp_a.latency_ms}ms]")
print(f"Router chose: {resp_auto.routing_decision.target}")

# View cumulative A/B stats
yc.dashboard().print()
```

---

## 📊 Performance

Experimental results (n=500 queries each):

| Method | Success Rate | Avg Latency | Cloud Usage |
|---|---|---|---|
| Baseline — System B only | 72.6% | 280ms | 0% |
| Random A/B routing | 83.6% | 1002ms | 47.4% |
| **YAP-CLAW (adaptive)** | **84.6% ↑** | **689ms ↓** | **27.0% ↓** |

> ✅ **+12% success** over local-only | **-31% latency** vs random | **-43% cloud cost** vs random

---

## 🔄 Self-Improving Loop

YAP-CLAW gets smarter as you use it — no retraining required:

```
Every query    →  Memory Layer  →  SessionAnalyzer    →  SkillHealthTracker  →  Router update
(record result)   (SQLite store)   (every 10 queries)    (failure detection)    (smarter next time)
```

Inspired by:
- **Hermes Agent** (NousResearch) — asynchronous background session analysis pattern
- **cognee-skills** — observe → inspect → amend → evaluate self-healing loop

---

## 📈 Dashboard

```bash
python3 examples/dashboard_demo.py
```

```
═══════════════════════════════════════════
  YAP-CLAW Routing Dashboard
═══════════════════════════════════════════
  Total queries:     142
  System B (local):   89 (62.7%)
  System A (cloud):   53 (37.3%)

  Success rates:
    System B:  82.0%  ████████░░
    System A:  92.5%  █████████░

  Avg latency:
    System B:   312ms
    System A:  1829ms

  Top routing reasons:
    default (→ B)      : 83 queries
    complexity (→ A)   : 36 queries
    privacy (→ B)      : 19 queries
    past_failure (→ A) :  4 queries
═══════════════════════════════════════════
```

---

## 📦 Project Structure

```
yap-claw/
├── amos/
│   ├── harness.py      # Main YAP-CLAW orchestrator
│   ├── router.py       # System M — routing logic
│   ├── memory.py       # SQLite + FTS5 experience store
│   ├── executor.py     # System A (cloud) + System B (local) executors
│   ├── analyzer.py     # Hermes-style session analyzer
│   ├── health.py       # cognee-style skill health tracker
│   ├── dashboard.py    # CLI A/B routing statistics dashboard
│   ├── exceptions.py   # Error hierarchy
│   └── models.py       # Pydantic data models
├── examples/
│   ├── basic_usage.py      # Simple routing demo
│   └── dashboard_demo.py   # A/B stats dashboard demo
└── tests/                  # 80 tests, all passing
```

---

## 🛣️ Roadmap

| Version | Status | Features |
|---|---|---|
| v0.1 | ✅ Done | Router (System M), Memory, Executor |
| v0.2 | ✅ Done | SessionAnalyzer (Hermes), SkillHealthTracker (cognee) |
| v0.3 | ✅ Done | Ollama real integration (System B), CLI Dashboard |
| v0.4 | 📋 TBD | TBD |
| v0.5 | 📋 TBD | TBD |

---

<div align="center">

*YAP-CLAW: Because your AI shouldn't have to shout at the cloud for every little thing.* 🐾

**[GitHub](https://github.com/jgon9031-ai/yap-claw)** · **[Issues](https://github.com/jgon9031-ai/yap-claw/issues)**

</div>
