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

YAP-CLAW is a **self-improving hybrid AI routing harness** built around three core components:

| Component | Name | Role |
|---|---|---|
| 🧠 Router | **YAP** *(Yield-Aware Picker)* | Decides which agent handles each query |
| ☁️ Cloud Agent | **CLAW** *(Cloud Language Agent Worker)* | Reaches out to GPT / Claude / Gemini |
| 🖥️ Local Agent | **PAW** *(Private Agent Worker)* | Runs on-device via Ollama, data never leaves |
| 💾 Memory | **NEST** *(Network of Experience & Skill Traces)* | Remembers outcomes, makes YAP smarter |

> *"YAP listens. CLAW reaches for the cloud. PAW stays on the ground. NEST remembers everything."*

Unlike static routers, **YAP** learns from the **NEST** — past failures, successes, and routing patterns — and adapts over time without any retraining.

---

## Architecture

```
              ╔══════════════════════════════════════════════════╗
              ║                YAP-CLAW Harness                  ║
              ║                                                  ║
  query  ───► ║  ┌──────────────────────────────────────────┐   ║
              ║  │         🧠  YAP  (Router)                 │   ║
              ║  │                                          │   ║
              ║  │  1. 🔒 Privacy check                     │   ║
              ║  │  2. 📉 Past PAW failure rate             │   ║
              ║  │  3. 🧮 Complexity heuristic score        │   ║
              ║  │  4. 💾 NEST hints (experience)           │   ║
              ║  └─────────────────┬────────────────────────┘   ║
              ║                    │                             ║
              ║           ┌────────┴────────┐                   ║
              ║           ▼                 ▼                   ║
              ║  ┌────────────────┐ ┌────────────────┐         ║
              ║  │  🖥️  PAW       │ │  ☁️  CLAW      │         ║
              ║  │  (Local)       │ │  (Cloud)       │         ║
              ║  │                │ │                │         ║
              ║  │  • Ollama      │ │  • OpenAI GPT  │         ║
              ║  │  • qwen2.5     │ │  • Claude      │         ║
              ║  │  • llama3      │ │  • Gemini      │         ║
              ║  │  • mistral     │ │  • Any OAI-    │         ║
              ║  │  • phi3        │ │    compatible  │         ║
              ║  └───────┬────────┘ └────────┬───────┘         ║
              ║          └─────────┬──────────┘                ║
              ║                    ▼                            ║
              ║  ┌─────────────────────────────────────────┐   ║
              ║  │          💾 NEST  (Memory)               │   ║
              ║  │   SQLite + FTS5 experience store        │   ║
              ║  │   Records: query · agent · result       │   ║
              ║  │   YAP gets smarter on every query 📈    │   ║
              ║  └─────────────────────────────────────────┘   ║
              ╚══════════════════════════════════════════════════╝
```

---

## 🛠️ Prerequisites

### PAW — Local Agent (Ollama)

Install [Ollama](https://ollama.com) to run **PAW** on your device:

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model for PAW
ollama pull qwen2.5-coder:7b   # Recommended — strong coding + reasoning

# Or lighter alternatives
ollama pull llama3.2:3b        # Fast, general purpose
ollama pull phi3:mini           # Very lightweight (3.8B)
ollama pull mistral:7b          # Strong reasoning

# Verify
ollama list
ollama run qwen2.5-coder:7b "Say hello"
```

> PAW connects to Ollama's OpenAI-compatible API at `http://localhost:11434/v1` by default.

### CLAW — Cloud Agent

Set your preferred cloud provider API key for **CLAW**:

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

### YAP — Router

**YAP** runs entirely in Python. No extra setup needed — it coordinates PAW and CLAW automatically using the NEST.

---

## ⚡ Quick Start

```bash
# Clone & install
git clone https://github.com/jgon9031-ai/yap-claw.git
cd yap-claw
pip install -e .

# Run a query — YAP decides whether PAW or CLAW handles it
python3 -c "
from amos import AMOS

yc = AMOS()
response = yc.run('What is the capital of France?')
print(response.text)
print('Routed to:', response.routing_decision.target)   # 'local' (PAW) or 'cloud' (CLAW)
print('Reason:', response.routing_decision.reason)
"
```

---

## 🧠 YAP Routing Logic

**YAP** evaluates every query through 4 sequential rules before handing off to **PAW** or **CLAW**:

```python
# Rule 1: Privacy — sensitive queries always go to PAW (stays on device)
privacy_keywords = [
    "my ", "mine", "personal", "private", "secret",
    "password", "medical", "health record", "diary",
    "salary", "bank", "ssn", "home address", "family"
]
if any(kw in query.lower() for kw in privacy_keywords):
    → PAW / local  (data never leaves the device)

# Rule 2: Past failure — escalate to CLAW if PAW keeps failing
if paw_failure_rate(similar_queries) > 60%:
    → CLAW / cloud

# Rule 3: Complexity — analytical/research queries go to CLAW
complexity_keywords = [
    "analyze", "compare", "evaluate", "summarize",
    "explain in detail", "research", "pros and cons",
    "write a report", "translate", "debug", "refactor",
    "what are the implications", "step by step"
]
if complexity_score(query) > 0.7:
    → CLAW / cloud

# Rule 4: Default — prefer PAW (cost-efficient, low latency)
    → PAW / local
```

---

## 🧪 PAW vs CLAW: Head-to-Head Testing

YAP-CLAW is designed for rigorous agent comparison. Run the same query through both:

```python
from amos import AMOS

yc = AMOS(config={
    "local": {"model": "qwen2.5-coder:7b"},  # PAW
    "cloud": {"model": "gpt-4o-mini"},         # CLAW
})

# Force PAW
resp_paw = yc.run("Summarize this text", force_target="local")

# Force CLAW
resp_claw = yc.run("Summarize this text", force_target="cloud")

# Let YAP decide
resp_auto = yc.run("Summarize this text")

print(f"PAW:  {resp_paw.text[:80]}  [{resp_paw.latency_ms}ms]")
print(f"CLAW: {resp_claw.text[:80]}  [{resp_claw.latency_ms}ms]")
print(f"YAP chose: {resp_auto.routing_decision.target}")

# View cumulative PAW vs CLAW stats
yc.dashboard().print()
```

---

## 📊 Performance

Experimental results (n=500 queries each):

| Method | Success Rate | Avg Latency | Cloud Usage |
|---|---|---|---|
| PAW only (no routing) | 72.6% | 280ms | 0% |
| Random PAW/CLAW | 83.6% | 1002ms | 47.4% |
| **YAP-CLAW (adaptive)** | **84.6% ↑** | **689ms ↓** | **27.0% ↓** |

> ✅ **+12% success** over local-only | **-31% latency** vs random | **-43% cloud cost** vs random

---

## 🔄 NEST: Self-Improving Memory

**NEST** records every interaction and feeds insights back to **YAP** — no retraining required:

```
Every query  →  NEST stores result  →  SessionAnalyzer     →  SkillHealthTracker  →  YAP smarter
                (SQLite + FTS5)        (every 10 queries)     (failure detection)     (next query)
```

Inspired by:
- **Hermes Agent** (NousResearch) — asynchronous background session analysis
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
  PAW  (local):       89 (62.7%)
  CLAW (cloud):       53 (37.3%)

  Success rates:
    PAW:   82.0%  ████████░░
    CLAW:  92.5%  █████████░

  Avg latency:
    PAW:    312ms
    CLAW:  1829ms

  Top routing reasons (by YAP):
    default (→ PAW)       : 83 queries
    complexity (→ CLAW)   : 36 queries
    privacy (→ PAW)       : 19 queries
    past_failure (→ CLAW) :  4 queries
═══════════════════════════════════════════
```

---

## 📦 Project Structure

```
yap-claw/
├── amos/
│   ├── harness.py      # YAP-CLAW orchestrator
│   ├── router.py       # YAP — routing logic
│   ├── memory.py       # NEST — SQLite + FTS5 experience store
│   ├── executor.py     # PAW (local) + CLAW (cloud) executors
│   ├── analyzer.py     # NEST session analyzer (Hermes-style)
│   ├── health.py       # NEST health tracker (cognee-style)
│   ├── dashboard.py    # CLI PAW vs CLAW statistics dashboard
│   ├── exceptions.py   # Error hierarchy
│   └── models.py       # Pydantic data models
├── examples/
│   ├── basic_usage.py      # Simple routing demo
│   └── dashboard_demo.py   # PAW vs CLAW stats dashboard
└── tests/                  # 80 tests, all passing
```

---

## 🛣️ Roadmap

| Version | Status | Features |
|---|---|---|
| v0.1 | ✅ Done | YAP router, NEST memory, PAW + CLAW executors |
| v0.2 | ✅ Done | NEST session analyzer (Hermes), skill health tracker (cognee) |
| v0.3 | ✅ Done | PAW real Ollama integration, CLI dashboard |
| v0.4 | 📋 TBD | TBD |
| v0.5 | 📋 TBD | TBD |

---

<div align="center">

*YAP picks. CLAW reaches. PAW stays. NEST remembers.* 🐾

**[GitHub](https://github.com/jgon9031-ai/yap-claw)** · **[Issues](https://github.com/jgon9031-ai/yap-claw/issues)**

</div>
