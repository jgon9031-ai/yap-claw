<div align="center">

```
 __   __ _   _   ____     ____  _        _    __        __
 \ \ / // \ | | |  _ \   / ___|| |      / \   \ \      / /
  \ V // _ \| | | |_) | | |    | |     / _ \   \ \ /\ / / 
   | |/ ___ | |_|  __/  | |___ | |___ / ___ \   \ V  V /  
   |_/_/   \_\___| |      \____||_____/_/   \_\   \_/\_/   
                  |_|                                        
```

**Yappy Agent Picking Cloud or Local Automatically & Wisely**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-80%20passed-brightgreen?logo=pytest)](tests/)
[![Version](https://img.shields.io/badge/version-0.3.0-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2603.15381-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.15381)
[![GitHub](https://img.shields.io/badge/GitHub-yap--claw-181717?logo=github)](https://github.com/jgon9031-ai/yap-claw)

*Drop a query. YAP-CLAW figures out where to send it — local or cloud — and gets smarter every time.* 🐾

</div>

---

## What is YAP-CLAW?

YAP-CLAW is a **self-improving hybrid AI routing harness** that intelligently dispatches queries between on-device (Ollama) and cloud (Claude/GPT) models. Unlike static routers, YAP-CLAW remembers past successes and failures — and adapts its routing decisions over time without retraining.

Inspired by the cognitive science paper **["Why AI systems don't learn and what to do about it"](https://arxiv.org/abs/2603.15381)** (META FAIR, NYU, UC Berkeley, 2026):

| Paper Concept | YAP-CLAW Layer |
|---|---|
| System A — Learning from observation | ☁️ Cloud models (Claude/GPT) |
| System B — Learning from action | 🖥️ Local models (Ollama) |
| System M — Meta-controller | 🧠 YAP-CLAW Router |
| Persistent memory | 💾 SQLite Experience Store |

---

## Architecture

```
              ╔══════════════════════════════════════╗
              ║         YAP-CLAW Harness             ║
              ║                                      ║
  query  ───► ║  ┌─────────────────────────────┐    ║
              ║  │     🧠 System M (Router)     │    ║
              ║  │                             │    ║
              ║  │  1. 🔒 Privacy check        │    ║
              ║  │  2. 📉 Past failure rate    │    ║
              ║  │  3. 🧮 Complexity score     │    ║
              ║  │  4. 💾 Memory hints         │    ║
              ║  └──────────┬──────────────────┘    ║
              ║             │                        ║
              ║      ┌──────┴──────┐                ║
              ║      ▼             ▼                ║
              ║  ┌────────┐  ┌──────────┐           ║
              ║  │ Sys B  │  │  Sys A   │           ║
              ║  │ Ollama │  │ Claude/  │           ║
              ║  │ Local  │  │   GPT    │           ║
              ║  └───┬────┘  └────┬─────┘           ║
              ║      └──────┬──────┘                ║
              ║             ▼                        ║
              ║  ┌──────────────────────┐           ║
              ║  │  💾 Memory Layer     │           ║
              ║  │  SQLite + FTS5       │           ║
              ║  │  (gets smarter 📈)  │           ║
              ║  └──────────────────────┘           ║
              ╚══════════════════════════════════════╝
```

---

## ⚡ Quick Start

```bash
# Clone & install
git clone https://github.com/jgon9031-ai/yap-claw.git
cd yap-claw
pip install -e .

# Run a query
python3 -c "
from amos import AMOS
yc = AMOS()
response = yc.run('What is the capital of France?')
print(response.text)
print('Routed to:', response.routing_decision.target)
"
```

---

## 🧠 Routing Logic (System M)

YAP-CLAW routes every query through 4 layers:

```python
# 1. Privacy — personal data stays local
if "my weight" in query or "내 체중" in query:
    → local (always)

# 2. Past failure — cloud if local keeps failing
if local_failure_rate > 60%:
    → cloud

# 3. Complexity — long/analytical queries go cloud
if complexity_score(query) > 0.7:
    → cloud

# 4. Default — local first (cost & latency)
    → local
```

**Personal keywords (→ local):** `나의`, `내`, `my`, `personal`, `체중`, `식단`, `건강`

**Complexity keywords (→ cloud):** `analyze`, `compare`, `research`, `분석`, `비교`, `조사`

---

## 📊 Performance

Experimental results (n=500 queries each):

| Method | Success Rate | Avg Latency | Cloud Usage |
|---|---|---|---|
| Baseline (Local Only) | 72.6% | 280ms | 0% |
| Random Routing | 83.6% | 1002ms | 47.4% |
| **YAP-CLAW** | **84.6% ↑** | **689ms ↓** | **27.0% ↓** |

> ✅ **+12% success** over baseline | **-31% latency** vs random | **-43% cloud cost** vs random

---

## 🔄 Self-Improving Loop

YAP-CLAW gets smarter as you use it:

```
Raw Experience  →  SessionAnalyzer  →  SkillHealthTracker  →  Router Update
(every query)      (every 10 turns)    (failure detection)     (no retraining)
```

Inspired by:
- **Hermes Agent** (NousResearch) — background session analysis pattern
- **cognee-skills** — observe → inspect → amend → evaluate loop

---

## 📦 Project Structure

```
yap-claw/
├── amos/
│   ├── harness.py      # Main AMOS/YAP-CLAW class
│   ├── router.py       # System M routing logic
│   ├── memory.py       # SQLite experience store
│   ├── executor.py     # Local (Ollama) + Cloud executors
│   ├── analyzer.py     # Hermes-style session analyzer
│   ├── health.py       # cognee-style skill health tracker
│   ├── dashboard.py    # CLI routing statistics dashboard
│   ├── exceptions.py   # Error hierarchy
│   └── models.py       # Pydantic data models
├── examples/
│   ├── basic_usage.py
│   └── dashboard_demo.py
└── tests/              # 80 tests, all passing
```

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
  Local queries:      89 (62.7%)
  Cloud queries:      53 (37.3%)

  Success rates:
    Local:   82.0%  ████████░░
    Cloud:   92.5%  █████████░

  Avg latency:
    Local:    312ms
    Cloud:   1829ms

  Top routing reasons:
    default         : 83 queries
    complexity      : 36 queries
    privacy         : 19 queries
    past_failure    : 4 queries
═══════════════════════════════════════════
```

---

## 🛣️ Roadmap

| Version | Status | Features |
|---|---|---|
| v0.1 | ✅ Done | Router, Memory, Executor |
| v0.2 | ✅ Done | SessionAnalyzer (Hermes), SkillHealthTracker (cognee) |
| v0.3 | ✅ Done | Ollama real integration, CLI Dashboard |
| v0.4 | 📋 TBD | TBD |
| v0.5 | 📋 TBD | TBD |

---

---

<div align="center">

*YAP-CLAW: Because your AI shouldn't have to shout at the cloud for every little thing.* 🐾

**[GitHub](https://github.com/jgon9031-ai/yap-claw)** · **[Issues](https://github.com/jgon9031-ai/yap-claw/issues)**

</div>
