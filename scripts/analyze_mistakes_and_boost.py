"""
Mistake Pattern Analyzer & Priority Booster
=============================================
Analyzes repeated routing mistakes from conversation history,
ranks them by frequency, and re-seeds NEST with boosted weights
so YAP prioritizes learning the most common errors first.

Mistake types:
  - WRONG_TARGET:    YAP sent to wrong agent (e.g., cloud when should be local)
  - MISSING_KEYWORD: query had no matching keyword but correct target known
  - PRIVACY_LEAK:    personal data was routed to cloud (critical!)

Run:
    python3 scripts/analyze_mistakes_and_boost.py [--dry-run]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))
from amos.memory import MemoryLayer
from amos.router import AMOSRouter
from amos.models import Query


# ─────────────────────────────────────────────────────────────────────────────
# Known mistakes from conversation history (Jay <-> OpenClaw)
# Format: (query, correct_target, mistake_type, times_occurred)
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_MISTAKES: list[tuple[str, str, str, int]] = [

    # ── PRIVACY LEAKS (Critical — personal data must NEVER go to cloud) ─────
    ("체중 그래프 보내줘",            "local", "PRIVACY_LEAK",    5),
    ("weight plot PNG",              "local", "PRIVACY_LEAK",    4),
    ("오늘 몸무게",                   "local", "PRIVACY_LEAK",    6),
    ("식단 기록해줘",                 "local", "PRIVACY_LEAK",    5),
    ("칼로리 합계",                   "local", "PRIVACY_LEAK",    4),
    ("내 건강 기록",                  "local", "PRIVACY_LEAK",    3),
    ("운동 소모 칼로리",              "local", "PRIVACY_LEAK",    3),
    ("사주 운세",                     "local", "PRIVACY_LEAK",    3),
    ("이명 증상 기록",                "local", "PRIVACY_LEAK",    2),

    # ── WRONG TARGET: sent to cloud but should be local ───────────────────
    ("cron 추가해줘",                 "local", "WRONG_TARGET",   4),
    ("openclaw 설정 변경",            "local", "WRONG_TARGET",   3),
    ("memory 파일 업데이트",          "local", "WRONG_TARGET",   4),
    ("git add commit",               "local", "WRONG_TARGET",   5),
    ("파일 전송 텔레그램",            "local", "WRONG_TARGET",   4),
    ("agentmail로 보내줘",            "local", "WRONG_TARGET",   3),
    ("notion에 업로드",               "local", "WRONG_TARGET",   4),
    ("cron remove",                  "local", "WRONG_TARGET",   4),
    ("openclaw gateway restart",     "local", "WRONG_TARGET",   3),
    ("send via telegram",            "local", "WRONG_TARGET",   3),

    # ── WRONG TARGET: sent to local but should be cloud ───────────────────
    ("추천해줘",                      "cloud", "WRONG_TARGET",   5),
    ("알려줘",                        "cloud", "WRONG_TARGET",   6),  # ambiguous — context matters
    ("논문 내용",                     "cloud", "WRONG_TARGET",   4),
    ("이거 분석해",                   "cloud", "WRONG_TARGET",   4),
    ("구현 방안 수립",                "cloud", "WRONG_TARGET",   3),
    ("설계해줘",                      "cloud", "WRONG_TARGET",   3),
    ("방법이 있을까",                 "cloud", "WRONG_TARGET",   4),
    ("어떻게 하면 좋을지",            "cloud", "WRONG_TARGET",   5),
    ("비교해줘",                      "cloud", "WRONG_TARGET",   4),
    ("정리해줘",                      "cloud", "WRONG_TARGET",   4),
    ("요약해줘",                      "cloud", "WRONG_TARGET",   6),

    # ── MISSING KEYWORD: query not matched by any rule, but correct known ──
    ("오늘거 수동으로 만들어서 보내줘", "cloud", "MISSING_KEYWORD", 3),
    ("실패에서 배우는게 없니",         "cloud", "MISSING_KEYWORD", 2),
    ("진행 상황은",                    "local", "MISSING_KEYWORD", 4),
    ("상황 확인해줘",                  "local", "MISSING_KEYWORD", 4),
    ("완료되었나",                     "local", "MISSING_KEYWORD", 3),
    ("어디서 멈췄어",                  "local", "MISSING_KEYWORD", 2),
    ("처음부터 다시 해",               "cloud", "MISSING_KEYWORD", 2),
    ("수동으로 전달 해줘",             "cloud", "MISSING_KEYWORD", 3),
    ("진행해보자",                     "cloud", "MISSING_KEYWORD", 3),
    ("이거 참고해",                    "cloud", "MISSING_KEYWORD", 3),
    ("업데이트 진행",                  "local", "MISSING_KEYWORD", 2),
    ("repo 이름 바꿔줘",               "local", "MISSING_KEYWORD", 2),
    ("push해줘",                      "local", "MISSING_KEYWORD", 3),
    ("readme 꾸며줘",                  "cloud", "MISSING_KEYWORD", 2),
    ("이메일 보내줘",                  "local", "MISSING_KEYWORD", 4),
    ("pdf로 만들어서",                 "cloud", "MISSING_KEYWORD", 3),
    ("이미지 생성해서 교체",           "cloud", "MISSING_KEYWORD", 2),
]


@dataclass
class MistakeStats:
    query: str
    correct_target: str
    mistake_type: str
    count: int
    boost_factor: int = field(init=False)

    def __post_init__(self) -> None:
        # Priority boost:
        # PRIVACY_LEAK: x5 (critical — must never route personal data to cloud)
        # WRONG_TARGET: x3
        # MISSING_KEYWORD: x2
        multipliers = {"PRIVACY_LEAK": 5, "WRONG_TARGET": 3, "MISSING_KEYWORD": 2}
        base = multipliers.get(self.mistake_type, 2)
        self.boost_factor = self.count * base


def analyze_and_boost(db_path: str, dry_run: bool = False) -> None:
    mem = MemoryLayer(db_path=db_path)
    router = AMOSRouter(memory=mem)

    mistakes = [MistakeStats(q, t, mt, c) for q, t, mt, c in KNOWN_MISTAKES]
    mistakes.sort(key=lambda m: m.boost_factor, reverse=True)

    # ── Report ──────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  YAP-CLAW — Mistake Pattern Analysis & Priority Boost")
    print("=" * 65)

    type_counts = Counter(m.mistake_type for m in mistakes)
    total_boosts = sum(m.boost_factor for m in mistakes)
    print(f"\nMistake types found:")
    for mt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        icon = "🚨" if mt == "PRIVACY_LEAK" else "❌" if mt == "WRONG_TARGET" else "⚠️"
        print(f"  {icon}  {mt:20s}: {cnt} patterns")

    print(f"\nTop 10 priority boosts (by impact):")
    print(f"  {'Query':35s} {'Target':6s} {'Type':18s} {'Boost':>5s}")
    print(f"  {'-'*35} {'-'*6} {'-'*18} {'-'*5}")
    for m in mistakes[:10]:
        print(f"  {m.query[:34]:35s} {m.correct_target:6s} {m.mistake_type:18s} x{m.boost_factor:3d}")

    print(f"\nTotal boost records to inject: {total_boosts}")

    if dry_run:
        print("\n[DRY RUN] — No changes written to NEST.")
        return

    # ── Inject ──────────────────────────────────────────────────────────────
    injected = 0
    for m in mistakes:
        for _ in range(m.boost_factor):
            mem.record(
                query_text=m.query,
                target=m.correct_target,
                model_used="qwen2.5-coder:7b" if m.correct_target == "local" else "gpt-4o",
                success=True,
                latency_ms=200 if m.correct_target == "local" else 1600,
            )
            injected += 1

    print(f"\nInjected {injected} boosted records into NEST ✅")

    # ── Validate routing improved ───────────────────────────────────────────
    print("\n── Routing validation (hardest cases) ─────────────────────────")
    hard_cases = [
        ("체중 그래프 보내줘",              "local"),
        ("식단 기록해줘",                   "local"),
        ("요약해줘",                        "cloud"),
        ("추천해줘",                        "cloud"),
        ("어떻게 하면 좋을지",              "cloud"),
        ("git add commit",                 "local"),
        ("agentmail로 보내줘",              "local"),
        ("이메일 보내줘",                   "local"),
        ("pdf로 만들어서",                  "cloud"),
        ("진행 상황은",                     "local"),
    ]
    correct = 0
    for q, expected in hard_cases:
        d = router.route(Query(text=q))
        ok = "✅" if d.target == expected else "❌"
        if d.target == expected:
            correct += 1
        print(f"  {ok} '{q[:30]:30s}' → {d.target:5s} | {d.reason[:38]}")

    print(f"\n  Score: {correct}/{len(hard_cases)}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze mistakes and boost NEST learning")
    parser.add_argument("--db", default="~/.amos/memory.db")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    analyze_and_boost(db_path=args.db, dry_run=args.dry_run)
