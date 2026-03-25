"""
NEST Seed Script — Conversation Pattern Importer
=================================================
Seeds the NEST memory layer with routing patterns learned from
real conversations between Jay Kim and OpenClaw.

Run:
    python3 scripts/seed_nest_from_conversation.py [--db ~/.amos/memory.db] [--dry-run]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from amos.memory import MemoryLayer


# ─────────────────────────────────────────────────────────────────────────────
# Conversation-derived patterns
# Each entry: (query_text, target, model_used, success, latency_ms, count)
# count = how many times this pattern appeared (records multiple experiences)
# ─────────────────────────────────────────────────────────────────────────────

PATTERNS: list[tuple[str, str, str, bool, int, int]] = [

    # ── Personal / Health data → PAW (always local) ──────────────────────────
    ("76.5",                            "local", "qwen2.5-coder:7b", True,  180, 5),
    ("몸무게는 76.1",                   "local", "qwen2.5-coder:7b", True,  190, 3),
    ("체중",                            "local", "qwen2.5-coder:7b", True,  175, 8),
    ("weight is",                       "local", "qwen2.5-coder:7b", True,  180, 4),
    ("점심 저녁 식단",                  "local", "qwen2.5-coder:7b", True,  210, 6),
    ("점심: 라면 저녁: 떡볶이",         "local", "qwen2.5-coder:7b", True,  215, 4),
    ("점심 돈카츠카레 저녁 포케",       "local", "qwen2.5-coder:7b", True,  220, 3),
    ("칼로리 기록",                     "local", "qwen2.5-coder:7b", True,  200, 4),
    ("오늘 식단",                       "local", "qwen2.5-coder:7b", True,  195, 5),
    ("운동 웨이트트레이닝",             "local", "qwen2.5-coder:7b", True,  185, 3),
    ("스크린골프 소모",                 "local", "qwen2.5-coder:7b", True,  190, 3),
    ("내 사주",                         "local", "qwen2.5-coder:7b", True,  200, 2),
    ("이명 귀마개",                     "local", "qwen2.5-coder:7b", True,  195, 2),
    ("고지혈약 복용",                   "local", "qwen2.5-coder:7b", True,  185, 2),
    ("로수바스타틴",                    "local", "qwen2.5-coder:7b", True,  190, 2),
    ("my weight",                       "local", "qwen2.5-coder:7b", True,  180, 3),
    ("personal health",                 "local", "qwen2.5-coder:7b", True,  175, 2),
    ("diet log",                        "local", "qwen2.5-coder:7b", True,  185, 2),

    # ── File / System ops → PAW ───────────────────────────────────────────────
    ("그래프 보내줘",                   "local", "qwen2.5-coder:7b", True,  250, 4),
    ("PNG 전송",                        "local", "qwen2.5-coder:7b", True,  230, 3),
    ("weight_plot.png",                 "local", "qwen2.5-coder:7b", True,  220, 4),
    ("파일 전송",                       "local", "qwen2.5-coder:7b", True,  240, 3),
    ("memory 파일 기록",                "local", "qwen2.5-coder:7b", True,  210, 5),
    ("cron list",                       "local", "qwen2.5-coder:7b", True,  300, 5),
    ("cron remove",                     "local", "qwen2.5-coder:7b", True,  280, 4),
    ("openclaw status",                 "local", "qwen2.5-coder:7b", True,  260, 3),
    ("git commit push",                 "local", "qwen2.5-coder:7b", True,  350, 5),
    ("github push",                     "local", "qwen2.5-coder:7b", True,  370, 4),
    ("send telegram",                   "local", "qwen2.5-coder:7b", True,  280, 4),
    ("agentmail send",                  "local", "qwen2.5-coder:7b", True,  310, 3),
    ("notion 페이지 생성",              "local", "qwen2.5-coder:7b", True,  400, 4),
    ("노션에 올려줘",                   "local", "qwen2.5-coder:7b", True,  420, 3),

    # ── Research / Analysis → CLAW ────────────────────────────────────────────
    ("논문 요약해줘",                   "cloud", "gpt-4o",           True, 1800, 5),
    ("논문 분석",                       "cloud", "gpt-4o",           True, 1750, 4),
    ("arXiv",                           "cloud", "gpt-4o",           True, 1900, 4),
    ("연구해줘",                        "cloud", "gpt-4o",           True, 2100, 5),
    ("조사해서 알려줘",                 "cloud", "gpt-4o",           True, 1950, 6),
    ("고지혈증약이 이명에 미치는 영향", "cloud", "gpt-4o",           True, 2000, 2),
    ("스타틴 이명 연구",               "cloud", "gpt-4o",           True, 1900, 2),
    ("PID 알고리즘 SLO",               "cloud", "gpt-4o",           True, 2200, 2),
    ("compensation 설계",               "cloud", "gpt-4o",           True, 2100, 2),
    ("AutoResearchClaw",               "cloud", "gpt-4o",           True, 2400, 5),
    ("research paper generate",        "cloud", "gpt-4o",           True, 2500, 4),
    ("이 논문 구현 가능한가",           "cloud", "gpt-4o",           True, 1850, 2),
    ("AI 하네스 설계",                  "cloud", "gpt-4o",           True, 2000, 3),

    # ── Recommendations / Search → CLAW ──────────────────────────────────────
    ("강남역 식당 추천",               "cloud", "gpt-4o",           True, 1600, 3),
    ("이자카야 추천",                   "cloud", "gpt-4o",           True, 1550, 2),
    ("맛집 추천",                       "cloud", "gpt-4o",           True, 1500, 4),
    ("날씨 알려줘",                     "cloud", "gpt-4o",           True,  900, 3),
    ("공공 API",                        "cloud", "gpt-4o",           True, 1400, 2),
    ("LinkedIn 요약",                   "cloud", "gpt-4o",           True, 2200, 6),
    ("Daum 트렌드",                     "cloud", "gpt-4o",           True, 1300, 4),
    ("X 포스트 초안",                   "cloud", "gpt-4o",           True, 1700, 5),
    ("주식 예측",                       "cloud", "gpt-4o",           True, 1800, 4),
    ("코스닥 추천",                     "cloud", "gpt-4o",           True, 1750, 4),
    ("하이닉스 예측",                   "cloud", "gpt-4o",           True, 1900, 3),

    # ── Coding / Build → CLAW ────────────────────────────────────────────────
    ("구현해줘",                        "cloud", "gpt-4o",           True, 2300, 6),
    ("코드 짜줘",                       "cloud", "gpt-4o",           True, 2100, 5),
    ("v0.3 도 구현해",                  "cloud", "gpt-4o",           True, 2500, 2),
    ("README 꾸며줘",                   "cloud", "gpt-4o",           True, 1800, 3),
    ("로고 이미지 생성",               "cloud", "gpt-4o",           True, 3000, 2),
    ("build AMOS harness",              "cloud", "gpt-4o",           True, 2800, 3),
    ("implement router",               "cloud", "gpt-4o",           True, 2400, 3),
    ("PDF 만들어서 보내줘",            "cloud", "gpt-4o",           True, 2200, 3),
    ("DOCX 변환",                       "cloud", "gpt-4o",           True, 2100, 2),
    ("arXiv 제출 패키지",              "cloud", "gpt-4o",           True, 1900, 2),

    # ── Quick system commands → PAW ───────────────────────────────────────────
    ("업데이트 진행",                   "local", "qwen2.5-coder:7b", True,  400, 2),
    ("cron 제거해줘",                   "local", "qwen2.5-coder:7b", True,  350, 3),
    ("설정 변경",                       "local", "qwen2.5-coder:7b", True,  320, 3),
    ("openclaw cron add",               "local", "qwen2.5-coder:7b", True,  380, 4),
    ("researcher 모델 변경",           "local", "qwen2.5-coder:7b", True,  310, 2),
]


def seed(db_path: str, dry_run: bool = False) -> None:
    mem = MemoryLayer(db_path=db_path)
    total = 0

    print(f"{'[DRY RUN] ' if dry_run else ''}Seeding NEST at {db_path}")
    print(f"Patterns to import: {len(PATTERNS)}")

    for query_text, target, model_used, success, latency_ms, count in PATTERNS:
        for _ in range(count):
            if not dry_run:
                mem.record(
                    query_text=query_text,
                    target=target,
                    model_used=model_used,
                    success=success,
                    latency_ms=latency_ms,
                )
            total += 1

    stats = mem.stats() if not dry_run else {}
    print(f"\n{'Would import' if dry_run else 'Imported'}: {total} experience records")
    if stats:
        print(f"NEST total records : {stats.get('total_records', 0)}")
        print(f"Local success rate : {stats.get('local_success_rate', 0):.1%}")
        print(f"Cloud success rate : {stats.get('cloud_success_rate', 0):.1%}")

    # Spot-check routing
    if not dry_run:
        from amos.router import AMOSRouter
        router = AMOSRouter(memory=mem)
        from amos.models import Query
        tests = [
            ("몸무게는 76.5",            "local"),
            ("점심 라면 저녁 떡볶이",    "local"),
            ("이 논문 요약해줘",          "cloud"),
            ("강남역 맛집 추천",          "cloud"),
            ("git push origin master",    "local"),
            ("AMOS 구현해줘",             "cloud"),
        ]
        print("\n── Routing spot-check ────────────────────────────")
        for q, expected in tests:
            decision = router.route(Query(text=q))
            status = "✅" if decision.target == expected else "❌"
            print(f"  {status}  '{q[:30]:30s}' → {decision.target:5s} [{decision.reason}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed NEST from conversation patterns")
    parser.add_argument("--db", default="~/.amos/memory.db", help="NEST DB path")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()
    seed(db_path=args.db, dry_run=args.dry_run)
