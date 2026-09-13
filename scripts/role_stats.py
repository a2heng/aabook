#!/usr/bin/env python3
"""Print/save the role distribution of a built script.

Only speaking roles are counted (narrator + characters with dialogue/monologue).
Named characters that never speak are listed separately, even if mentioned often.

Examples:
    python scripts/role_stats.py outputs/mybook/script.csv --cast outputs/mybook/cast.json
    python scripts/role_stats.py outputs/mybook/script.csv --cast outputs/mybook/cast.json --out outputs/mybook/role_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from audiobook.canonical import canonicalize_rows  # noqa: E402
from audiobook.schema import Cast, read_script  # noqa: E402
from audiobook.stats import distribution_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Role distribution for an audiobook script")
    parser.add_argument("script", help="path to script.csv")
    parser.add_argument("--cast", default=None, help="path to cast.json (enables mention counts)")
    parser.add_argument("--out", default=None, help="write role_stats.json here")
    parser.add_argument("--top", type=int, default=0, help="only show the top N speaking roles (0 = all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_script(args.script)
    cast = Cast.load(args.cast) if args.cast else None
    canonicalize_rows(rows, cast)
    report = distribution_report(rows, cast)
    speaking = report["speaking_roles"]
    if args.top:
        speaking = speaking[: args.top]

    print(f"total   : {report['total_minutes']} min ({report['total_seconds']} s) across {len(rows)} rows")
    print(f"{'#':>2}  {'role':<16} {'rows':>5} {'dial':>5} {'min':>6} {'ch':>4} {'mention':>7}")
    for rank, stat in enumerate(speaking, start=1):
        total = stat["dialogue_rows"] + stat["monologue_rows"] + stat["narration_rows"]
        print(
            f"{rank:>2}  {stat['name']:<16} {total:>5} {stat['dialogue_rows']:>5} "
            f"{stat['est_minutes']:>6} {stat['chapters']:>4} {stat['mentions']:>7}"
        )

    non_speaking = report["non_speaking_named"]
    if non_speaking:
        print("\nnamed but non-speaking (excluded):")
        for stat in non_speaking:
            print(f"    {stat['name']:<16} mentions={stat['mentions']}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
