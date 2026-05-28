"""
Semantic guard health check for SRSFact logs.

Usage examples:
  python -m eval.semantic_guard_health --logs-dir out/averitec/srsfact/gemini_25_flash/2026-03-23_18-31/logs
  python -m eval.semantic_guard_health --run-dir out/averitec/srsfact/gemini_25_flash/2026-03-23_18-31
  python -m eval.semantic_guard_health --auto-latest --base-dir out/averitec/srsfact/gemini_25_flash
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EVIDENCE_PATTERN = re.compile(
    r"Semantic guard on evidence: before=(\d+), after=(\d+)(?:, .*?bypass=([a-zA-Z0-9_]+))?"
)
QQUERY_PATTERN = re.compile(
    r"Semantic guard on question-query pairs: before=(\d+), after=(\d+)(?:, .*?bypass=([a-zA-Z0-9_]+))?"
)


@dataclass
class GuardStats:
    events: int = 0
    sum_before: int = 0
    sum_after: int = 0
    bypass_events: int = 0

    @property
    def keep_rate(self) -> float:
        return (self.sum_after / self.sum_before) if self.sum_before else 0.0

    @property
    def bypass_rate(self) -> float:
        return (self.bypass_events / self.events) if self.events else 0.0

    def to_dict(self) -> dict:
        return {
            "events": self.events,
            "sum_before": self.sum_before,
            "sum_after": self.sum_after,
            "keep_rate": self.keep_rate,
            "bypass_events": self.bypass_events,
            "bypass_rate": self.bypass_rate,
        }


def list_log_files(logs_dir: Path) -> Iterable[Path]:
    return sorted(logs_dir.glob("*.txt"))


def collect_stats(logs_dir: Path, pattern: re.Pattern) -> GuardStats:
    stats = GuardStats()
    for log_file in list_log_files(logs_dir):
        text = log_file.read_text(encoding="utf-8", errors="ignore")
        for match in pattern.finditer(text):
            before = int(match.group(1))
            after = int(match.group(2))
            bypass = match.group(3) or "none"
            stats.events += 1
            stats.sum_before += before
            stats.sum_after += after
            if bypass != "none":
                stats.bypass_events += 1
    return stats


def judge_evidence(stats: GuardStats) -> str:
    keep_rate = stats.keep_rate
    if keep_rate < 0.45:
        return "误伤风险高(keep_rate<0.45)"
    if keep_rate > 0.80:
        return "守卫偏宽松(keep_rate>0.80)"
    return "区间合理"


def hint_bypass(stats: GuardStats) -> str:
    bypass_rate = stats.bypass_rate
    if bypass_rate < 0.10:
        return "旁路偏少，可能仍偏严"
    if bypass_rate > 0.60:
        return "旁路偏多，守卫可能过松"
    return "旁路比例正常"


def find_latest_run_dir(base_dir: Path) -> Path:
    candidates = [p for p in base_dir.iterdir() if p.is_dir() and (p / "logs").exists()]
    if not candidates:
        raise FileNotFoundError(f"No run directories with logs found under: {base_dir}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def resolve_logs_dir(args) -> tuple[Path, Path | None]:
    if args.logs_dir:
        logs_dir = Path(args.logs_dir)
        run_dir = logs_dir.parent if logs_dir.name == "logs" else None
        return logs_dir, run_dir

    if args.run_dir:
        run_dir = Path(args.run_dir)
        return run_dir / "logs", run_dir

    if args.auto_latest:
        base_dir = Path(args.base_dir)
        run_dir = find_latest_run_dir(base_dir)
        return run_dir / "logs", run_dir

    raise ValueError("Please provide one of --logs-dir, --run-dir, or --auto-latest.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic guard keep_rate and bypass_rate from logs.")
    parser.add_argument("--logs-dir", type=str, default=None, help="Path to logs directory.")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to a single run directory containing logs/.")
    parser.add_argument("--auto-latest", action="store_true", help="Automatically pick the latest run under --base-dir.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="out/averitec/srsfact/gemini_25_flash",
        help="Base directory used with --auto-latest.",
    )
    parser.add_argument("--save-json", action="store_true", help="Save summary json to run_dir or logs_dir.")
    args = parser.parse_args()

    logs_dir, run_dir = resolve_logs_dir(args)
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    evidence = collect_stats(logs_dir, EVIDENCE_PATTERN)
    qquery = collect_stats(logs_dir, QQUERY_PATTERN)

    summary = {
        "logs_dir": str(logs_dir),
        "run_dir": str(run_dir) if run_dir else None,
        "evidence_guard": evidence.to_dict(),
        "qquery_guard": qquery.to_dict(),
        "judgement": judge_evidence(evidence),
        "bypass_hint": hint_bypass(evidence),
    }

    print(f"logs_dir={summary['logs_dir']}")
    print(f"evidence_events={evidence.events}")
    print(f"evidence_keep_rate={evidence.keep_rate:.4f}" if evidence.sum_before else "evidence_keep_rate=NA")
    print(f"evidence_bypass_rate={evidence.bypass_rate:.4f}" if evidence.events else "evidence_bypass_rate=NA")
    print(f"qquery_events={qquery.events}")
    print(f"qquery_keep_rate={qquery.keep_rate:.4f}" if qquery.sum_before else "qquery_keep_rate=NA")
    print(f"qquery_bypass_rate={qquery.bypass_rate:.4f}" if qquery.events else "qquery_bypass_rate=NA")
    print(f"judgement={summary['judgement']}")
    print(f"bypass_hint={summary['bypass_hint']}")

    if args.save_json:
        output_dir = run_dir if run_dir else logs_dir
        output_path = output_dir / "semantic_guard_health.json"
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json={output_path}")


if __name__ == "__main__":
    main()
