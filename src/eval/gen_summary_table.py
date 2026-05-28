"""
Generate a Markdown summary table from all stealth evaluation results in eval/out/.

Usage:
    cd src
    python -m eval.gen_summary_table
    python -m eval.gen_summary_table --out-dir eval/out --output eval/out/summary.md
"""

import json
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_all_results(out_dir):
    out_dir = Path(out_dir)
    results = []
    for json_path in sorted(out_dir.glob("*/stealth_eval.json")):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data["_subdir"] = json_path.parent.name
        results.append(data)
    return results


def parse_subdir_name(name):
    parts = name.split("_")
    if len(parts) >= 4:
        variant = parts[0]
        poison_rate = parts[-1]
        victim = parts[-2]
        attack_type = "_".join(parts[1:-2])
        return variant, attack_type, victim, poison_rate
    return "", name, "", ""


def generate_markdown(results):
    lines = []
    lines.append("# Attack Stealth Evaluation Summary")
    lines.append("")
    lines.append("Model: Sentence-BERT (all-mpnet-base-v2)")
    lines.append("")

    lines.append("## Overall Similarity")
    lines.append("")
    lines.append(
        "| Attack Type | Poison Rate | N "
        "| Claim-Before | Claim-After | Before-After | Delta |"
    )
    lines.append(
        "|-------------|-------------|---"
        "|--------------|-------------|--------------|-------|"
    )

    for r in results:
        _, attack_type, _, pr = parse_subdir_name(r["_subdir"])
        o = r["overall"]
        lines.append(
            "| %s | %s | %d "
            "| %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f | %+.4f |"
            % (
                attack_type, pr, r["total_claims"],
                o["avg_sim_claim_before_justification"],
                o["std_sim_claim_before_justification"],
                o["avg_sim_claim_after_justification"],
                o["std_sim_claim_after_justification"],
                o["avg_sim_before_after_justification"],
                o["std_sim_before_after_justification"],
                o["avg_delta_similarity"],
            )
        )

    lines.append("")

    lines.append("## Before-After Similarity by Attack Outcome")
    lines.append("")
    lines.append(
        "| Attack Type | Poison Rate "
        "| Success (n) | FC Fail (n) | No Effect (n) |"
    )
    lines.append(
        "|-------------|-------------"
        "|-------------|-------------|---------------|"
    )

    for r in results:
        _, attack_type, _, pr = parse_subdir_name(r["_subdir"])
        by = r["by_attack_outcome"]
        s = by["attack_success"]
        f = by["fact_check_fail_only"]
        n = by["no_effect"]
        lines.append(
            "| %s | %s "
            "| %.4f (%d) | %.4f (%d) | %.4f (%d) |"
            % (
                attack_type, pr,
                s["avg_sim_before_after"], s["count"],
                f["avg_sim_before_after"], f["count"],
                n["avg_sim_before_after"], n["count"],
            )
        )

    lines.append("")

    lines.append("## Attack Effectiveness Context")
    lines.append("")
    lines.append("| Attack Type | Poison Rate | Total | ASR | SFR |")
    lines.append("|-------------|-------------|-------|-----|-----|")

    for r in results:
        _, attack_type, _, pr = parse_subdir_name(r["_subdir"])
        total = r["total_claims"]
        asr = r["attack_success_count"] / total if total else 0
        sfr = r["fact_check_fail_count"] / total if total else 0
        lines.append(
            "| %s | %s | %d | %.2f%% | %.2f%% |"
            % (attack_type, pr, total, asr * 100, sfr * 100)
        )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown summary from stealth eval results",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory with experiment subdirs (default: eval/out/)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output Markdown file (default: <out-dir>/summary.md)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.out_dir is None:
        out_dir = Path(__file__).parent / "out"
    else:
        out_dir = Path(args.out_dir)

    results = load_all_results(out_dir)
    if not results:
        logger.error("No stealth_eval.json found in %s", out_dir)
        return

    logger.info("Loaded %d experiment results", len(results))
    md = generate_markdown(results)

    if args.output is None:
        output_path = out_dir / "summary.md"
    else:
        output_path = Path(args.output)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print(md)
    print("\nSaved to: %s" % output_path)


if __name__ == "__main__":
    main()
