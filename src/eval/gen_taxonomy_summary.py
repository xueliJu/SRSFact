"""
Generate a Markdown summary table from all taxonomy evaluation results.

Reads taxonomy_eval.json from each subdirectory in eval/out/taxonomy/
and produces a cross-experiment comparison table.

Usage:
    cd src
    python -m eval.gen_taxonomy_summary
    python -m eval.gen_taxonomy_summary --taxonomy-dir eval/out/taxonomy --output eval/out/taxonomy/summary.md
"""

import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

CONTENT_CATEGORIES = {
    "A1": "Fact Fabrication",
    "A2": "Fact Distortion",
    "A3": "False Attribution",
    "A4": "Fabricated Data",
    "A5": "Context Manipulation",
    "A6": "Direct Instruction Injection",
}

MECHANISM_CATEGORIES = {
    "B1": "Evidence Monopolization",
    "B2": "Evidence Contamination",
    "B3": "Reasoning Chain Reconstruction",
    "B4": "Key Evidence Displacement",
    "B5": "Authority Deception",
}

VULNERABILITY_CATEGORIES = {
    "C1": "Retrieval Trust Blindness",
    "C2": "Quantity Over Quality",
    "C3": "Missing Source Verification",
    "C4": "Insufficient Consistency Check",
    "C5": "Inherent Claim Ambiguity",
}

ALL_CATS = {
    "A": ("Content", CONTENT_CATEGORIES),
    "B": ("Mechanism", MECHANISM_CATEGORIES),
    "C": ("Vulnerability", VULNERABILITY_CATEGORIES),
}


def load_all_taxonomy_results(taxonomy_dir):
    taxonomy_dir = Path(taxonomy_dir)
    results = []
    for json_path in sorted(taxonomy_dir.glob("*/taxonomy_eval.json")):
        subdir = json_path.parent.name
        if subdir == "combined" or subdir.startswith("combined_"):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_subdir"] = subdir
        results.append(data)
    return results


def parse_subdir_name(name):
    """Parse subdir like dev_fact2fiction_infact_0.04 or dev_fplus_infact_0.08_fail."""
    is_fail = name.endswith("_fail")
    is_all = name.endswith("_all")
    clean = name
    sample_type = "success"
    if is_fail:
        clean = name[:-5]
        sample_type = "fail"
    elif is_all:
        clean = name[:-4]
        sample_type = "all"

    parts = clean.split("_")
    if len(parts) >= 4:
        variant = parts[0]
        poison_rate = parts[-1]
        victim = parts[-2]
        attack_type = "_".join(parts[1:-2])
        return variant, attack_type, victim, poison_rate, sample_type
    return "", clean, "", "", sample_type


def _top_n(dist, n=3):
    """Return top-n codes from a primary distribution dict, sorted by pct."""
    if not dist:
        return []
    items = sorted(dist.items(), key=lambda x: x[1].get("pct", 0), reverse=True)
    return items[:n]


def _fmt_top(items, cat_map):
    """Format top-n items as 'A1(70%)'."""
    parts = []
    for code, info in items:
        pct = info.get("pct", 0)
        parts.append("%s(%.0f%%)" % (code, pct))
    return " ".join(parts)


def generate_overview_table(results):
    lines = []
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "| Experiment | Type | N "
        "| Top Content | Top Mechanism | Top Vulnerability |"
    )
    lines.append(
        "|------------|------|---"
        "|-------------|---------------|-------------------|"
    )

    for r in results:
        _, attack_type, _, pr, st = parse_subdir_name(r["_subdir"])
        agg = r["aggregate"]
        n = agg["total_classified"]
        label = "%s / %s" % (attack_type, pr)
        if st != "success":
            label += " [%s]" % st

        top_a = _fmt_top(
            _top_n(agg["dimension_A_content"]["primary"], 2),
            CONTENT_CATEGORIES,
        )
        top_b = _fmt_top(
            _top_n(agg["dimension_B_mechanism"]["primary"], 2),
            MECHANISM_CATEGORIES,
        )
        top_c = _fmt_top(
            _top_n(agg["dimension_C_vulnerability"]["primary"], 2),
            VULNERABILITY_CATEGORIES,
        )
        lines.append(
            "| %s | %s | %d | %s | %s | %s |"
            % (label, st, n, top_a, top_b, top_c)
        )

    lines.append("")
    return lines


def generate_dimension_table(results, dim_key, dim_agg_key, cat_map, title):
    """Generate a per-dimension comparison table showing percentages across experiments."""
    lines = []
    lines.append("## %s" % title)
    lines.append("")

    codes = list(cat_map.keys())
    header = "| Experiment | N |"
    sep = "|------------|---|"
    for code in codes:
        header += " %s |" % code
        sep += "----|"

    lines.append(header)
    lines.append(sep)

    for r in results:
        _, attack_type, _, pr, st = parse_subdir_name(r["_subdir"])
        agg = r["aggregate"]
        n = agg["total_classified"]
        label = "%s / %s" % (attack_type, pr)
        if st != "success":
            label += " [%s]" % st

        primary = agg.get(dim_agg_key, {}).get("primary", {})
        row = "| %s | %d |" % (label, n)
        for code in codes:
            pct = primary.get(code, {}).get("pct", 0)
            if pct > 0:
                row += " %.1f%% |" % pct
            else:
                row += " - |"
        lines.append(row)

    lines.append("")
    return lines


def generate_cross_attack_comparison(results):
    """Group results by attack_type and compare across poison rates."""
    groups = defaultdict(list)
    for r in results:
        _, attack_type, _, pr, st = parse_subdir_name(r["_subdir"])
        groups[(attack_type, st)].append((pr, r))

    lines = []
    lines.append("## Cross Poison-Rate Comparison")
    lines.append("")

    for (attack_type, st), items in sorted(groups.items()):
        items.sort(key=lambda x: float(x[0]) if x[0].replace(".", "").isdigit() else 0)
        suffix = "" if st == "success" else " [%s]" % st
        lines.append("### %s%s" % (attack_type, suffix))
        lines.append("")
        lines.append("| Poison Rate | N | Top A | Top B | Top C |")
        lines.append("|-------------|---|-------|-------|-------|")

        for pr, r in items:
            agg = r["aggregate"]
            n = agg["total_classified"]

            top_a_items = _top_n(agg["dimension_A_content"]["primary"], 1)
            top_b_items = _top_n(agg["dimension_B_mechanism"]["primary"], 1)
            top_c_items = _top_n(agg["dimension_C_vulnerability"]["primary"], 1)

            def _fmt1(top_items):
                if not top_items:
                    return "-"
                code, info = top_items[0]
                return "%s(%.0f%%)" % (code, info.get("pct", 0))

            lines.append(
                "| %s | %d | %s | %s | %s |"
                % (pr, n, _fmt1(top_a_items), _fmt1(top_b_items), _fmt1(top_c_items))
            )

        lines.append("")

    return lines


def generate_secondary_analysis(results):
    """Analyze secondary classifications to show co-occurrence patterns."""
    lines = []
    lines.append("## Secondary Classification Patterns")
    lines.append("")
    lines.append("Top primary+secondary co-occurrence pairs across all experiments:")
    lines.append("")

    from collections import Counter
    pair_counts = {"content": Counter(), "mechanism": Counter(), "vulnerability": Counter()}

    for r in results:
        for claim in r.get("per_claim", []):
            for dim in ["content", "mechanism", "vulnerability"]:
                p = claim.get("%s_primary" % dim, "")
                s = claim.get("%s_secondary" % dim)
                if p and s:
                    pair_counts[dim]["%s+%s" % (p, s)] += 1

    for dim, counter in pair_counts.items():
        if not counter:
            continue
        lines.append("**%s:**" % dim.capitalize())
        for pair, cnt in counter.most_common(5):
            lines.append("- %s (%d)" % (pair, cnt))
        lines.append("")

    return lines


def generate_markdown(results):
    lines = []
    lines.append("# Attack Taxonomy Evaluation Summary")
    lines.append("")
    lines.append("Generated from %d experiment results." % len(results))
    lines.append("")

    lines.extend(generate_overview_table(results))

    lines.extend(generate_dimension_table(
        results, "A", "dimension_A_content",
        CONTENT_CATEGORIES, "Dimension A: Malicious Content Characteristics",
    ))
    lines.extend(generate_dimension_table(
        results, "B", "dimension_B_mechanism",
        MECHANISM_CATEGORIES, "Dimension B: Attack Success Mechanism",
    ))
    lines.extend(generate_dimension_table(
        results, "C", "dimension_C_vulnerability",
        VULNERABILITY_CATEGORIES, "Dimension C: System Vulnerability Exploited",
    ))

    lines.extend(generate_cross_attack_comparison(results))
    lines.extend(generate_secondary_analysis(results))

    lines.append("---")
    lines.append("")
    lines.append("### Category Legend")
    lines.append("")
    for prefix, (dim_name, cat_map) in ALL_CATS.items():
        lines.append("**%s (%s):**" % (dim_name, prefix))
        for code, name in cat_map.items():
            lines.append("- %s: %s" % (code, name))
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown summary from taxonomy eval results",
    )
    parser.add_argument(
        "--taxonomy-dir", default=None,
        help="Directory with taxonomy experiment subdirs (default: eval/out/taxonomy/)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output Markdown file (default: <taxonomy-dir>/summary.md)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.taxonomy_dir is None:
        taxonomy_dir = Path(__file__).parent / "out" / "taxonomy"
    else:
        taxonomy_dir = Path(args.taxonomy_dir)

    results = load_all_taxonomy_results(taxonomy_dir)
    if not results:
        logger.error("No taxonomy_eval.json found in %s", taxonomy_dir)
        return

    logger.info("Loaded %d taxonomy experiment results", len(results))
    md = generate_markdown(results)

    if args.output is None:
        output_path = taxonomy_dir / "summary.md"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print("\nSaved to: %s" % output_path)


if __name__ == "__main__":
    main()
