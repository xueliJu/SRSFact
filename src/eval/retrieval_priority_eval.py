"""
Retrieval priority evaluation for fake evidence.

Goal:
- Quantify how often fake evidence is retrieved early (Top-k curve).
- Provide an AUC-like score over that curve.
- Measure whether fake evidence appears before original evidence.

Usage:
    cd src
    python -m eval.retrieval_priority_eval --attack-dir attack/attack_results/dev_naive_infact_0.08
    python -m eval.retrieval_priority_eval --attack-dir attack/attack_results/dev_naive_infact_0.08 --max-k 15
"""

import argparse
import json
import logging
from pathlib import Path

import json_repair

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

logger = logging.getLogger(__name__)

FILTER_SUCCESS = "success"
FILTER_FAIL_ONLY = "fail_only"
FILTER_ALL = "all"


def load_attack_results(jsonl_path: Path) -> list[dict]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    records.append(json_repair.loads(line))
                except Exception:
                    pass
    return records


def find_attack_results_file(attack_dir: Path) -> Path:
    results_dir = attack_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError("Results dir not found: %s" % results_dir)
    jsonl_files = list(results_dir.glob("attack_results_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError("No attack_results_*.jsonl in %s" % results_dir)
    return jsonl_files[0]


def _filter_records(records: list[dict], filter_mode: str) -> list[dict]:
    if filter_mode == FILTER_SUCCESS:
        return [r for r in records if r.get("attack_success")]
    if filter_mode == FILTER_FAIL_ONLY:
        return [r for r in records if r.get("fact_check_fail") and not r.get("attack_success")]
    return records


def _extract_rank_from_urls(urls: list[str], target_fake: bool) -> int | None:
    if not isinstance(urls, list):
        return None
    for idx, u in enumerate(urls):
        if not isinstance(u, str):
            continue
        is_fake = "created" in u
        if is_fake == target_fake:
            return idx + 1
    return None


def _infer_ranks(record: dict) -> tuple[int | None, int | None]:
    first_fake_rank = record.get("first_fake_rank")
    first_original_rank = record.get("first_original_rank")
    if isinstance(first_fake_rank, int) or isinstance(first_original_rank, int):
        return first_fake_rank, first_original_rank

    urls = record.get("used_evidence_urls_ordered")
    if isinstance(urls, list):
        return _extract_rank_from_urls(urls, True), _extract_rank_from_urls(urls, False)
    return None, None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_priority_metrics(records: list[dict], max_k: int = 20) -> dict:
    n = len(records)
    if n == 0:
        return {}

    first_fake_ranks = []
    first_original_ranks = []
    missing_rank_count = 0

    for r in records:
        fr, orr = _infer_ranks(r)
        first_fake_ranks.append(fr)
        first_original_ranks.append(orr)
        if fr is None and orr is None:
            missing_rank_count += 1

    available_fake_ranks = [x for x in first_fake_ranks if isinstance(x, int) and x > 0]
    if not available_fake_ranks:
        return {
            "total_records": n,
            "missing_rank_count": missing_rank_count,
            "error": (
                "No rank information found. Please regenerate attack results "
                "with `used_evidence_urls_ordered` / `first_fake_rank` fields."
            ),
        }

    max_observed_k = max(available_fake_ranks)
    k_cap = max(1, min(max_k, max_observed_k))
    ks = list(range(1, k_cap + 1))

    # Top-k fake hit curve: P(first_fake_rank <= k)
    topk_probs = []
    for k in ks:
        hit = sum(1 for fr in first_fake_ranks if isinstance(fr, int) and fr <= k)
        topk_probs.append(hit / n)

    # AUC-like score over Top-k hit curve (normalized to [0,1]).
    # Discrete area via trapezoid from x=0..k_cap where y(0)=0.
    xs = [0] + ks
    ys = [0.0] + topk_probs
    auc_raw = 0.0
    for i in range(1, len(xs)):
        auc_raw += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    auc_normalized = auc_raw / k_cap if k_cap > 0 else 0.0

    comparable_pairs = []
    for fr, orr in zip(first_fake_ranks, first_original_ranks):
        if isinstance(fr, int) and isinstance(orr, int):
            comparable_pairs.append((fr, orr))

    fake_before_original_rate = _mean(
        [1.0 if fr < orr else 0.0 for fr, orr in comparable_pairs]
    )

    mrr_fake = _mean([1.0 / fr if isinstance(fr, int) and fr > 0 else 0.0 for fr in first_fake_ranks])

    return {
        "total_records": n,
        "missing_rank_count": missing_rank_count,
        "k_cap": k_cap,
        "curve": [{"k": k, "p_fake_in_topk": p} for k, p in zip(ks, topk_probs)],
        "auc_topk": auc_raw,
        "auc_topk_normalized": auc_normalized,
        "p_fake_top1": topk_probs[0] if topk_probs else 0.0,
        "p_fake_top3": topk_probs[min(2, len(topk_probs) - 1)] if topk_probs else 0.0,
        "p_fake_top5": topk_probs[min(4, len(topk_probs) - 1)] if topk_probs else 0.0,
        "p_fake_any": _mean([1.0 if isinstance(fr, int) else 0.0 for fr in first_fake_ranks]),
        "mrr_fake": mrr_fake,
        "comparable_pair_count": len(comparable_pairs),
        "fake_before_original_rate": fake_before_original_rate,
    }


def build_output_subdir(config: dict) -> str:
    variant = config.get("variant", "unknown")
    attack_type = config.get("attack_type", "unknown")
    victim = config.get("victim", "unknown")
    poison_rate = config.get("poison_rate", 0)
    return "%s_%s_%s_%s" % (variant, attack_type, victim, poison_rate)


def plot_curve(curve: list[dict], title: str, out_png: Path) -> None:
    if plt is None:
        logger.warning("matplotlib is not installed, skip plotting curve: %s", out_png)
        return

    ks = [x["k"] for x in curve]
    ys = [x["p_fake_in_topk"] for x in curve]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, ys, marker="o", linewidth=2)
    plt.ylim(0, 1.02)
    plt.xlim(min(ks), max(ks))
    plt.xlabel("k (Top-k)")
    plt.ylabel("P(fake evidence in Top-k)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def format_report(metrics: dict, name: str) -> str:
    if "error" in metrics:
        return (
            "=== Retrieval Priority Eval: %s ===\n"
            "Total records: %d\n"
            "Missing rank count: %d\n"
            "Error: %s\n"
            % (
                name,
                metrics.get("total_records", 0),
                metrics.get("missing_rank_count", 0),
                metrics["error"],
            )
        )
    return (
        "=== Retrieval Priority Eval: %s ===\n"
        "Total records: %d\n"
        "Missing rank count: %d\n"
        "k_cap: %d\n"
        "AUC(Top-k curve): %.4f\n"
        "AUC normalized: %.4f\n"
        "P(fake in Top-1): %.4f\n"
        "P(fake in Top-3): %.4f\n"
        "P(fake in Top-5): %.4f\n"
        "P(fake ever used): %.4f\n"
        "MRR(fake): %.4f\n"
        "Comparable pairs: %d\n"
        "P(fake before original): %.4f\n"
        % (
            name,
            metrics["total_records"],
            metrics["missing_rank_count"],
            metrics["k_cap"],
            metrics["auc_topk"],
            metrics["auc_topk_normalized"],
            metrics["p_fake_top1"],
            metrics["p_fake_top3"],
            metrics["p_fake_top5"],
            metrics["p_fake_any"],
            metrics["mrr_fake"],
            metrics["comparable_pair_count"],
            metrics["fake_before_original_rate"],
        )
    )


def run_eval(
    attack_dirs: list[str],
    output_dir: str | None = None,
    max_k: int = 20,
    filter_mode: str = FILTER_ALL,
) -> None:
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "out" / "retrieval_priority")
    base_out = Path(output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    all_records = []
    dir_bundle = []
    suffix = ""
    if filter_mode == FILTER_SUCCESS:
        suffix = "_success"
    elif filter_mode == FILTER_FAIL_ONLY:
        suffix = "_fail"
    else:
        suffix = "_all"

    for ad in attack_dirs:
        ad_path = Path(ad)
        jsonl_path = find_attack_results_file(ad_path)
        records = _filter_records(load_attack_results(jsonl_path), filter_mode)

        cfg = {}
        cfg_path = ad_path / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        all_records.extend(records)
        dir_bundle.append((ad_path, cfg, records))
        logger.info("Loaded %d records from %s", len(records), ad_path.name)

    for ad_path, cfg, records in dir_bundle:
        if cfg:
            name = build_output_subdir(cfg) + suffix
        else:
            name = ad_path.name + suffix
        out_dir = base_out / name
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics = compute_priority_metrics(records, max_k=max_k)
        with open(out_dir / "retrieval_priority_eval.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        report = format_report(metrics, name)
        with open(out_dir / "retrieval_priority_eval.txt", "w", encoding="utf-8") as f:
            f.write(report)

        if "curve" in metrics:
            plot_curve(metrics["curve"], title=name, out_png=out_dir / "fake_topk_curve.png")

        print(report)
        print("Saved to: %s\n" % out_dir)

    if len(dir_bundle) > 1:
        combined_name = "combined" + suffix
        combined_dir = base_out / combined_name
        combined_dir.mkdir(parents=True, exist_ok=True)
        metrics = compute_priority_metrics(all_records, max_k=max_k)
        with open(combined_dir / "retrieval_priority_eval.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        report = format_report(metrics, combined_name)
        with open(combined_dir / "retrieval_priority_eval.txt", "w", encoding="utf-8") as f:
            f.write(report)
        if "curve" in metrics:
            plot_curve(metrics["curve"], title=combined_name, out_png=combined_dir / "fake_topk_curve.png")
        print(report)
        print("Combined saved to: %s" % combined_dir)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fake-evidence retrieval priority with Top-k curve/AUC")
    parser.add_argument("--attack-dir", nargs="+", required=True, help="One or more attack experiment directories")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: eval/out/retrieval_priority)")
    parser.add_argument("--max-k", type=int, default=20, help="Maximum k for Top-k curve")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--success-only", action="store_true", help="Use only attack_success samples")
    grp.add_argument("--fail-only", action="store_true", help="Use only fact_check_fail but not attack_success samples")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.success_only:
        mode = FILTER_SUCCESS
    elif args.fail_only:
        mode = FILTER_FAIL_ONLY
    else:
        mode = FILTER_ALL

    run_eval(
        attack_dirs=args.attack_dir,
        output_dir=args.output_dir,
        max_k=args.max_k,
        filter_mode=mode,
    )


if __name__ == "__main__":
    main()
