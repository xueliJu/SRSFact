"""
Attack Stealth Evaluation Module

Evaluates the stealth of adversarial attacks on fact-checking systems
by computing semantic similarity between claims and justifications using
Sentence-BERT (all-mpnet-base-v2).

Metrics:
- Claim <-> Before-Justification similarity (original fact-check)
- Claim <-> After-Justification similarity (post-attack fact-check)
- Before <-> After Justification similarity (justification drift)
- Delta similarity (change in claim-justification alignment)

Data sources:
- Original: predictions.csv (from InFact output)
- Attack: attack_results_*.jsonl (from attack pipeline)
"""

import json
import json_repair
import csv
import re
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class StealthRecord:
    claim_id: int
    claim_text: str
    gt_label: str
    pred_label: str
    attack_success: bool
    fact_check_fail: bool
    sim_claim_before: float
    sim_claim_after: float
    sim_before_after: float
    delta_sim: float


@dataclass
class StealthReport:
    total_claims: int
    attack_success_count: int
    fact_check_fail_count: int
    records: list = field(default_factory=list)

    avg_sim_claim_before: float = 0.0
    avg_sim_claim_after: float = 0.0
    avg_sim_before_after: float = 0.0
    avg_delta_sim: float = 0.0
    std_sim_claim_before: float = 0.0
    std_sim_claim_after: float = 0.0
    std_sim_before_after: float = 0.0
    std_delta_sim: float = 0.0

    avg_sim_before_after_success: float = 0.0
    avg_sim_before_after_fail: float = 0.0
    avg_sim_before_after_neutral: float = 0.0


def extract_claim_text(raw_claim):
    """Extract core claim text from the raw claim string.

    Attack JSONL claims: Text: "actual claim"\nClaim date: ...
    Predictions CSV claims: plain text.
    """
    match = re.search(r'Text:\s*"(.+?)"', raw_claim, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"Text:\s*(.+?)(?:\nClaim|\Z)", raw_claim, re.DOTALL)
    if match:
        return match.group(1).strip().strip('"')
    return raw_claim.strip()


def strip_url_references(text):
    """Remove inline URL references from justification text."""
    text = re.sub(r'\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    return text.strip()


def load_attack_results(jsonl_path):
    records = []
    with open(jsonl_path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    records.append(json_repair.loads(line))
                    logger.warning("Repaired malformed JSON on line %d", i + 1)
                except Exception as e:
                    logger.error("Skipping unreadable line %d: %s", i + 1, e)
    records.sort(key=lambda x: x['claim_id'])
    logger.info("Loaded %d attack results from %s", len(records), jsonl_path)
    return records


def load_predictions(csv_path):
    preds = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['sample_index'])
            preds[idx] = row
    logger.info("Loaded %d predictions from %s", len(preds), csv_path)
    return preds


def find_attack_results_file(attack_dir):
    results_dir = Path(attack_dir) / "results"
    if not results_dir.exists():
        raise FileNotFoundError("Results dir not found: %s" % results_dir)
    jsonl_files = list(results_dir.glob("attack_results_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError("No attack_results_*.jsonl in %s" % results_dir)
    if len(jsonl_files) > 1:
        logger.warning("Multiple JSONL files found, using: %s", jsonl_files[0].name)
    return jsonl_files[0]


def find_predictions_file(original_dir):
    original_dir = Path(original_dir)
    for c in [original_dir / "predictions.csv",
              original_dir / "search_top_five" / "predictions.csv"]:
        if c.exists():
            return c
    raise FileNotFoundError("predictions.csv not found in %s" % original_dir)


class StealthEvaluator:

    def __init__(self, model_name="all-mpnet-base-v2", device=None, batch_size=64):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Sentence-BERT model: %s on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.device = device

    def encode(self, texts):
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def cosine_similarity(self, emb_a, emb_b):
        return np.sum(emb_a * emb_b, axis=1)

    def evaluate(self, attack_results, predictions=None, strip_urls=True):
        claims, before_justs, after_justs, valid_records = [], [], [], []

        for rec in attack_results:
            claim_text = extract_claim_text(rec["claim"])
            before_just = rec.get("before_justification", "")
            after_just = rec.get("after_justification", "")

            if not before_just or not after_just:
                logger.warning("Skipping claim %d: missing justification", rec['claim_id'])
                continue

            if predictions and rec["claim_id"] in predictions:
                pred = predictions[rec["claim_id"]]
                csv_just = pred.get("justification", "")
                if csv_just:
                    before_just = csv_just

            if strip_urls:
                claim_text = strip_url_references(claim_text)
                before_just = strip_url_references(before_just)
                after_just = strip_url_references(after_just)

            claims.append(claim_text)
            before_justs.append(before_just)
            after_justs.append(after_just)
            valid_records.append(rec)

        if not valid_records:
            raise ValueError("No valid records found for evaluation")

        logger.info("Encoding %d claims and justifications...", len(valid_records))
        emb_claims = self.encode(claims)
        emb_before = self.encode(before_justs)
        emb_after = self.encode(after_justs)

        sim_cb = self.cosine_similarity(emb_claims, emb_before)
        sim_ca = self.cosine_similarity(emb_claims, emb_after)
        sim_ba = self.cosine_similarity(emb_before, emb_after)
        delta = sim_ca - sim_cb

        records = []
        for i, rec in enumerate(valid_records):
            records.append(StealthRecord(
                claim_id=rec["claim_id"],
                claim_text=extract_claim_text(rec["claim"]),
                gt_label=rec["gt_label"],
                pred_label=rec["pred_label"],
                attack_success=rec["attack_success"],
                fact_check_fail=rec["fact_check_fail"],
                sim_claim_before=float(sim_cb[i]),
                sim_claim_after=float(sim_ca[i]),
                sim_before_after=float(sim_ba[i]),
                delta_sim=float(delta[i]),
            ))

        success_ba = [r.sim_before_after for r in records if r.attack_success]
        fail_ba = [r.sim_before_after
                    for r in records if r.fact_check_fail and not r.attack_success]
        neutral_ba = [r.sim_before_after
                      for r in records if not r.attack_success and not r.fact_check_fail]

        return StealthReport(
            total_claims=len(records),
            attack_success_count=sum(1 for r in records if r.attack_success),
            fact_check_fail_count=sum(1 for r in records if r.fact_check_fail),
            records=records,
            avg_sim_claim_before=float(np.mean(sim_cb)),
            avg_sim_claim_after=float(np.mean(sim_ca)),
            avg_sim_before_after=float(np.mean(sim_ba)),
            avg_delta_sim=float(np.mean(delta)),
            std_sim_claim_before=float(np.std(sim_cb)),
            std_sim_claim_after=float(np.std(sim_ca)),
            std_sim_before_after=float(np.std(sim_ba)),
            std_delta_sim=float(np.std(delta)),
            avg_sim_before_after_success=(
                float(np.mean(success_ba)) if success_ba else 0.0),
            avg_sim_before_after_fail=(
                float(np.mean(fail_ba)) if fail_ba else 0.0),
            avg_sim_before_after_neutral=(
                float(np.mean(neutral_ba)) if neutral_ba else 0.0),
        )


def report_to_dict(report, include_records=True):
    d = {
        "total_claims": report.total_claims,
        "attack_success_count": report.attack_success_count,
        "fact_check_fail_count": report.fact_check_fail_count,
        "overall": {
            "avg_sim_claim_before_justification": round(report.avg_sim_claim_before, 4),
            "avg_sim_claim_after_justification": round(report.avg_sim_claim_after, 4),
            "avg_sim_before_after_justification": round(report.avg_sim_before_after, 4),
            "avg_delta_similarity": round(report.avg_delta_sim, 4),
            "std_sim_claim_before_justification": round(report.std_sim_claim_before, 4),
            "std_sim_claim_after_justification": round(report.std_sim_claim_after, 4),
            "std_sim_before_after_justification": round(report.std_sim_before_after, 4),
            "std_delta_similarity": round(report.std_delta_sim, 4),
        },
        "by_attack_outcome": {
            "attack_success": {
                "count": report.attack_success_count,
                "avg_sim_before_after": round(
                    report.avg_sim_before_after_success, 4),
            },
            "fact_check_fail_only": {
                "count": (report.fact_check_fail_count
                          - report.attack_success_count),
                "avg_sim_before_after": round(
                    report.avg_sim_before_after_fail, 4),
            },
            "no_effect": {
                "count": (report.total_claims
                          - report.fact_check_fail_count),
                "avg_sim_before_after": round(
                    report.avg_sim_before_after_neutral, 4),
            },
        },
    }

    if include_records:
        d["per_claim"] = [
            {
                "claim_id": r.claim_id,
                "claim_text": r.claim_text[:120],
                "gt_label": r.gt_label,
                "pred_label": r.pred_label,
                "attack_success": r.attack_success,
                "sim_claim_before": round(r.sim_claim_before, 4),
                "sim_claim_after": round(r.sim_claim_after, 4),
                "sim_before_after": round(r.sim_before_after, 4),
                "delta_sim": round(r.delta_sim, 4),
            }
            for r in report.records
        ]

    return d


def format_report(report):
    sep = "=" * 72
    lines = [
        sep,
        "  Attack Stealth Evaluation Report",
        "  Model: Sentence-BERT (all-mpnet-base-v2)",
        sep,
        "",
        "  Total claims evaluated: %d" % report.total_claims,
        "  Attack success (label flipped): %d" % report.attack_success_count,
        "  Fact-check fail (wrong/NEI): %d" % report.fact_check_fail_count,
        "",
        "  [Overall Semantic Similarity]",
        "    Claim <-> Before-Justification:  %.4f +/- %.4f" % (
            report.avg_sim_claim_before, report.std_sim_claim_before),
        "    Claim <-> After-Justification:   %.4f +/- %.4f" % (
            report.avg_sim_claim_after, report.std_sim_claim_after),
        "    Before <-> After Justification:  %.4f +/- %.4f" % (
            report.avg_sim_before_after, report.std_sim_before_after),
        "    Delta (After - Before):          %+.4f +/- %.4f" % (
            report.avg_delta_sim, report.std_delta_sim),
        "",
        "  [By Attack Outcome -- Before <-> After Justification Similarity]",
        "    Attack Success  (n=%3d): %.4f" % (
            report.attack_success_count,
            report.avg_sim_before_after_success),
        "    FC Fail Only    (n=%3d): %.4f" % (
            report.fact_check_fail_count - report.attack_success_count,
            report.avg_sim_before_after_fail),
        "    No Effect       (n=%3d): %.4f" % (
            report.total_claims - report.fact_check_fail_count,
            report.avg_sim_before_after_neutral),
        "",
        "  Interpretation:",
        "    * Higher Before<->After sim = more stealthy (less justification drift)",
        "    * Delta close to 0 = claim-justification alignment preserved",
        "    * Compare across outcomes to see if successful attacks are detectable",
        sep,
    ]
    return "\n".join(lines)


def build_output_subdir(config):
    """Build output subdirectory name from attack experiment config parameters.

    Format: {variant}_{attack_type}_{victim}_{poison_rate}
    """
    variant = config.get("variant", "unknown")
    attack_type = config.get("attack_type", "unknown")
    victim = config.get("victim", "unknown")
    poison_rate = config.get("poison_rate", 0)
    return "%s_%s_%s_%s" % (variant, attack_type, victim, poison_rate)


def get_default_output_dir():
    """Return eval/out/ directory relative to this script's location."""
    return Path(__file__).parent / "out"


def run_evaluation(
    attack_dir,
    original_dir=None,
    output_dir=None,
    model_name="all-mpnet-base-v2",
    device=None,
    batch_size=64,
    strip_urls=True,
    save_per_claim=True,
):
    """End-to-end stealth evaluation pipeline."""
    attack_dir = Path(attack_dir)
    if not attack_dir.exists():
        raise FileNotFoundError("Attack directory not found: %s" % attack_dir)

    config_path = attack_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

    if original_dir is None and config:
        attack_set_dir = Path(config.get("attack_set_dir", ""))
        victim = config.get("victim", "")
        original_model = config.get("original_fact_checker_model", "")
        inferred = attack_set_dir / victim / original_model / "search_top_five"
        if inferred.exists():
            original_dir = str(inferred)
            logger.info("Auto-inferred original dir: %s", original_dir)

    attack_jsonl = find_attack_results_file(attack_dir)
    attack_results = load_attack_results(attack_jsonl)

    predictions = None
    if original_dir:
        try:
            pred_csv = find_predictions_file(Path(original_dir))
            predictions = load_predictions(pred_csv)
        except FileNotFoundError as e:
            logger.warning("Could not load predictions: %s", e)

    evaluator = StealthEvaluator(
        model_name=model_name, device=device, batch_size=batch_size
    )
    report = evaluator.evaluate(attack_results, predictions, strip_urls=strip_urls)

    if output_dir is None:
        output_dir = str(get_default_output_dir())
    output_dir = Path(output_dir)

    if config:
        sub_dir = build_output_subdir(config)
    else:
        sub_dir = attack_dir.name
    output_dir = output_dir / sub_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report_dict = report_to_dict(report, include_records=save_per_claim)
    report_dict["config"] = {
        "model_name": model_name,
        "strip_urls": strip_urls,
        "attack_dir": str(attack_dir),
        "original_dir": original_dir,
        "attack_results_file": str(attack_jsonl),
    }
    if config:
        report_dict["experiment_config"] = config

    json_path = output_dir / "stealth_eval.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON report: %s", json_path)

    txt_path = output_dir / "stealth_eval.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(format_report(report))
    logger.info("Saved text report: %s", txt_path)

    if save_per_claim:
        csv_path = output_dir / "stealth_eval.csv"
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "claim_id", "claim_text", "gt_label", "pred_label",
                "attack_success", "fact_check_fail",
                "sim_claim_before", "sim_claim_after",
                "sim_before_after", "delta_sim",
            ])
            for r in report.records:
                writer.writerow([
                    r.claim_id, r.claim_text[:200], r.gt_label, r.pred_label,
                    r.attack_success, r.fact_check_fail,
                    "%.4f" % r.sim_claim_before,
                    "%.4f" % r.sim_claim_after,
                    "%.4f" % r.sim_before_after,
                    "%.4f" % r.delta_sim,
                ])
        logger.info("Saved per-claim CSV: %s", csv_path)

    print(format_report(report))
    print("\nResult saved to: %s" % output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate attack stealth via Sentence-BERT similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--attack-dir", required=True,
        help="Path to attack experiment directory",
    )
    parser.add_argument(
        "--original-dir", default=None,
        help="Path to original factcheck output dir (auto-inferred if omitted)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: eval/out/)",
    )
    parser.add_argument(
        "--model", default="all-mpnet-base-v2",
        help="Sentence-BERT model name (default: all-mpnet-base-v2)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device: cuda, cpu, cuda:0 (default: auto)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Encoding batch size (default: 64)",
    )
    parser.add_argument(
        "--no-strip-urls", action="store_true",
        help="Disable stripping URL references from text",
    )
    parser.add_argument(
        "--no-per-claim", action="store_true",
        help="Disable per-claim detail output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_evaluation(
        attack_dir=args.attack_dir,
        original_dir=args.original_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        strip_urls=not args.no_strip_urls,
        save_per_claim=not args.no_per_claim,
    )


if __name__ == "__main__":
    main()
