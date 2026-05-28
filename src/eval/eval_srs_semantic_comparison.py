#!/usr/bin/env python3
"""
SRSFact 语义对比分析脚本
对比两个攻击结果文件夹的语义指标（无需 evidence 文本）
"""

import json
import json_repair
import csv
import re
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_claim_text(raw_claim):
    """从原始 claim 字符串中提取核心声明文本"""
    match = re.search(r'Text:\s*"(.+?)"', raw_claim, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"Text:\s*(.+?)(?:\nClaim|\Z)", raw_claim, re.DOTALL)
    if match:
        return match.group(1).strip().strip('"')
    return raw_claim.strip()


def strip_url_references(text):
    """从文本中移除 URL 引用"""
    text = re.sub(r'\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    return text.strip()


def split_sentences(text):
    """尽量稳妥地切分句子。"""
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [p.strip() for p in parts if p and p.strip()]


CORE_SENTENCE_PATTERNS = re.compile(
    r"\b(refuted|refute|supported|support|not enough information|insufficient|"
    r"therefore|there is no|there is not|conclude|concludes|conclusion|"
    r"cannot verify|could not verify|does not|did not|never|is not|was not)\b",
    re.IGNORECASE,
)


def extract_core_justification(text, max_sentences=2):
    """提取 justification 中最接近结论的核心句。"""
    if not text:
        return ""

    text = strip_url_references(text)
    sentences = split_sentences(text)
    if not sentences:
        return text.strip()

    core_sentences = [s for s in sentences if CORE_SENTENCE_PATTERNS.search(s)]
    if core_sentences:
        return " ".join(core_sentences[:max_sentences]).strip()

    return " ".join(sentences[:max_sentences]).strip()


def count_extra_sentences(text):
    """统计非核心句数量，作为副指标。"""
    if not text:
        return 0
    sentences = split_sentences(strip_url_references(text))
    if not sentences:
        return 0
    core_count = sum(1 for s in sentences if CORE_SENTENCE_PATTERNS.search(s))
    return max(0, len(sentences) - core_count)


def load_attack_results(jsonl_path):
    """加载攻击结果 JSONL 文件"""
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
                except Exception as e:
                    logger.warning(f"Skipping unreadable line {i+1}: {e}")
    return records


def pair_records_by_claim_id(left_records: List[dict], right_records: List[dict]) -> tuple[list[dict], list[dict], int]:
    """Align two result sets by claim_id and keep only common claims."""
    left_map = {rec.get("claim_id"): rec for rec in left_records if rec.get("claim_id") is not None}
    right_map = {rec.get("claim_id"): rec for rec in right_records if rec.get("claim_id") is not None}
    common_ids = sorted(set(left_map).intersection(right_map))
    return [left_map[cid] for cid in common_ids], [right_map[cid] for cid in common_ids], len(common_ids)


@dataclass
class SemanticMetrics:
    """语义指标"""
    label: str
    n_total: int
    n_ranked: int
    rank_coverage: float
    asr: float  # 攻击成功率
    sfr: float  # 系统失败率
    avg_sim_claim_before: float
    avg_sim_claim_after: float
    avg_sim_before_after: float
    avg_delta_sim: float
    std_sim_before_after: float
    # 检索相关
    p_fake_top1: float
    p_fake_top3: float
    p_fake_before_original: float
    mrr_fake: float
    avg_used_fake: float
    avg_fake_usage_rate: float
    avg_extra_before: float
    avg_extra_after: float
    defense_score: float


def compute_semantic_metrics(records: List[dict], model_name: str = "all-mpnet-base-v2") -> SemanticMetrics:
    """计算语义指标"""
    
    if not records:
        raise ValueError("No records to evaluate")
    
    # 准备数据
    claims, before_justs, after_justs, valid_records = [], [], [], []
    extra_before_counts, extra_after_counts = [], []
    
    for rec in records:
        claim_text = extract_claim_text(rec.get("claim", ""))
        before_just = rec.get("before_justification", "")
        after_just = rec.get("after_justification", "")
        
        if not before_just or not after_just or not claim_text:
            continue
        
        claim_text = strip_url_references(claim_text)
        before_extra = count_extra_sentences(before_just)
        after_extra = count_extra_sentences(after_just)
        before_just = extract_core_justification(before_just)
        after_just = extract_core_justification(after_just)
        
        claims.append(claim_text)
        before_justs.append(before_just)
        after_justs.append(after_just)
        valid_records.append(rec)
        extra_before_counts.append(before_extra)
        extra_after_counts.append(after_extra)
    
    if not valid_records:
        raise ValueError("No valid records with justifications")
    
    # 加载模型并编码
    logger.info(f"Loading {model_name} for encoding...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    
    logger.info(f"Encoding {len(valid_records)} claims and justifications...")
    emb_claims = model.encode(claims, batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    emb_before = model.encode(before_justs, batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    emb_after = model.encode(after_justs, batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    
    # 计算余弦相似度
    sim_cb = np.sum(emb_claims * emb_before, axis=1)
    sim_ca = np.sum(emb_claims * emb_after, axis=1)
    sim_ba = np.sum(emb_before * emb_after, axis=1)
    delta = sim_ca - sim_cb
    
    # 计算攻击成功率
    asr = sum(1 for r in valid_records if r.get("attack_success", False)) / len(valid_records)
    sfr = sum(1 for r in valid_records if r.get("fact_check_fail", False)) / len(valid_records)
    
    # 检索相关指标
    fake_ranks = []
    fake_before_original_list = []
    used_fake_evidence_list = []
    fake_usage_rates = []
    
    for rec in valid_records:
        used_urls = rec.get("used_evidence_urls_ordered", [])
        if isinstance(used_urls, list):
            rank = None
            original_rank = None
            for idx, url in enumerate(used_urls):
                if "created" in url and rank is None:
                    rank = idx + 1
                if "created" not in url and original_rank is None:
                    original_rank = idx + 1
            
            if rank is not None:
                fake_ranks.append(rank)
                if original_rank is not None:
                    fake_before_original_list.append(1.0 if rank < original_rank else 0.0)
            
            used_fake = len([u for u in used_urls if "created" in u])
            used_fake_evidence_list.append(used_fake)
        
        fake_rate = rec.get("fake_evidence_usage_rate", 0.0)
        fake_usage_rates.append(fake_rate)
    
    # 计算检索指标
    n_ranked = len(fake_ranks)
    rank_coverage = n_ranked / len(valid_records) if valid_records else 0.0
    if fake_ranks:
        p_fake_top1 = sum(1 for r in fake_ranks if r <= 1) / n_ranked
        p_fake_top3 = sum(1 for r in fake_ranks if r <= 3) / n_ranked
        p_fake_before_original = np.mean(fake_before_original_list) if fake_before_original_list else 0.0
        mrr_fake = np.mean([1.0 / r for r in fake_ranks])
    else:
        p_fake_top1 = float("nan")
        p_fake_top3 = float("nan")
        p_fake_before_original = float("nan")
        mrr_fake = float("nan")

    # Defense-oriented score (higher is better):
    # emphasize end-task robustness (ASR/SFR) and include retrieval signals when available.
    components = [
        (1.0 - asr, 0.40),
        (1.0 - sfr, 0.35),
    ]
    optional = [
        (1.0 - p_fake_top1, 0.10),
        (1.0 - p_fake_top3, 0.07),
        (1.0 - p_fake_before_original, 0.05),
        (1.0 - mrr_fake, 0.03),
    ]
    for value, weight in optional:
        if not np.isnan(value):
            components.append((value, weight))
    weight_sum = sum(w for _, w in components)
    defense_score = float(sum(v * w for v, w in components) / weight_sum) if weight_sum > 0 else 0.0

    return SemanticMetrics(
        label="metrics",
        n_total=len(valid_records),
        n_ranked=n_ranked,
        rank_coverage=float(rank_coverage),
        asr=float(asr),
        sfr=float(sfr),
        avg_sim_claim_before=float(np.mean(sim_cb)),
        avg_sim_claim_after=float(np.mean(sim_ca)),
        avg_sim_before_after=float(np.mean(sim_ba)),
        avg_delta_sim=float(np.mean(delta)),
        std_sim_before_after=float(np.std(sim_ba)),
        p_fake_top1=float(p_fake_top1),
        p_fake_top3=float(p_fake_top3),
        p_fake_before_original=float(p_fake_before_original),
        mrr_fake=float(mrr_fake),
        avg_used_fake=float(np.mean(used_fake_evidence_list)) if used_fake_evidence_list else 0.0,
        avg_fake_usage_rate=float(np.mean(fake_usage_rates)) if fake_usage_rates else 0.0,
        avg_extra_before=float(np.mean(extra_before_counts)) if extra_before_counts else 0.0,
        avg_extra_after=float(np.mean(extra_after_counts)) if extra_after_counts else 0.0,
        defense_score=defense_score,
    )


def list_attack_results_files(attack_dir):
    """列出目录内所有 attack_results_*.jsonl 文件。"""
    results_dir = Path(attack_dir) / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    jsonl_files = sorted(results_dir.glob("attack_results_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No attack_results_*.jsonl in {results_dir}")
    return jsonl_files


def find_attack_results_file(attack_dir, selected_result=None):
    """查找 attack_results_*.jsonl 文件，可按文件名或子串选择。"""
    jsonl_files = list_attack_results_files(attack_dir)

    if selected_result:
        selected_result = selected_result.strip()

        # 1) 精确按文件名匹配
        for path in jsonl_files:
            if path.name == selected_result:
                return path

        # 2) 按子串匹配
        matches = [p for p in jsonl_files if selected_result in p.name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                "Multiple result files matched '%s' in %s: %s" % (
                    selected_result,
                    attack_dir,
                    ", ".join(p.name for p in matches),
                )
            )

        raise FileNotFoundError(
            "No result file matched '%s' in %s. Available: %s" % (
                selected_result,
                attack_dir,
                ", ".join(p.name for p in jsonl_files),
            )
        )

    if len(jsonl_files) > 1:
        logger.warning(
            "Multiple result files found in %s, defaulting to first: %s",
            attack_dir,
            jsonl_files[0].name,
        )
    return jsonl_files[0]


def format_table(metrics_list: List[SemanticMetrics]) -> str:
    """格式化为 Markdown 表格"""
    
    headers = ["Metric", metrics_list[0].label, metrics_list[1].label]
    
    metric_names = [
        ("Defense Score (higher better)", "defense_score"),
        ("Rank Coverage", "rank_coverage"),
        ("ASR (Attack Success Rate)", "asr"),
        ("SFR (System Fail Rate)", "sfr"),
        ("Avg Sim(Claim, Before)", "avg_sim_claim_before"),
        ("Avg Sim(Claim, After)", "avg_sim_claim_after"),
        ("Avg Sim(Before, After)", "avg_sim_before_after"),
        ("Avg Delta Sim", "avg_delta_sim"),
        ("P(Fake in Top-1)", "p_fake_top1"),
        ("P(Fake in Top-3)", "p_fake_top3"),
        ("P(Fake before Original)", "p_fake_before_original"),
        ("MRR(Fake)", "mrr_fake"),
        ("Avg Extra Sentences (Before)", "avg_extra_before"),
        ("Avg Extra Sentences (After)", "avg_extra_after"),
    ]
    
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    
    for metric_label, attr_name in metric_names:
        values = [metric_label]
        for metrics in metrics_list:
            val = getattr(metrics, attr_name, 0.0)
            if isinstance(val, float) and np.isnan(val):
                values.append("N/A")
            elif isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SRSFact 语义对比分析")
    parser.add_argument("--dir1", type=str, required=True, help="第一个攻击结果文件夹（Fact2Fiction）")
    parser.add_argument("--dir2", type=str, required=True, help="第二个攻击结果文件夹（FPlus）")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 文件路径")
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2", help="句子编码模型")
    parser.add_argument("--left-label", type=str, default="SRSFact", help="左侧列标签")
    parser.add_argument("--right-label", type=str, default="InFact", help="右侧列标签")
    parser.add_argument(
        "--left-result",
        type=str,
        default=None,
        help="左侧目录使用的结果文件名或子串（例如: att_gpt-5.4-mini）",
    )
    parser.add_argument(
        "--right-result",
        type=str,
        default=None,
        help="右侧目录使用的结果文件名或子串（例如: att_gpt-5.4-mini）",
    )
    
    args = parser.parse_args()
    
    # 加载两个文件夹的攻击结果
    logger.info(f"Loading attack results from {args.dir1}...")
    jsonl1 = find_attack_results_file(args.dir1, args.left_result)
    records1 = load_attack_results(jsonl1)
    
    logger.info(f"Loading attack results from {args.dir2}...")
    jsonl2 = find_attack_results_file(args.dir2, args.right_result)
    records2 = load_attack_results(jsonl2)

    records1, records2, common_count = pair_records_by_claim_id(records1, records2)
    if common_count == 0:
        raise ValueError("No common claim_id found between the two result folders.")
    
    logger.info(f"Using left result file: {jsonl1.name}")
    logger.info(f"Using right result file: {jsonl2.name}")
    logger.info(f"Loaded {len(records1)} paired records from {jsonl1.name}")
    logger.info(f"Loaded {len(records2)} paired records from {jsonl2.name}")
    
    # 计算语义指标
    logger.info(f"Computing semantic metrics for {args.left_label}...")
    metrics1 = compute_semantic_metrics(records1, args.model)
    metrics1.label = args.left_label
    
    logger.info(f"Computing semantic metrics for {args.right_label}...")
    metrics2 = compute_semantic_metrics(records2, args.model)
    metrics2.label = args.right_label
    
    # 格式化输出
    table = format_table([metrics1, metrics2])
    
    print("\n" + "="*80)
    print("SRSFact 语义对比分析")
    print("="*80)
    print(table)
    print("="*80)
    
    # 保存 JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            args.left_label: asdict(metrics1),
            args.right_label: asdict(metrics2),
            "left_result_file": jsonl1.name,
            "right_result_file": jsonl2.name,
            "paired_claim_count": common_count,
            "comparison": {
                "defense_score_delta": metrics1.defense_score - metrics2.defense_score,
                "asr_delta": metrics1.asr - metrics2.asr,
                "sfr_delta": metrics1.sfr - metrics2.sfr,
                "sim_before_after_delta": metrics1.avg_sim_before_after - metrics2.avg_sim_before_after,
                "p_fake_top1_delta": metrics1.p_fake_top1 - metrics2.p_fake_top1,
                "avg_fake_usage_rate_delta": metrics1.avg_fake_usage_rate - metrics2.avg_fake_usage_rate,
                "avg_extra_after_delta": metrics1.avg_extra_after - metrics2.avg_extra_after,
            }
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON report to {output_path}")


if __name__ == "__main__":
    main()