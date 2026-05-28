"""Run InFact on a subset of FEVER claims.

Examples:
  python -m scripts.fever.evaluate_subset --variant dev --n-samples 50
  python -m scripts.fever.evaluate_subset --variant dev --sample-ids 1,7,42
  python -m scripts.fever.evaluate_subset --variant dev --n-samples 100 --random-sampling true
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import set_start_method

from config.globals import data_base_dir
from infact.eval.evaluate import evaluate


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_sample_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    chunks = [c.strip() for c in raw.split(",") if c.strip()]
    if not chunks:
        return None
    try:
        return [int(x) for x in chunks]
    except ValueError as exc:
        raise ValueError("--sample-ids must be a comma-separated integer list, e.g. 1,7,42") from exc


def assert_fever_inputs_ready(version: int, variant: str) -> None:
    claims_path = os.path.join(data_base_dir, "FEVER", f"fever{version}_{variant}.jsonl")
    wiki_db_path = os.path.join(data_base_dir, "FEVER", "wiki.db")

    missing = []
    if not os.path.exists(claims_path):
        missing.append(claims_path)
    if not os.path.exists(wiki_db_path):
        missing.append(wiki_db_path)

    if missing:
        lines = ["Missing required FEVER files:"] + [f"  - {p}" for p in missing]
        lines += [
            "",
            "Expected FEVER layout under data/FEVER/:",
            f"  - fever{version}_{variant}.jsonl",
            "  - wiki.db",
            "",
            "Note: FEVER benchmark in this codebase uses wiki_dump retrieval, so wiki.db is required.",
        ]
        raise FileNotFoundError("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="gemini_25_flash")
    parser.add_argument("--procedure-variant", type=str, default="infact")
    parser.add_argument("--variant", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--version", type=int, default=1, choices=[1, 2])

    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of claims to evaluate. Ignored when --sample-ids is used.")
    parser.add_argument("--sample-ids", type=str, default=None,
                        help="Comma-separated FEVER claim IDs, e.g. 1,7,42")
    parser.add_argument("--random-sampling", type=str, default="false")

    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-result-len", type=int, default=64000)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--n-workers", type=int, default=2)

    args = parser.parse_args()

    sample_ids = parse_sample_ids(args.sample_ids)
    random_sampling = parse_bool(args.random_sampling)

    assert_fever_inputs_ready(version=args.version, variant=args.variant)

    try:
        set_start_method("spawn")
    except RuntimeError:
        # Start method can already be set in some environments.
        pass

    evaluate(
        llm=args.llm,
        tools_config=dict(
            searcher=dict(
                search_engine_config=dict(
                    wiki_dump=dict(),
                ),
                limit_per_search=5,
            )
        ),
        fact_checker_kwargs=dict(
            procedure_variant=args.procedure_variant,
            max_iterations=args.max_iterations,
            max_result_len=args.max_result_len,
        ),
        llm_kwargs=dict(temperature=args.temperature),
        benchmark_name="fever",
        benchmark_kwargs=dict(version=args.version, variant=args.variant),
        n_samples=None if sample_ids else args.n_samples,
        sample_ids=sample_ids,
        random_sampling=random_sampling,
        print_log_level="info",
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
