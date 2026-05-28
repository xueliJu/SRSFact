from infact.eval.evaluate import evaluate
from multiprocessing import set_start_method
import argparse
import json

def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

if __name__ == '__main__':  # evaluation uses multiprocessing
    set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--procedure_variant", type=str, default="no_qa")
    parser.add_argument("--llm", type=str, default="gemini_25_flash")
    parser.add_argument("--variant", type=str, default="dev")
    parser.add_argument("--procedure_kwargs_json", type=str, default=None)
    parser.add_argument("--srs", type=str, default="false",
                        help="Enable SRS evidence review in srsfact procedure (true/false).")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Number of questions for srsfact (e.g. 5).")
    parser.add_argument("--srs_top_k", type=int, default=None,
                        help="SRS: number of top search hits to run group review on (srsfact only).")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Number of parallel workers for evaluation.")
    args = parser.parse_args()

    procedure_kwargs = {}
    if args.procedure_kwargs_json:
        try:
            procedure_kwargs = json.loads(args.procedure_kwargs_json)
        except Exception:
            procedure_kwargs = {}

    if args.procedure_variant.lower() == "srsfact":
        procedure_kwargs["srs"] = parse_bool(args.srs)
        if args.n_questions is not None:
            procedure_kwargs["srs_n_questions"] = int(args.n_questions)
        if args.srs_top_k is not None:
            procedure_kwargs["srs_top_k"] = int(args.srs_top_k)

    evaluate(
        llm=args.llm,
        tools_config=dict(searcher=dict(
            search_engine_config=dict(
                averitec_kb=dict(variant=args.variant),
            ),
            limit_per_search=5
        )),
        fact_checker_kwargs=dict(
            procedure_variant=args.procedure_variant,
            max_iterations=3,
            max_result_len=64_000,  # characters
            procedure_kwargs=procedure_kwargs,
        ),
        llm_kwargs=dict(temperature=0.01),
        benchmark_name="averitec",
        benchmark_kwargs=dict(variant=args.variant),
        random_sampling=False,
        print_log_level="info",
        n_workers=args.n_workers,
    )


# from infact.eval.evaluate import evaluate
# from multiprocessing import set_start_method

# variant = "dev"

# if __name__ == '__main__':  # evaluation uses multiprocessing
#     set_start_method("spawn")
#     evaluate(
#         llm="gpt_4o_mini",
#         tools_config=dict(searcher=dict(
#             search_engine_config=dict(
#                 averitec_kb=dict(variant=variant),
#             ),
#             limit_per_search=5
#         )),
#         fact_checker_kwargs=dict(
#             procedure_variant="no_qa",
#             max_iterations=3,
#             max_result_len=64_000,  # characters
#         ),
#         llm_kwargs=dict(temperature=0.01),
#         benchmark_name="averitec",
#         benchmark_kwargs=dict(variant=variant),
#         random_sampling=False,
#         print_log_level="info",
#         n_workers=4,
#     )
