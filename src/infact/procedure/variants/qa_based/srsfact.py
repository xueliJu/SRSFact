import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from infact.common import FCDocument, Action, Label, SearchResult
from infact.common.action import WebSearch
from infact.procedure.variants.qa_based.infact import InFact
from infact.prompts.prompt import Prompt
from infact.utils.parsing import find_code_span


class StrategyAndQueriesPrompt(Prompt):
    template_text = """# Instructions
You are a fact-checker. Your overall motivation is to verify a given Claim. You are at the beginning of the fact-check, i.e. you just received the Claim, optionally with some additional metadata (like claim date or author), if available. **Your task right now is to prepare the fact-check.** That is,

1. **Verification Strategy**: Briefly analyze the claim. Identify the key entities, specific relationships, dates, or numerical claims that must be verified. Outline a concise strategy for verification.
2. **Questions and Search Queries**: Based on your strategy, propose [N_QUESTIONS] pairs of specific questions and corresponding search queries.

IMPORTANT: Follow these rules:
* State every single question in a way that it can be understood independently and without additional context. Therefore, be explicit and do not use pronouns or generic terms in place of names or objects.
* Enclose each single question with backticks like `this`.
* Enclose each single search query with backticks like `this`.
* The Search Query should be keyword-optimized suitable for a search engine to find the answer to the question.

# Examples
Claim: "New Zealand’s new Food Bill bans gardening"
Strategy: The claim is about a specific bill in New Zealand. I need to check if such a bill exists and if it contains provisions banning gardening.
1. Question: `Did New Zealand's government pass a food bill that restricted gardening activities for its citizen?`
   Query: `New Zealand Food Bill gardening ban`

2. Question: `What are the provisions of New Zealand's Food Bill regarding home gardening?`
   Query: `New Zealand Food Bill home gardening provisions`

# The Claim
[CLAIM]

# Verification Strategy
"""

    def __init__(self, doc: FCDocument, n_questions: int = 8):
        placeholder_targets = {
            "[CLAIM]": doc.claim,
            "[N_QUESTIONS]": n_questions,
        }
        super().__init__(placeholder_targets=placeholder_targets, text=self.template_text)

    def get_template(self) -> str:
        return self.template_text


class SRSFact(InFact):
    """
    SRSFact: improved InFact with strategy-guided Q/A generation.
    Semantic guard is currently disabled/removed.
    """

    def __init__(
            self,
            srs: bool = False,
            srs_n_questions: int = 10,
            srs_top_k: int = 2,
            semantic_guard_enabled: bool = False,
            semantic_guard_model: str = "all-mpnet-base-v2",
            semantic_guard_fixed_floor: float = 0.80,
            semantic_guard_quantile: float = 0.30,
            semantic_guard_stats_path: str = None,
            semantic_guard_min_pass: int = 1,
            semantic_guard_top_k: int = 3,
            semantic_guard_batch_size: int = 64,
            semantic_guard_on_qquery: bool = False,
            semantic_guard_on_evidence: bool = True,
            semantic_guard_low_confidence_keep_all: bool = True,
            semantic_guard_low_confidence_max_score_margin: float = 0.0,
            semantic_guard_similarity_combine: str = "max",
            semantic_guard_similarity_weight_qe: float = 0.5,
            semantic_guard_injection_filter_mode: str = "off",
            semantic_guard_injection_downweight: float = 0.20,
            semantic_guard_min_clean_sources: int = 0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.srs_enabled = self._as_bool(srs)
        self.srs_n_questions = max(1, int(srs_n_questions))
        self.srs_top_k = max(1, int(srs_top_k))
        # Keep kwargs in signature for backward compatibility, but disable guard behavior.
        _ = (
            semantic_guard_enabled,
            semantic_guard_model,
            semantic_guard_fixed_floor,
            semantic_guard_quantile,
            semantic_guard_stats_path,
            semantic_guard_min_pass,
            semantic_guard_top_k,
            semantic_guard_batch_size,
            semantic_guard_on_qquery,
            semantic_guard_on_evidence,
            semantic_guard_low_confidence_keep_all,
            semantic_guard_low_confidence_max_score_margin,
            semantic_guard_similarity_combine,
            semantic_guard_similarity_weight_qe,
            semantic_guard_injection_filter_mode,
            semantic_guard_injection_downweight,
            semantic_guard_min_clean_sources,
        )
        self.semantic_guard_on_qquery = False
        self.semantic_guard_on_evidence = False
        self.semantic_guard = None

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        self._query_cache = {}
        # Cache SRS review by (source URL, question) within one claim — same URL may appear under different questions.
        self._srs_review_cache: dict[tuple[str, str], dict[str, Any]] = {}
        questions = self._pose_questions(no_of_questions=self.srs_n_questions, doc=doc)
        q_and_a, search_results = self.approach_question_batch(questions, doc)
        label = self.judge.judge(doc)
        return label, dict(q_and_a=q_and_a, used_evidence=search_results)

    def _pose_questions(self, no_of_questions: int, doc: FCDocument) -> list[str]:
        """
        Generate question-query pairs once, then cache per question.
        """
        prompt = StrategyAndQueriesPrompt(doc, n_questions=no_of_questions)
        response = self.llm.generate(prompt)
        matches = find_code_span(response)

        # Pair extraction: question, query, question, query ...
        pairs: list[tuple[str, str]] = []
        for i in range(0, len(matches) - 1, 2):
            question = matches[i]
            query = matches[i + 1]
            pairs.append((question, query))

        # Optional question-query semantic guard
        if self.semantic_guard is not None and self.semantic_guard_on_qquery:
            claim_text = str(doc.claim)
            filtered_pairs, detail = self.semantic_guard.filter_question_query_pairs(claim_text, pairs)
            if detail:
                self.logger.info(
                    "Semantic guard on question-query pairs: "
                    f"before={detail['before']}, after={detail['after']}, "
                    f"threshold={detail['threshold']:.4f}, "
                    f"mean_score={detail['mean_score_before']:.4f}, "
                    f"max_score={detail.get('max_score', -1.0):.4f}, "
                    f"bypass={detail.get('bypass_reason', 'none')}"
                )
            pairs = filtered_pairs

        questions = []
        for question, query in pairs:
            questions.append(question)
            self._query_cache[question] = query

        return questions

    def propose_queries_for_question(self, question: str, doc: FCDocument) -> list[Action]:
        """
        Retrieve pre-generated query from cache to avoid extra LLM calls.
        """
        if hasattr(self, "_query_cache") and question in self._query_cache:
            query_str = self._query_cache[question]
            return [WebSearch(f'"{query_str}"')]

        return [WebSearch(f'"{question}"')]

    def approach_question(self, question: str, doc: FCDocument = None):
        if not self.srs_enabled:
            return super().approach_question(question, doc)

        self.logger.debug(f"SRS enabled for question: {question}")
        self.actor.reset()
        queries = self.propose_queries_for_question(question, doc)
        if queries is None or len(queries) == 0:
            return None, []

        search_results = self.retrieve_resources(queries)
        search_results = self._apply_semantic_guard_to_results(question, search_results, doc)
        srs_start = time.perf_counter()
        reviewed_results, review_map = self._review_and_repair_results(question, doc, search_results)
        srs_elapsed = time.perf_counter() - srs_start
        self.logger.info(
            f"SRS timing - question='{question[:80]}...' "
            f"evaluated_top_k={min(len(search_results), self.srs_top_k)} "
            f"total_results={len(search_results)} "
            f"elapsed={srs_elapsed:.2f}s"
        )

        if len(reviewed_results) == 0:
            return None, []

        qa_instance = self.generate_answer(question, reviewed_results, doc)
        if qa_instance is None:
            return None, []

        review = review_map.get(qa_instance.get("url"))
        if review:
            source_text = ""
            for result in reviewed_results:
                if result.source == qa_instance.get("url"):
                    source_text = (result.text or "").strip()
                    break
            source_text = source_text[:600]
            qa_instance["evidence_assessment"] = review.get("overall_evaluation", "")
            qa_instance["supporter_reason"] = review.get("supporter_reason", "")
            qa_instance["refuter_reason"] = review.get("refuter_reason", "")
            qa_instance["evidence_with_evaluation"] = (
                f"Evidence: {source_text}\n"
                f"Overall Evaluation: {review.get('overall_evaluation', '')}"
            )
        return qa_instance, reviewed_results

    def _review_and_repair_results(
            self,
            question: str,
            doc: FCDocument,
            results: list[SearchResult]
    ) -> tuple[list[SearchResult], dict[str, dict[str, Any]]]:
        claim_text = str(doc.claim) if doc is not None else ""
        candidates = results[: self.srs_top_k]
        cache = getattr(self, "_srs_review_cache", {})
        review_map: dict[str, dict[str, Any]] = {}
        uncached_candidates: list[SearchResult] = []
        cache_hits = 0
        for result in candidates:
            cache_key = (result.source, question)
            cached = cache.get(cache_key)
            if cached is not None:
                review_map[result.source] = cached
                cache_hits += 1
            else:
                uncached_candidates.append(result)

        if uncached_candidates:
            uncached_review_map = self._run_srs_group_review(claim_text, question, uncached_candidates)
            for source, review in uncached_review_map.items():
                review_map[source] = review
                cache[(source, question)] = review
            self._srs_review_cache = cache

        self.logger.info(
            f"SRS cache - question='{question[:80]}...' "
            f"hits={cache_hits} misses={len(uncached_candidates)} "
            f"cache_size={len(cache)}"
        )
        # SRS in this mode only provides model assessment; no evidence filtering.
        return results, review_map

    def _run_srs_group_review(
            self,
            claim: str,
            question: str,
            results: list[SearchResult]
    ) -> dict[str, dict[str, Any]]:
        if not results:
            return {}

        evidence_lines = []
        for i, result in enumerate(results, start=1):
            evidence_lines.append(
                f"[{i}] source={result.source}\n"
                f"text={(result.text or '')[:1200]}"
            )
        evidence_block = "\n\n".join(evidence_lines)

        supporter_prompt = Prompt(text=f"""
You are Supporter.
Given claim, question, and a group of evidences, briefly explain why each evidence is valid and effective for answering the question.
Each support_reason MUST be at most 30 words.
Return STRICT JSON ONLY:
{{
  "items": [
    {{"source": "evidence source url", "support_reason": "short reason"}}
  ]
}}

Claim: {claim}
Question: {question}
Evidence Group:
{evidence_block}
""".strip())

        refuter_prompt = Prompt(text=f"""
You are Refuter.
Given claim, question, and a group of evidences, assess whether each evidence could potentially be fabricated, misleading, or unreliable. Provide a brief reason.
Each refute_reason MUST be at most 30 words.
Return STRICT JSON ONLY:
{{
  "items": [
    {{"source": "evidence source url", "refute_reason": "short reason"}}
  ]
}}

Claim: {claim}
Question: {question}
Evidence Group:
{evidence_block}
""".strip())
        # Run supporter/refuter in parallel, then feed both into summarizer.
        with ThreadPoolExecutor(max_workers=2) as pool:
            supporter_future = pool.submit(self.llm.generate, supporter_prompt, max_attempts=1)
            refuter_future = pool.submit(self.llm.generate, refuter_prompt, max_attempts=1)
            supporter_reason = (supporter_future.result() or "").strip()
            refuter_reason = (refuter_future.result() or "").strip()

        supporter_map = self._parse_role_json(
            supporter_reason,
            key_field="support_reason",
            expected_sources=[r.source for r in results],
        )
        refuter_map = self._parse_role_json(
            refuter_reason,
            key_field="refute_reason",
            expected_sources=[r.source for r in results],
        )

        summarizer_prompt = Prompt(text=f"""
You are Summarizer.
Decide whether each evidence in the group is REAL/FAKE/UNCERTAIN for this question.
Use both supporter and refuter views.
Each overall_evaluation MUST be brief (max 25 words).
Return STRICT JSON ONLY:
{{
  "items": [
    {{
      "source": "evidence source url",
      "verdict": "REAL" | "FAKE" | "UNCERTAIN",
      "overall_evaluation": "short judgement",
      "confidence": 0.0
    }}
  ]
}}

Claim: {claim}
Question: {question}
Evidence Group:
{evidence_block}
Supporter View JSON: {supporter_reason}
Refuter View JSON: {refuter_reason}
""".strip())
        summary_raw = (self.llm.generate(summarizer_prompt, max_attempts=1) or "").strip()
        summary_map = self._parse_summary_group_json(summary_raw, [r.source for r in results])
        review_map: dict[str, dict[str, Any]] = {}
        for result in results:
            source = result.source
            parsed = summary_map.get(source, {})
            verdict = str(parsed.get("verdict", "UNCERTAIN")).upper()
            review_map[source] = {
                "supporter_reason": supporter_map.get(source, ""),
                "refuter_reason": refuter_map.get(source, ""),
                "overall_evaluation": parsed.get("overall_evaluation", summary_raw),
                "confidence": parsed.get("confidence", 0.0),
                "verdict": verdict,
                "is_fake": verdict == "FAKE",
            }
        return review_map

    def _parse_role_json(
            self,
            text: str,
            key_field: str,
            expected_sources: list[str],
    ) -> dict[str, str]:
        parsed = self._parse_json_object(text)
        result: dict[str, str] = {s: "" for s in expected_sources}
        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                source = item.get("source")
                if source in result:
                    result[source] = str(item.get(key_field, "")).strip()
        if not any(result.values()):
            preview = (text or "").strip().replace("\n", " ")[:200]
            self.logger.warning(
                "SRS role JSON parse produced no per-source fields; leaving supporter/refuter strings empty. "
                f"Response preview: {preview!r}"
            )
        return result

    def _parse_summary_group_json(
            self,
            text: str,
            expected_sources: list[str]
    ) -> dict[str, dict[str, Any]]:
        parsed = self._parse_json_object(text)
        result: dict[str, dict[str, Any]] = {
            s: {"verdict": "UNCERTAIN", "overall_evaluation": text, "confidence": 0.0}
            for s in expected_sources
        }
        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                source = item.get("source")
                if source in result:
                    result[source] = {
                        "verdict": str(item.get("verdict", "UNCERTAIN")).upper(),
                        "overall_evaluation": item.get("overall_evaluation", ""),
                        "confidence": item.get("confidence", 0.0),
                    }
        return result

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        if not text:
            return {}
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            candidate = candidate.replace("json", "", 1).strip()
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
        return {}
