from __future__ import annotations

import json
from dataclasses import dataclass

from ..query_understanding.schema import QueryAnalysisResult

from .models import LLMExpansionPayload


QUERY_EXPANSION_SYSTEM_PROMPT = r"""
You generate search queries for scholarly-paper retrieval.
Your goal is RECALL, but you must preserve the user's original academic intent.

RULES:
1. Generate only high-confidence alternative academic phrasings/synonyms.
2. Do NOT add new authors, venues, years, datasets, named methods, named models, tasks, or publication types that are not supported by the input.
3. Do NOT narrow a broad method into specific submethods (e.g. reinforcement learning -> PPO/GRPO/DQN) unless that exact submethod is already present in the input.
4. Preserve essential relation semantics (e.g. method X optimizes task/system Y). Do not turn a required relation into unrelated co-occurring keywords.
5. Soft preferences must not become mandatory search terms.
6. Do NOT add survey/review/state-of-the-art terms unless the user's task explicitly asks for a literature review/overview or publication type review/survey.
7. Prefer concise English academic search phrases when appropriate, while preserving proper names and high-confidence aliases.
8. Each query should usually contain 3-12 informative terms. Avoid full natural-language questions.
9. Queries must be meaningfully different; no near-duplicates.
10. Output JSON only.
""".strip()


@dataclass(frozen=True)
class ExpansionPromptConfig:
    max_queries: int = 2


def _constraint_summary(analysis: QueryAnalysisResult) -> dict:
    """
    只给 LLM 必要的边界信息，不重复传整个 QueryAnalysisResult，控制 Token。
    """

    def simple(items):
        return [
            {
                "id": x.id,
                "type": x.type.value,
                "operator": x.operator.value,
                "value": x.value,
                "values": x.values,
                "source_text": x.source_text,
            }
            for x in items
        ]

    return {
        "hard": simple(analysis.constraints.hard),
        "soft": simple(analysis.constraints.soft),
        "exclude": simple(analysis.constraints.exclude),
    }


def build_expansion_user_prompt(
    analysis: QueryAnalysisResult,
    *,
    existing_queries: list[str],
    max_queries: int,
) -> str:
    entities = [
        {
            "type": e.type.value,
            "canonical_name": e.canonical_name,
            "aliases": e.aliases,
        }
        for e in analysis.entities
    ]

    relations = [
        {
            "id": r.id,
            "subject": r.subject,
            "predicate": r.predicate,
            "object": r.object,
            "level": r.level.value,
        }
        for r in analysis.relations
    ]

    compact_input = {
        "original_query": analysis.original_query,
        "content": analysis.content,
        "query_type": analysis.query_type.value,
        "intent": {
            "task": analysis.intent.task.value,
            "mode": analysis.intent.mode.value,
            "goal": analysis.intent.goal,
        },
        "entities": entities,
        "relations": relations,
        "constraints": _constraint_summary(analysis),
        "already_generated_queries": existing_queries,
        "max_new_queries": max_queries,
    }

    return (
        "Generate additional scholarly search queries from this structured query analysis.\n"
        "Do not repeat already_generated_queries.\n\n"
        + json.dumps(compact_input, ensure_ascii=False, separators=(",", ":"))
        + "\n\nReturn exactly: "
        + '{"expanded_queries":[{"query":"...","reason":"..."}]}'
    )


def build_expansion_messages(
    analysis: QueryAnalysisResult,
    *,
    existing_queries: list[str],
    max_queries: int,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": QUERY_EXPANSION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_expansion_user_prompt(
                analysis,
                existing_queries=existing_queries,
                max_queries=max_queries,
            ),
        },
    ]


def get_llm_expansion_json_schema() -> dict:
    return LLMExpansionPayload.model_json_schema()


__all__ = [
    "ExpansionPromptConfig",
    "QUERY_EXPANSION_SYSTEM_PROMPT",
    "build_expansion_user_prompt",
    "build_expansion_messages",
    "get_llm_expansion_json_schema",
]
