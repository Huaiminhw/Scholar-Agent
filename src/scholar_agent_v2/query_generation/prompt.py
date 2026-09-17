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
4. A retrieval query is a recall probe, not the full final-selection expression. final_selection_requirements are enforced downstream; do NOT pack every requirement into every query.
5. Keep at least one core entity or relation anchor in each query. Across variants, cover complementary semantic angles rather than repeating the same constraint bundle.
6. Omit filter-like metadata (year, venue, author, publication type, citation count, field, language) from query text. For any_of, prefer separate alternatives instead of combining them as if all were required.
7. Preserve an essential relation when that relation is the angle being expanded. Do not reduce it to unrelated co-occurring keywords.
8. Soft preferences must not become mandatory terms. Do not add survey/review terms unless explicitly requested.
9. Prefer concise English academic phrases of 3-12 informative terms; preserve proper names and confirmed aliases. Queries must be meaningfully different.
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
                "type": x.type.value,
                "operator": x.operator.value,
                "value": x.value,
                "values": x.values,
            }
            for x in items
        ]

    return {
        "final_selection_requirements": simple(analysis.constraints.hard),
        "soft_preferences": simple(analysis.constraints.soft),
        "exclusions": simple(analysis.constraints.exclude),
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
        "content": analysis.content,
        "query_type": analysis.query_type.value,
        "intent": {
            "task": analysis.intent.task.value,
            "mode": analysis.intent.mode.value,
        },
        "entities": entities,
        "relations": relations,
        "constraints": _constraint_summary(analysis),
        "already_generated_queries": existing_queries,
        "max_new_queries": max_queries,
    }

    return (
        "Generate additional scholarly search queries from this structured query analysis.\n"
        "Treat final_selection_requirements as downstream boundaries, not a checklist "
        "for every query. Do not repeat already_generated_queries.\n\n"
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
