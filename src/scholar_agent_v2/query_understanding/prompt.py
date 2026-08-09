# query_understanding/prompt.py
"""
Compact Query Understanding prompt.

兼容现有：
- schema.py
- analyzer.py
- validator.py
- tests/test_analyzer.py

目标：
- 保留影响检索 F1 的核心语义规则
- 减少主 Prompt Token
- 主路径仍尽量 1 次 LLM
- Schema/业务错误仍交给 validator + Repair
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .schema import QueryAnalysisResult


SchemaMode = Literal["none", "summary", "full"]


# ============================================================
# 1. Compact Schema Summary
# ============================================================

QUERY_ANALYSIS_SCHEMA_SUMMARY = r"""
Return exactly one JSON object with ALL required fields:

{
  "schema_version": "1.0",
  "status": "success|needs_clarification|unsupported",
  "original_query": "string",
  "query_type": "BROAD_BY_DESCRIPTION|SPECIFIC_BY_TITLE|SPECIFIC_BY_NAME|SPECIFIC_BY_IDENTIFIER|BY_AUTHOR|METADATA_ONLY_NO_AUTHOR|UNKNOWN",
  "intent": {
    "task": "paper_search|paper_lookup|author_search|citation_search|literature_review",
    "mode": "comprehensive|precise|latest|seminal|overview|specific",
    "goal": "string"
  },
  "content": "string",
  "constraints": {
    "hard": [ConstraintItem],
    "soft": [ConstraintItem],
    "exclude": [ConstraintItem]
  },
  "relations": [RelationConstraint],
  "logic": {"hard": LogicNode|null, "soft": LogicNode|null, "exclude": LogicNode|null},
  "properties": {
    "recent_first": boolean,
    "recent_last": boolean,
    "central_first": boolean,
    "central_last": boolean,
    "specific_paper_name": "string|null",
    "suitable_for_by_citing": "boolean|null"
  },
  "entities": [{
    "type": "topic|method|model|dataset|author|venue|paper|other",
    "canonical_name": "string",
    "aliases": ["string"],
    "source_text": "string"
  }],
  "domains": {"main_field": "string|null", "secondary_fields": ["string"]},
  "clarification": {
    "needed": boolean,
    "unresolved_ambiguities": ["string"],
    "questions": ["string"]
  },
  "result_requirements": {
    "requested_top_k": "integer|null",
    "required_fields": ["string"],
    "output_structure": "list|table|relation_graph|mixed|unspecified",
    "need_relation_graph": boolean,
    "need_open_access": boolean,
    "need_code": boolean
  },
  "warnings": ["string"]
}

ConstraintItem:
{
  "id": "C*/P*/E*",
  "type": "topic|task|method|technology|dataset|evaluation|author|venue|year|publication_type|min_citations|field_of_study|language|identifier|code_availability|data_availability|other",
  "operator": "eq|ne|gt|gte|lt|lte|any_of|all_of|contains|not_contains|between",
  "value": "string|integer|float|boolean|null",
  "values": ["scalar"],
  "source_text": "string",
  "normalized_text": "string|null"
}

RelationConstraint:
{"id":"R*","subject":"string","predicate":"string","object":"string","level":"hard|soft|exclude","source_text":"string"}

LogicNode:
LEAF={"operator":"LEAF","ref_id":"C1","children":[]}
AND/OR={"operator":"AND|OR","ref_id":null,"children":[LogicNode,...]}
NOT={"operator":"NOT","ref_id":null,"children":[one LogicNode]}
"""


# ============================================================
# 2. Compact Main Prompt
# ============================================================

QUERY_ANALYSIS_SYSTEM_PROMPT_TEMPLATE = r"""
You are the Query Understanding component of a scholarly paper search system.
Convert the user's request into QueryAnalysisResult JSON for downstream retrieval.
Do NOT search for papers, generate search-engine queries, or invent requirements.

Current year: __CURRENT_YEAR__

=== 1. CORE RULES ===
- Be faithful to the user's wording. Normalize only high-confidence names/abbreviations.
- "content" contains scientific meaning only; move author/year/venue/result-format/open-access/code requirements to their structured fields.
- Preserve important semantic relations; co-occurring keywords are not equivalent to a relation.
- hard = required; soft = explicit preference ("prefer/最好/优先/尽量/if possible");
  exclude = explicit negative ("不要/exclude/without/not").
- Never turn soft into hard.
- Default hard logic is AND. Default exclusion means any matching exclude condition can reject.
- Use explicit logic trees only when grouping/OR/NOT matters.
- Output JSON only, with no extra keys or explanation.

=== 2. QUERY TYPE ===
Choose one:
- SPECIFIC_BY_IDENTIFIER: DOI/arXiv/PMID/Corpus ID or other unique paper identifier.
- SPECIFIC_BY_TITLE: one exact paper title.
- SPECIFIC_BY_NAME: one specific paper/resource located by a known alias/name; a named method/model alone is not enough.
- BY_AUTHOR: papers authored by specified author(s).
- METADATA_ONLY_NO_AUTHOR: usable metadata constraints but essentially no topical content and no author requirement.
- BROAD_BY_DESCRIPTION: normal topic/method/task/benchmark/citation-description search.
- UNKNOWN: cannot reliably represent as academic-paper search.

Priority:
identifier > exact title > explicit author search > specific alias/name > metadata-only > broad.

=== 3. INTENT & PROPERTIES ===
intent.task:
paper_lookup=one specific paper;
paper_search=matching papers;
author_search=author-centered;
citation_search=citation relation is central;
literature_review=explicit review/synthesis request.

intent.mode:
specific=exact lookup;
overview=review/survey;
latest=explicit recent/latest;
seminal=explicit classic/influential/highly-cited;
precise=explicit strict/exact match;
comprehensive=ordinary/broad coverage default.

Properties:
- recent_first=true ONLY for explicit recent/latest/newest/current; an absolute year such as "since 2023" alone does NOT imply it.
- recent_last=true only for earliest/older/foundational preference.
- central_first=true for classic/seminal/influential/highly-cited preference.
- central_last=true only for explicit lesser-known/low-citation preference.
- Never set opposite flags true together.
- specific_paper_name: central clearly named paper/resource, else null.
- suitable_for_by_citing=true only when citation expansion from a named resource naturally matches the request; otherwise false/null.

=== 4. CONSTRAINTS ===
Extract only conditions that actually constrain desired papers.

IDs:
hard C1,C2...; soft P1,P2...; exclude E1,E2...; each ID unique.

Single-value operators use "value" and values=[].
any_of/all_of/between use "values" and value=null.
between has exactly two endpoints.

Important mappings:
- "since 2023"/"2023年以来" -> year gte 2023
- "after 2023" -> year gt 2023
- "before 2020" -> year lt 2020
- "2020-2024" -> year between [2020,2024]
- "in 2024" -> year eq 2024
- recent/latest without year -> properties.recent_first, not invented year
- "at least 100 citations" -> min_citations gte 100
- "prefer highly cited" -> soft min_citations eq "high" + central_first=true

Author:
only use author when the user requires papers written by that person; do not infer coauthors.

Venue:
separate venue from year. "ACL 2024" means venue=ACL AND year=2024.
Several interchangeable venues may use any_of unless venue-year pairing must be preserved.

Publication type:
add only when explicit (e.g. survey/review).

Code/data availability:
use hard/soft constraint only if availability is a candidate-paper condition.
If user merely requests links in the output, use result_requirements instead.

=== 5. RELATIONS ===
Create RelationConstraint when the connection is necessary for relevance, e.g.:
- reinforcement learning -> optimize -> LLM search agent
- method X -> applied_to -> task Y
- model X -> evaluated_on -> dataset Y
- target paper -> cites/cited_by -> paper X
- method A -> compared_with -> method B

Relation level follows hard/soft/exclude.
Do not create trivial topic-membership relations.
Do not split an essential relation into independent keywords.
Avoid redundant hard constraints when the relation already fully expresses the same requirement; keep independently meaningful year/venue/dataset/evaluation constraints.

=== 6. BOOLEAN LOGIC ===
Use an explicit logic tree only when grouping matters.

Critical example:
"ACL 2024 or NeurIPS 2023"
MUST preserve paired alternatives:
OR(
  AND(venue=ACL, year=2024),
  AND(venue=NeurIPS, year=2023)
)
Do NOT flatten to venue=[ACL,NeurIPS] AND year=[2023,2024].

Logic leaves may reference constraint IDs or relation IDs.
Use NOT only when exclude cannot represent the negative group cleanly.

=== 7. ENTITIES, DOMAIN, CLARIFICATION ===
Entities:
extract only useful downstream entities. Use canonical names/aliases only when high-confidence.
Never expand a broad class into specific methods (e.g. reinforcement learning -> PPO/GRPO/DQN is forbidden).
Preserve source_text.

Domain:
choose one broad main_field and only materially relevant secondary fields; use "Unknown" if unclear.

Clarification:
prefer best-effort parsing.
Use status=needs_clarification only when genuine ambiguity would materially change retrieval and cannot be safely preserved.
If needed: clarification.needed=true and ask at most 2 short questions.
Otherwise needed=false and questions=[].
Use warnings for non-blocking uncertainty.

Status:
success=usable for retrieval;
needs_clarification=critical ambiguity blocks reliable retrieval;
unsupported=no reasonable academic-paper-search interpretation.

=== 8. RESULT REQUIREMENTS ===
- requested_top_k only if user explicitly gives a number.
- required_fields only for explicitly requested fields (title/authors/year/doi/url/etc.).
- output_structure: list/table/relation_graph/mixed/unspecified.
- need_relation_graph only when explicitly requested.
- need_open_access=true for accessible/free/full-text/open-access result requests.
- need_code=true for requested code/repository links.

=== 9. FINAL CHECK ===
Before output, silently ensure:
1. Every constraint/relation is grounded in the user's query.
2. Soft preferences remain soft; negatives are preserved.
3. Essential relations and explicit OR pairings are preserved.
4. IDs are unique and logic references existing IDs.
5. No invented author/venue/dataset/method/year.
6. Output is directly usable by downstream retrieval.
"""


# ============================================================
# 3. Optional Few-shot
# ============================================================

FEW_SHOT_EXAMPLES = r"""
EXAMPLE 1
Query: 找2023年以来使用强化学习优化LLM学术搜索Agent，并在真实科研查询数据集上验证的论文，最好来自ACL、NeurIPS。
Interpretation:
year>=2023 hard; real-query evaluation hard; ACL/NeurIPS soft;
preserve RL -> optimize -> LLM academic-search agent;
BROAD_BY_DESCRIPTION; recent_first=false.

EXAMPLE 2
Query: 找 ACL 2024 或 NeurIPS 2023 的多模态大模型论文，不要综述。
Interpretation:
OR(AND(ACL,2024),AND(NeurIPS,2023));
exclude publication_type=survey/review.

EXAMPLE 3
Query: 找 Dan Weld 写的 automated planning 论文，优先高引用，给我10篇并附 DOI。
Interpretation:
BY_AUTHOR; author hard; topic hard; high-citation soft + central_first=true;
requested_top_k=10; required_fields includes doi.
"""


# ============================================================
# 4. User Prompt
# ============================================================

QUERY_ANALYSIS_USER_PROMPT = r"""
Analyze this scholarly-paper query.

USER QUERY:
<<<__USER_QUERY__>>>

Return only the final JSON object.
"""


# ============================================================
# 5. Repair Prompt
# ============================================================

QUERY_REPAIR_SYSTEM_PROMPT = r"""
Repair the previous scholarly-query JSON with minimal changes.
Make it valid QueryAnalysisResult while remaining faithful to the original query.
Fix schema/type/enum/ID/logic/flag errors only; do not invent requirements.
Preserve valid information. Return JSON only.
"""


QUERY_REPAIR_USER_PROMPT = r"""
ORIGINAL QUERY:
<<<__USER_QUERY__>>>

PREVIOUS OUTPUT:
<<<__RAW_OUTPUT__>>>

VALIDATION ERROR:
<<<__VALIDATION_ERROR__>>>

TARGET SCHEMA:
__SCHEMA__

Return only corrected JSON.
"""


# ============================================================
# 6. Public API -- 保持与原 prompt.py 一致
# ============================================================

@dataclass(frozen=True)
class PromptConfig:
    schema_mode: SchemaMode = "summary"
    include_examples: bool = False
    current_year: int | None = None


def get_query_analysis_json_schema() -> dict:
    return QueryAnalysisResult.model_json_schema()


def get_query_analysis_json_schema_text(indent: int = 2) -> str:
    return json.dumps(
        get_query_analysis_json_schema(),
        ensure_ascii=False,
        indent=indent,
    )


def build_query_analysis_system_prompt(
    config: PromptConfig | None = None,
) -> str:
    config = config or PromptConfig()
    current_year = config.current_year or datetime.now().year

    prompt = QUERY_ANALYSIS_SYSTEM_PROMPT_TEMPLATE.replace(
        "__CURRENT_YEAR__",
        str(current_year),
    ).strip()

    if config.schema_mode == "summary":
        prompt += "\n\n=== OUTPUT SCHEMA SUMMARY ===\n"
        prompt += QUERY_ANALYSIS_SCHEMA_SUMMARY.strip()

    elif config.schema_mode == "full":
        prompt += "\n\n=== FULL OUTPUT JSON SCHEMA ===\n"
        prompt += get_query_analysis_json_schema_text()

    if config.include_examples:
        prompt += "\n\n=== FEW-SHOT EXAMPLES ===\n"
        prompt += FEW_SHOT_EXAMPLES.strip()

    return prompt


def build_query_analysis_user_prompt(user_query: str) -> str:
    return QUERY_ANALYSIS_USER_PROMPT.replace(
        "__USER_QUERY__",
        user_query.strip(),
    ).strip()


def build_query_analysis_messages(
    user_query: str,
    config: PromptConfig | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": build_query_analysis_system_prompt(config),
        },
        {
            "role": "user",
            "content": build_query_analysis_user_prompt(user_query),
        },
    ]


def build_query_repair_messages(
    *,
    user_query: str,
    raw_output: str,
    validation_error: str,
    schema_mode: SchemaMode = "full",
) -> list[dict[str, str]]:
    if schema_mode == "none":
        schema_text = "Use the externally enforced QueryAnalysisResult schema."
    elif schema_mode == "summary":
        schema_text = QUERY_ANALYSIS_SCHEMA_SUMMARY.strip()
    else:
        schema_text = get_query_analysis_json_schema_text()

    user_prompt = (
        QUERY_REPAIR_USER_PROMPT
        .replace("__USER_QUERY__", user_query.strip())
        .replace("__RAW_OUTPUT__", raw_output.strip())
        .replace("__VALIDATION_ERROR__", validation_error.strip())
        .replace("__SCHEMA__", schema_text)
        .strip()
    )

    return [
        {
            "role": "system",
            "content": QUERY_REPAIR_SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


__all__ = [
    "PromptConfig",
    "QUERY_ANALYSIS_SCHEMA_SUMMARY",
    "QUERY_ANALYSIS_SYSTEM_PROMPT_TEMPLATE",
    "QUERY_ANALYSIS_USER_PROMPT",
    "QUERY_REPAIR_SYSTEM_PROMPT",
    "QUERY_REPAIR_USER_PROMPT",
    "FEW_SHOT_EXAMPLES",
    "get_query_analysis_json_schema",
    "get_query_analysis_json_schema_text",
    "build_query_analysis_system_prompt",
    "build_query_analysis_user_prompt",
    "build_query_analysis_messages",
    "build_query_repair_messages",
]
