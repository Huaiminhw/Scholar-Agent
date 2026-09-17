from __future__ import annotations

import json
from collections.abc import Sequence

from ..candidate_pool.models import CandidatePaper
from .models import SelectorCriterion


SELECTOR_SYSTEM_PROMPT = r"""
You are the semantic Selector in an academic-paper retrieval system.
Your job is NOT to rank papers and NOT to reward keyword overlap. Judge each paper against EACH criterion independently.

Evidence policy:
1. Use ONLY the supplied title and abstract. Do not use outside knowledge.
2. PASS means the title/abstract gives concrete evidence that the criterion is satisfied.
3. FAIL means the title/abstract gives strong evidence that the criterion is not satisfied or that the paper's actual scientific focus is incompatible with it.
4. UNKNOWN means the supplied title/abstract is insufficient. Missing mention alone is NOT enough for FAIL when the property could plausibly appear only in the full paper.
5. For relation criteria, preserve subject, predicate, object, and direction. Co-occurrence is not a relation.
6. For PASS or FAIL, copy one short exact evidence span from title or abstract whenever possible. Evidence must be verbatim, no paraphrase, preferably <= 30 words. If there is no trustworthy exact span, use UNKNOWN.
7. Never invent methods, datasets, evaluations, claims, or evidence.
8. Judge the criterion itself even when its level is "exclude". A PASS on an exclude criterion means the paper DOES match the excluded property; downstream deterministic logic will reject it.
9. Return every requested candidate_id exactly once and every requested criterion_id exactly once. Do not create extra IDs.

Allowed status values: "pass", "fail", "unknown".

Return JSON only in this shape:
{
  "papers": [
    {
      "candidate_id": "...",
      "summary": "short factual summary or null",
      "judgements": [
        {
          "criterion_id": "...",
          "status": "pass|fail|unknown",
          "evidence": "exact title/abstract span or null",
          "reason": "brief reasoning",
          "confidence": 0.0
        }
      ]
    }
  ]
}
""".strip()


SELECTOR_REPAIR_SYSTEM_PROMPT = r"""
Repair the malformed Selector JSON below. Preserve the intended judgements but make the output strictly valid JSON.
Do not add candidate IDs or criterion IDs that were not requested. Allowed status values are pass, fail, unknown.
Return JSON only.
""".strip()


def _criterion_payload(criterion: SelectorCriterion) -> dict:
    payload = {
        "criterion_id": criterion.criterion_id,
        "kind": criterion.kind.value,
        "level": criterion.level.value,
        "description": criterion.description,
        "source_text": criterion.source_text,
    }
    if criterion.constraint_type is not None:
        payload["constraint_type"] = criterion.constraint_type.value
    if criterion.operator is not None:
        payload["operator"] = criterion.operator.value
    if criterion.expected_values:
        payload["expected_values"] = criterion.expected_values
    if criterion.kind.value == "relation":
        payload["subject"] = criterion.subject
        payload["predicate"] = criterion.predicate
        payload["object"] = criterion.object
    return payload


def _paper_payload(candidate: CandidatePaper) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "title": candidate.title or "",
        "abstract": candidate.abstract or "",
    }


def build_selector_messages(
    candidates: Sequence[CandidatePaper],
    criteria: Sequence[SelectorCriterion],
) -> list[dict[str, str]]:
    request = {
        "criteria": [_criterion_payload(criterion) for criterion in criteria],
        "papers": [_paper_payload(candidate) for candidate in candidates],
    }
    return [
        {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def build_selector_repair_messages(
    *,
    raw_output: str,
    candidates: Sequence[CandidatePaper],
    criteria: Sequence[SelectorCriterion],
    error: str,
) -> list[dict[str, str]]:
    requested = {
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "criterion_ids": [criterion.criterion_id for criterion in criteria],
        "validation_error": error,
        "raw_output": raw_output,
    }
    return [
        {"role": "system", "content": SELECTOR_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(requested, ensure_ascii=False),
        },
    ]


__all__ = [
    "SELECTOR_SYSTEM_PROMPT",
    "SELECTOR_REPAIR_SYSTEM_PROMPT",
    "build_selector_messages",
    "build_selector_repair_messages",
]
