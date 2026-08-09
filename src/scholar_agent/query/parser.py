from openai import OpenAI
import json
from scholar_agent.config import settings
from scholar_agent.utils.logger import logger

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    return _client


def parse_query(user_query: str) -> dict:
    """
    查询理解（Query Understanding）
    把用户自然语言查询解析成结构化检索条件
    """
    prompt = f"""You are an academic paper search expert. Parse the user query into structured search conditions.

User query: {user_query}

Rules:
- sub_queries must faithfully reflect the original query intent, do NOT change the domain
- If query mentions specific techniques, keep them exactly
- Generate sub_queries directly usable in search engines
- keywords should be specific technical terms

Output JSON only, no other text:
{{
  "core_topic": "short core topic",
  "sub_queries": ["sub query 1", "sub query 2", "sub query 3"],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "year_from": null,
  "methods": ["method1"],
  "datasets": [],
  "venues": []
}}"""

    response = _get_client().chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)
    logger.debug(f"Parsed query: core_topic={parsed.get('core_topic')}, sub_queries={len(parsed.get('sub_queries', []))}")
    return parsed
