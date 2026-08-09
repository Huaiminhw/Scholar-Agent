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


def evolve_query(original_query: str, found_papers: list[dict]) -> list[str]:
    """
    查询演化（Query Evolution）
    用已找到的相关论文，反向生成新的检索词，
    发现用原始关键词搜不到的论文。
    """
    if not found_papers:
        return []

    # 取前5篇高分论文
    top_papers = sorted(found_papers, key=lambda x: x.get('score', 0), reverse=True)[:5]

    papers_text = ""
    for p in top_papers:
        papers_text += f"- Title: {p['title']}\n  Abstract: {(p['abstract'] or '')[:200]}\n\n"

    prompt = f"""You are an academic search expert. Based on the original query and relevant papers found so far, generate NEW search queries to find MORE related papers that might have been missed.

Original query: {original_query}

Relevant papers found so far:
{papers_text}

Analyze the papers above and generate 3 NEW search queries that:
1. Use different terminology found in these papers
2. Target related subtopics not yet covered
3. Are specific enough to find precise results

Output ONLY a JSON array of 3 query strings:
["new query 1", "new query 2", "new query 3"]"""

    try:
        response = _get_client().chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        new_queries = json.loads(raw)
        logger.info(f"Evolved queries: {new_queries}")
        return new_queries
    except Exception as e:
        logger.error(f"Query evolution failed: {e}")
        return []
