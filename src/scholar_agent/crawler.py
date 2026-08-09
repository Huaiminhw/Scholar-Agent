from openai import OpenAI
import json
from scholar_agent.config import settings
from scholar_agent.utils.logger import logger


class Crawler:
    """
    Crawler：搜索策略 Agent（大脑）

    职责：
    1. plan_search: 分析用户查询，输出多维度搜索计划
    2. evolve_query: 基于已找到论文生成新搜索词
    """

    def __init__(self, track_callback=None) -> None:
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        self._track_cb = track_callback

    def plan_search(self, user_query: str) -> dict:
        """
        分析查询，输出多维度搜索计划（JSON）。
        失败时降级为只搜原始查询。
        """
        prompt = """You are an academic paper search strategist (Crawler).
Analyze the user query and design a multi-dimensional search plan.

User query: """ + user_query + """

Requirements:
- Identify the core topic
- Break down the search into at least 4 dimensions: topic, method, dataset, survey
- For each dimension, provide a specific search query and the rationale
- List 5-8 specific keywords/phrases
- Specify any constraints (year range, venue, etc.)

Output ONLY valid JSON (no markdown, no extra text):
{
  "core_topic": "short core topic",
  "search_dimensions": [
    {"dimension": "topic", "query": "search query for this dimension", "reason": "why this dimension matters"},
    {"dimension": "method", "query": "search query for methods", "reason": "..."},
    {"dimension": "dataset", "query": "search query for datasets", "reason": "..."},
    {"dimension": "survey", "query": "survey or review search", "reason": "..."}
  ],
  "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5", "kw6"],
  "constraints": {}
}"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "system", "content": "You are an academic paper search strategist. Output ONLY valid JSON."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            plan = json.loads(raw)
            if self._track_cb:
                tokens = response.usage.total_tokens if response.usage else 0
                self._track_cb("crawler_plan", tokens)
            logger.info(f"Search plan: {plan.get('core_topic')}, {len(plan.get('search_dimensions', []))} dimensions")
            return plan
        except Exception as e:
            logger.warning(f"plan_search failed, fallback to raw query: {e}")
            return {
                "core_topic": user_query,
                "search_dimensions": [
                    {"dimension": "topic", "query": user_query, "reason": "fallback"}
                ],
                "keywords": [w for w in user_query.split() if len(w) > 2][:8],
                "constraints": {},
            }

    def evolve_query(self, original_query: str, found_papers: list[dict]) -> list[str]:
        """
        基于已找到的论文生成3个新搜索词。
        从初稿 query_evolver.py 迁移核心逻辑。
        """
        if not found_papers:
            return []

        top_papers = sorted(found_papers, key=lambda x: x.get('citations', 0), reverse=True)[:5]
        papers_text = ""
        for p in top_papers:
            papers_text += f"- Title: {p['title']}\n  Abstract: {(p['abstract'] or '')[:200]}\n\n"

        prompt = f"""Based on the original query and relevant papers found, generate 3 NEW search queries.

Original query: {original_query}

Papers:
{papers_text}

Analyze the papers above. Find terminology, techniques, or subtopics not covered by the original query.
Generate 3 specific search queries to find more relevant papers.

Output ONLY a JSON array:
["new query 1", "new query 2", "new query 3"]"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "system", "content": "You are an academic paper search expert. Output ONLY valid JSON."},
                          {"role": "user", "content": prompt}],
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            new_queries = json.loads(raw)
            if self._track_cb:
                tokens = response.usage.total_tokens if response.usage else 0
                self._track_cb("crawler_evolve", tokens)
            logger.info(f"Evolved {len(new_queries)} new queries: {new_queries}")
            return new_queries
        except Exception as e:
            logger.warning(f"evolve_query failed: {e}")
            return []
