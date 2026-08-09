from openai import OpenAI
import json
from scholar_agent.config import settings
from scholar_agent.utils.logger import logger


class DimensionAnalyzer:
    """
    DimensionAnalyzer: 查询维度分析器

    解析用户查询，判断应该激活哪些搜索维度，
    每个维度生成专门的搜索策略。
    """

    def __init__(self, track_callback=None) -> None:
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        self._track_cb = track_callback

    def analyze(self, user_query: str) -> dict:
        """分析查询，返回激活的维度及其搜索策略。"""
        prompt = f"""You are an academic query dimension analyzer.
Given a research query, identify which search dimensions should be activated.

Query: {user_query}

Available dimensions:
- topic: core subject matter search
- method: methodology/technique/algorithms search
- dataset: dataset/benchmark/corpus search
- survey: survey/review/overview search
- application: application/use-case/domain search
- comparison: comparison/benchmarking/versus search
Choose ONLY the 2-3 MOST RELEVANT dimensions. Skip any that are not essential.

For each RELEVANT dimension, generate a specialized search query.
Skip dimensions that are not relevant to this query.

Output ONLY JSON:
{{
  "dimensions": [
    {{"name": "topic", "query": "...", "priority": "high", "reason": "..."}},
    {{"name": "method", "query": "...", "priority": "high", "reason": "..."}}
  ],
  "must_include_terms": ["term1", "term2"],
  "must_exclude_terms": ["term1"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            if self._track_cb:
                tokens = response.usage.total_tokens if response.usage else 0
                self._track_cb("dimension_analyzer", tokens)
            logger.info(f"DimensionAnalyzer: {len(result.get('dimensions', []))} dimensions")
            return result
        except Exception as e:
            logger.error(f"DimensionAnalyzer failed: {e}")
            return {
                "dimensions": [{"name": "topic", "query": user_query, "priority": "high", "reason": "fallback"}],
                "must_include_terms": [],
                "must_exclude_terms": [],
            }
