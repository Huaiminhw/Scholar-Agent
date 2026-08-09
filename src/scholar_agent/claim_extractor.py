from openai import OpenAI
import json
from scholar_agent.config import settings
from scholar_agent.utils.logger import logger


class ClaimExtractor:
    """
    ClaimExtractor: 查询断言提取器

    将用户查询拆解为 3-6 个独立可验证的断言（claims），
    每个断言有类型（method/task/time 等）和权重。
    """

    def __init__(self, track_callback=None) -> None:
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        self._track_cb = track_callback

    def extract(self, user_query: str) -> list[dict]:
        """分析查询，返回断言列表。"""
        prompt = f"""You are an academic query decomposer.
Break down the user query into 3-6 independently verifiable claims.
Each claim must be a single, specific assertion.

User query: {user_query}

Rules:
- required claims: MUST be satisfied for a paper to be relevant
- bonus claims: nice-to-have but not mandatory
- Each claim must be independently checkable from title+abstract

Output ONLY JSON array:
[
  {{"text": "uses Transformer or Vision Transformer architecture", "category": "method", "required": true, "weight": 1.0}},
  {{"text": "task is medical image segmentation", "category": "task", "required": true, "weight": 1.0}},
  {{"text": "published 2023 or later", "category": "time", "required": false, "weight": 0.5}}
]"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            claims = json.loads(raw)
            if self._track_cb:
                tokens = response.usage.total_tokens if response.usage else 0
                self._track_cb("claim_extractor", tokens)
            logger.info(f"ClaimExtractor: {len(claims)} claims extracted")
            return claims
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return [{"text": user_query, "category": "topic", "required": True, "weight": 1.0}]
