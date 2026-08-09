from openai import OpenAI
import json
from scholar_agent.config import settings
from scholar_agent.utils.logger import logger


class Selector:
    """
    Selector：相关性评估 Agent（眼睛）

    职责：
    - score_batch(): 传统批量总分（0-10），向后兼容
    - score_claims(): Claim-level 细粒度打分 + 覆盖度矩阵
    """

    def __init__(self, track_callback=None) -> None:
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        self._track_cb = track_callback

    def score_batch(self, user_query: str, papers: list[dict]) -> list[dict]:
        """批量总分入口，每批10篇，按 score 降序返回。"""
        if not papers:
            return []

        scored: list[dict] = []
        for i in range(0, len(papers), 10):
            batch = papers[i:i + 10]
            scored.extend(self._score_batch(user_query, batch))

        scored.sort(key=lambda x: x.get('score', 0), reverse=True)
        return scored

    def _score_batch(self, user_query: str, papers: list[dict]) -> list[dict]:
        """对一批论文调用 LLM 打总分。"""
        papers_text = ""
        for idx, p in enumerate(papers):
            abstract_short = (p.get('abstract') or '')[:300]
            papers_text += (
                f"[{idx}] Title: {p['title']}\n"
                f"     Year: {p.get('year', '?')}  Citations: {p.get('citations', 0)}\n"
                f"     Abstract: {abstract_short}\n\n"
            )

        prompt = f"""You are an academic paper relevance evaluator (Selector).

User query: {user_query}

Papers:
{papers_text}

Score each paper 0-10 based on relevance to the query.
Be strict — do NOT give high scores just for keyword overlap.

Scoring guide:
- 9-10: perfect match, directly addresses the query
- 7-8: highly relevant
- 4-6: partially relevant
- 1-3: low relevance
- 0: completely irrelevant

Output ONLY a JSON array:
[
  {{"idx": 0, "score": 9, "reason": "directly studies this topic"}},
  {{"idx": 1, "score": 2, "reason": "unrelated domain"}}
]"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You are an academic paper relevance evaluator. Score strictly. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            scores = json.loads(raw)
            if self._track_cb:
                tokens = response.usage.total_tokens if response.usage else 0
                self._track_cb("selector_score_batch", tokens)

            for s in scores:
                idx = s.get('idx', -1)
                if isinstance(idx, int) and 0 <= idx < len(papers):
                    papers[idx]['score'] = min(10, max(0, int(s.get('score', 5))))
                    papers[idx]['reason'] = s.get('reason', '')

            logger.debug(f"Scored batch of {len(papers)} papers")
            return papers
        except Exception as e:
            logger.warning(f"Batch scoring failed: {e}")
            for p in papers:
                p.setdefault('score', 5)
                p['reason'] = 'scoring error'
            return papers

    def score_claims(self, user_query: str, papers: list[dict], claims: list[dict]) -> list[dict]:
        """
        Claim-level 细粒度打分。

        对每篇论文逐 claim 判断，生成覆盖度矩阵。
        加权总分 = required_score * 0.7 + bonus_score * 0.3
        """
        if not papers or not claims:
            return papers

        for paper in papers:
            paper_scores = []
            for claim in claims:
                satisfied, confidence = self._check_claim(paper, claim)
                paper_scores.append({
                    "claim_text": claim.get('text', ''),
                    "satisfied": satisfied,
                    "confidence": confidence,
                    "required": claim.get('required', True),
                })
            paper['claim_scores'] = paper_scores

            # 加权总分
            required_claims = [s for s in paper_scores if s['required']]
            bonus_claims = [s for s in paper_scores if not s['required']]

            required_score = sum(s['confidence'] for s in required_claims if s['satisfied']) / max(1, len(required_claims))
            bonus_score = sum(s['confidence'] * 0.5 for s in bonus_claims if s['satisfied']) / max(1, len(bonus_claims))
            paper['score'] = round((required_score * 0.7 + bonus_score * 0.3) * 10, 1)

        papers.sort(key=lambda x: x.get('score', 0), reverse=True)
        logger.info(f"Claim scoring completed for {len(papers)} papers")
        return papers

    def _check_claim(self, paper: dict, claim: dict) -> tuple[bool, float]:
        """检查单篇论文是否满足某个 claim。"""
        prompt = f"""Check if this paper satisfies the claim.

Claim: {claim.get('text', '')}
Title: {paper.get('title', '')}
Abstract: {(paper.get('abstract') or '')[:300]}

Output JSON only:
{{"satisfied": true/false, "confidence": 0.0-1.0, "evidence": "quote from text"}}"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            if self._track_cb:
                tokens = response.usage.total_tokens if response.usage else 0
                self._track_cb("selector_check_claim", tokens)
            return bool(result.get('satisfied', False)), float(result.get('confidence', 0.0))
        except Exception:
            return False, 0.0
