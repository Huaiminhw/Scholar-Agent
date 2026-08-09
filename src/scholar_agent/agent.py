import time
from datetime import datetime
from scholar_agent.config import settings
from scholar_agent.crawler import Crawler
from scholar_agent.selector import Selector
from scholar_agent.claim_extractor import ClaimExtractor
from scholar_agent.dimension_analyzer import DimensionAnalyzer
from scholar_agent.result_merger import ResultMerger
from scholar_agent.memory import SearchMemory
from scholar_agent.retrieval.retriever import search_papers
from scholar_agent.retrieval.citation import get_references
from scholar_agent.presentation.formatter import save_markdown, save_json
from scholar_agent.utils.logger import logger

class CostTracker:
    def __init__(self):
        self.llm_calls = 0
        self.tokens_used = 0
        self.per_module = {}

    def track(self, module, tokens=0):
        self.llm_calls += 1
        self.tokens_used += tokens
        self.per_module.setdefault(module, {"calls": 0, "tokens": 0})
        self.per_module[module]["calls"] += 1
        self.per_module[module]["tokens"] += tokens

    def get_summary(self):
        return {
            "total_llm_calls": self.llm_calls,
            "total_tokens": self.tokens_used,
            "per_module": self.per_module,
        }




class SearchAgent:
    """
    SearchAgent：搜索主控

    将 Crawler（搜索策略）+ Selector（相关性评估）+ DimensionAnalyzer
    + ClaimExtractor 组合成完整搜索流程。
    """

    def __init__(self) -> None:
        self.cost_tracker = CostTracker()
        self.dimension_analyzer = DimensionAnalyzer(track_callback=self.cost_tracker.track)
        self.crawler = Crawler(track_callback=self.cost_tracker.track)
        self.selector = Selector(track_callback=self.cost_tracker.track)
        self.merger = ResultMerger()
        self.memory = SearchMemory()
        self.claim_extractor = ClaimExtractor(track_callback=self.cost_tracker.track)
        logger.info("SearchAgent initialized")

    def run(self, user_query: str) -> dict:
        """执行完整搜索流程。"""
        logger.info(f"Starting search for: {user_query}")
        all_papers: dict[str, dict] = {}

        # Warm start: 查询记忆推荐策略
        remembered = self.memory.recommend(user_query)
        if remembered:
            logger.info(f"  Memory suggests strategies: {remembered}")

        # Step 1: DimensionAnalyzer 分析查询维度
        logger.info("[Step 1] Analyzing query dimensions...")
        analysis = self.dimension_analyzer.analyze(user_query)

        # Step 2: Multi-Crawler 并行搜索
        logger.info("[Step 2] Multi-Crawler search...")
        dim_results: dict[str, list[dict]] = {}
        for dim in analysis.get('dimensions', []):
            query = dim.get('query', user_query)
            logger.info(f"  [{dim['name']}] priority={dim.get('priority', 'medium')} '{query}'")
            papers = search_papers(query, limit=settings.search_limit)
            dim_results[dim['name']] = papers
            time.sleep(0.3)

        # Merge and deduplicate
        merged_papers = self.merger.merge(dim_results)
        for p in merged_papers:
            all_papers[p['id']] = p
        logger.info(f"  After merge: {len(all_papers)} unique papers")

        # Step 3: 关键词组合补充检索
        logger.info("[Step 3] Keyword combination search...")
        kw = analysis.get('must_include_terms', [])
        if len(kw) >= 3:
            kw_queries = [
                ' '.join(kw[:3]),
                ' '.join(kw[1:4]) if len(kw) >= 4 else ' '.join(kw[:3]),
            ]
            for q in kw_queries:
                logger.info(f"  Keyword search: {q}")
                for p in search_papers(q, limit=settings.search_limit):
                    if p['id'] and p['id'] not in all_papers:
                        all_papers[p['id']] = p
                time.sleep(0.5)
        else:
            plan = self.crawler.plan_search(user_query)
            kw = plan.get('keywords', [])
            if len(kw) >= 3:
                for q in [' '.join(kw[:3]), ' '.join(kw[1:4]) if len(kw) >= 4 else ' '.join(kw[:3])]:
                    logger.info(f"  Keyword search: {q}")
                    for p in search_papers(q, limit=settings.search_limit):
                        if p['id'] and p['id'] not in all_papers:
                            all_papers[p['id']] = p
                    time.sleep(0.5)

        logger.info(f"  Total papers after steps 1-3: {len(all_papers)}")

        # Step 4: 查询演化
        logger.info("[Step 4] Query evolution...")
        papers_list = list(all_papers.values())
        papers_list.sort(key=lambda x: x.get('citations', 0), reverse=True)
        evolved = self.crawler.evolve_query(user_query, papers_list[:10])
        if evolved:
            for q in evolved:
                logger.info(f"  Evolved search: {q}")
                for p in search_papers(q, limit=settings.search_limit):
                    if p['id'] and p['id'] not in all_papers:
                        all_papers[p['id']] = p
                time.sleep(0.5)

        logger.info(f"  Total papers after evolution: {len(all_papers)}")

        # Step 5: 引文追踪
        logger.info("[Step 5] Citation tracking...")
        papers_list = list(all_papers.values())
        top3 = sorted(papers_list, key=lambda x: x.get('citations', 0), reverse=True)[:3]
        for p in top3:
            logger.info(f"  Tracking refs: {p['title'][:60]}...")
            refs = get_references(p['id'], limit=5)
            for r in refs:
                if r['id'] and r['id'] not in all_papers:
                    all_papers[r['id']] = r
            time.sleep(0.5)

        logger.info(f"  Total papers after citation tracking: {len(all_papers)}")

        # Step 6: Claim-level 打分 + 缺口分析
        logger.info("[Step 6] Claim-level scoring + gap analysis...")
        papers_list = list(all_papers.values())
        claims = self.claim_extractor.extract(user_query)
        scored = self.selector.score_claims(user_query, papers_list, claims)

        # Step 6.5: 缺口驱动的补充检索
        gap_claims = self._identify_gaps(scored, claims)
        if gap_claims:
            logger.info(f"  Found {len(gap_claims)} gap claims, doing targeted search...")
            gap_query = ' '.join([c.get('text', '') for c in gap_claims[:2]])
            logger.info(f"  Gap search: {gap_query}")
            for p in search_papers(gap_query, limit=10):
                if p['id'] and p['id'] not in all_papers:
                    all_papers[p['id']] = p
            time.sleep(0.5)
            # Re-score with gap papers added
            papers_list = list(all_papers.values())
            scored = self.selector.score_claims(user_query, papers_list, claims)

        # Step 6.75: 全文深度验证
        logger.info("[Step 6.75] Deep full-text verification...")
        scored = self._deep_verify(scored, claims, all_papers)

        # Step 7: Split into high/partial relevance
        # Lower threshold since score_batch scores tend to be more conservative
        high_relevant = [p for p in scored if p.get('score', 0) >= 6]
        partial_relevant = [p for p in scored if 3 <= p.get('score', 0) < 6]
        logger.info(f"  High relevance: {len(high_relevant)}, Partial: {len(partial_relevant)}")

        # Step 8: 构建结果
        result = self._build_result(user_query, analysis, len(all_papers), high_relevant, partial_relevant)
        self._save_result(result)

        # Step 9: 记录搜索经验到记忆
        query_type = self.memory._classify(user_query)
        strategies_used = [d.get("name", "") for d in analysis.get("dimensions", [])]
        if not strategies_used:
            strategies_used = ["keyword+search"]
        total = len(all_papers) or 1
        high_count = len(high_relevant)
        estimated_recall = min(1.0, high_count / max(len(claims) * 2, 1))
        estimated_precision = high_count / total
        self.memory.record(query_type, strategies_used, estimated_recall, estimated_precision)

        self._print_result(result)
        # Include full paper list for structure_score evaluation
        result["all_papers"] = [
            {k: v for k, v in p.items() if k in ("id", "title", "score", "reason", "claim_scores", "fulltext_verified_claims")}
            for p in scored
        ]
        return result

    def _identify_gaps(self, scored_papers: list[dict], claims: list[dict]) -> list[dict]:
        """识别多数论文都不满足的 claim（即为搜索缺口）。"""
        top_papers = scored_papers[:5]
        gap_claims = []
        for c in claims:
            satisfied_count = sum(
                1 for p in top_papers
                if any(
                    s.get('satisfied', False)
                    for s in p.get('claim_scores', [])
                    if s.get('claim_text') == c.get('text')
                )
            )
            if satisfied_count < 2 and c.get('required', True):
                gap_claims.append(c)
        logger.debug(f"Gap analysis: {len(gap_claims)} gaps identified")
        return gap_claims

    def _deep_verify(
        self,
        scored_papers: list[dict],
        claims: list[dict],
        all_papers: dict[str, dict],
    ) -> list[dict]:
        """利用 FullTextAnalyzer 对 top 论文进行全文深度验证。"""
        from scholar_agent.full_text import FullTextAnalyzer
        from scholar_agent.retrieval.retriever import search_papers

        analyzer = FullTextAnalyzer()
        top_papers = scored_papers[:5]

        for paper in top_papers:
            paper_id = paper.get('id', '')
            logger.info(f"  Deep verify: {paper.get('title', '')[:60]}...")

            pdf_path = analyzer.fetch_full_text(paper)
            if not pdf_path:
                logger.debug(f"    No PDF available for {paper_id}")
                continue

            sections = analyzer.parse_sections(pdf_path)
            if not sections or sections.get('full_text', '') == '':
                continue

            # 提取参考文献并补充检索
            refs = analyzer.extract_references(sections)
            for ref_title in refs[:10]:
                ref_papers = search_papers(ref_title, limit=3)
                for rp in ref_papers:
                    if rp['id'] and rp['id'] not in all_papers:
                        all_papers[rp['id']] = rp
                time.sleep(0.3)

            # 逐一验证 claim 在正文中的证据
            for c in claims:
                claim_text = c.get('text', '')
                if not claim_text:
                    continue
                result = analyzer.verify_claim_in_fulltext(paper, claim_text)
                if result['found']:
                    for ps in scored_papers:
                        if ps.get('id') == paper_id:
                            for cs in ps.get('claim_scores', []):
                                if cs.get('claim_text') == claim_text:
                                    cs['fulltext_verified'] = True
                                    cs['snippet'] = result['snippet']
                            current = ps.get('score', 0)
                            ps['score'] = min(10, current + 0.5)
                            ps['fulltext_verified_claims'] = ps.get('fulltext_verified_claims', 0) + 1

        logger.info(f"  Deep verify complete. Total papers: {len(all_papers)}")
        return scored_papers

    def _build_result(
        self,
        query: str,
        plan: dict,
        total_retrieved: int,
        high_relevant: list[dict],
        partial_relevant: list[dict],
    ) -> dict:
        """构建标准返回结构。"""
        return {
            "query": query,
            "plan": plan,
            "total_retrieved": total_retrieved,
            "high_relevant": high_relevant[:settings.top_k],
            "partial_relevant": partial_relevant[:settings.top_k],
            "timestamp": datetime.now().isoformat(),
            "cost_tracker": self.cost_tracker.get_summary(),
        }

    def _save_result(self, result: dict) -> None:
        """持久化结果（JSON + Markdown）。"""
        save_json(result)
        save_markdown(result)

    def _print_result(self, result: dict) -> None:
        """用 logger 打印结果摘要。"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Search Results for: {result['query']}")
        logger.info(f"{'='*50}")
        logger.info(f"Total retrieved: {result['total_retrieved']}, High relevance: {len(result['high_relevant'])}, Partial: {len(result['partial_relevant'])}")
        logger.info("--- High Relevance ---")
        for i, p in enumerate(result['high_relevant'][:5], 1):
            logger.info(f"  {i}. [{p.get('score', '?')}] {p['title']}")
        if result['partial_relevant']:
            logger.info("--- Partial Relevance (top 5) ---")
            for i, p in enumerate(result['partial_relevant'][:5], 1):
                logger.info(f"  {i}. [{p.get('score', '?')}] {p['title']}")
