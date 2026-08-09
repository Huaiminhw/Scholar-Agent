from difflib import SequenceMatcher
from scholar_agent.utils.logger import logger


class ResultMerger:
    """
    ResultMerger: 多 Crawler 结果合并器

    合并多维度搜索结果，基于 ID 去重 + 标题模糊匹配。
    """

    def merge(self, dimension_results: dict[str, list[dict]]) -> list[dict]:
        """合并各维度结果，去重后返回唯一论文列表。"""
        merged: dict[str, dict] = {}
        for dim_name, papers in dimension_results.items():
            for paper in papers:
                pid = paper.get('id', '')
                if pid and pid in merged:
                    merged[pid].setdefault('source_dimensions', []).append(dim_name)
                    continue
                # 标题去重
                if self._title_exists(paper.get('title', ''), merged):
                    continue
                paper['source_dimensions'] = [dim_name]
                key = pid or f"{dim_name}_{paper.get('title', '')[:20]}"
                merged[key] = paper
        total_raw = sum(len(v) for v in dimension_results.values())
        logger.info(f"Merger: {total_raw} raw -> {len(merged)} unique papers")
        return list(merged.values())

    def _title_exists(self, title: str, merged: dict[str, dict]) -> bool:
        """模糊标题匹配，相似度 > 0.85 视为重复。"""
        for p in merged.values():
            if SequenceMatcher(None, title.lower(), p.get('title', '').lower()).ratio() > 0.85:
                return True
        return False
