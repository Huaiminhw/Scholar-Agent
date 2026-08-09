import json
from datetime import datetime
from pathlib import Path
from scholar_agent.config import settings
from scholar_agent.utils.logger import logger


class SearchMemory:
    """搜索经验记忆：记录历史搜索的策略和效果，跨查询复用最佳策略。"""

    QUERY_TYPES = ["survey", "method_comparison", "specific_dataset", "frontier_tracking", "application"]

    def __init__(self) -> None:
        self.path: Path = settings.data_dir / "search_memory.json"
        self.entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"SearchMemory: failed to load {self.path}: {e}")
                return []
        return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def record(
        self,
        query_type: str,
        strategies: list[str],
        recall: float,
        precision: float,
    ) -> None:
        """记录一次搜索的经验。

        Args:
            query_type: 查询类型（来自 QUERY_TYPES）。
            strategies: 使用的搜索策略列表。
            recall: 估计召回率（0~1）。
            precision: 估计精确率（0~1）。
        """
        if query_type not in self.QUERY_TYPES:
            logger.warning(f"SearchMemory: unknown query type '{query_type}', recording anyway")
        self.entries.append({
            "query_type": query_type,
            "strategies": strategies,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "timestamp": datetime.now().isoformat(),
        })
        # 保留最近 100 条防止膨胀
        self.entries = self.entries[-100:]
        self._save()
        logger.debug(f"SearchMemory: recorded {query_type}, recall={recall}, precision={precision}")

    def recommend(self, user_query: str) -> list[str]:
        """根据查询文本推荐历史上效果最好的策略列表。

        如果无历史记录返回空列表。
        """
        query_type = self._classify(user_query)
        relevant = [e for e in self.entries if e.get("query_type") == query_type]
        if not relevant:
            logger.debug(f"SearchMemory: no memory for query type '{query_type}'")
            return []
        best = max(relevant, key=lambda x: x.get("recall", 0))
        strategies = best.get("strategies", [])
        logger.info(
            f"SearchMemory: found {len(relevant)} records for '{query_type}', "
            f"best recall={best.get('recall', 'N/A')}"
        )
        return strategies

    def _classify(self, query: str) -> str:
        """基于关键词对查询进行分类。"""
        q = query.lower()
        if any(w in q for w in ["survey", "review", "overview", "comprehensive", "literature"]):
            return "survey"
        if any(w in q for w in ["compare", "versus", "vs", "benchmark", "comparison"]):
            return "method_comparison"
        if any(w in q for w in ["dataset", "corpus", "collection", "data"]):
            return "specific_dataset"
        if any(w in q for w in ["latest", "recent", "state-of-the-art", "前沿", "最新"]):
            return "frontier_tracking"
        return "application"

    def clear(self) -> None:
        """清空所有记忆（测试用）。"""
        self.entries.clear()
        if self.path.exists():
            self.path.unlink()
