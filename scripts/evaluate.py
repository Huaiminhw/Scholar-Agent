"""评测指标体系：NDCG、MRR、Recall@K、Precision@K、F1 及 CostTracker。"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from scholar_agent.agent import SearchAgent
from scholar_agent.config import settings
from scholar_agent.utils.logger import logger


# ── 标题匹配 ──────────────────────────────────────────────────────────


def _keep_letters(s: str) -> str:
    return "".join(c for c in s if c.isalpha()).lower()


def is_match(pred_title: str, ans_title: str, threshold: float = 0.6) -> bool:
    """模糊标题匹配：词级别重叠率。"""
    pred = _keep_letters(pred_title)
    ans = _keep_letters(ans_title)
    if ans in pred or pred in ans:
        return True
    pred_words = set(pred_title.lower().split())
    ans_words = set(ans_title.lower().split())
    if len(ans_words) == 0:
        return False
    overlap = len(pred_words & ans_words) / len(ans_words)
    return overlap >= threshold


# ── 指标函数 ──────────────────────────────────────────────────────────


def calc_precision_at_k(
    pred_titles: list[str],
    answer_titles: list[str],
    k: int = 20,
) -> float:
    """Precision@K: 前 K 个结果中命中的比例。"""
    if k <= 0 or not pred_titles:
        return 0.0
    hits = 0
    for title in pred_titles[:k]:
        if any(is_match(title, ans) for ans in answer_titles):
            hits += 1
    return hits / min(k, len(pred_titles))


def calc_recall_at_k(
    pred_titles: list[str],
    answer_titles: list[str],
    k: int = 20,
) -> float:
    """Recall@K: 答案中被前 K 个结果覆盖的比例。"""
    if not answer_titles:
        return 0.0
    hits = 0
    for ans in answer_titles:
        if any(is_match(pred, ans) for pred in pred_titles[:k]):
            hits += 1
    return hits / len(answer_titles)


def calc_f1(precision: float, recall: float) -> float:
    """F1 = 2 * P * R / (P + R)。"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calc_ndcg(
    pred_titles: list[str],
    answer_titles: list[str],
    k: int = 20,
) -> float:
    """NDCG@K: 归一化折损累计增益。"""
    dcg = 0.0
    idcg = 0.0
    for i, title in enumerate(pred_titles[:k]):
        if any(is_match(title, ans) for ans in answer_titles):
            dcg += 1.0 / math.log2(i + 2)
    ideal_count = min(len(answer_titles), k)
    for i in range(ideal_count):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def calc_mrr(pred_titles: list[str], answer_titles: list[str]) -> float:
    """MRR: 第一个命中的倒数排名。"""
    for i, title in enumerate(pred_titles):
        if any(is_match(title, ans) for ans in answer_titles):
            return 1.0 / (i + 1)
    return 0.0


# ── 成本追踪 ──────────────────────────────────────────────────────────


class CostTracker:
    """追踪 LLM 调用次数和 Token 消耗。"""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.tokens_used = 0

    @property
    def avg_tokens_per_call(self) -> float:
        if self.llm_calls == 0:
            return 0.0
        return self.tokens_used / self.llm_calls


# ── 评测主函数 ────────────────────────────────────────────────────────


def load_test_queries(
    path: str | Path = "data/AutoScholarQuery/test.jsonl",
    num_samples: int = 0,
) -> list[dict]:
    """加载测试集。num_samples=0 表示全部加载。"""
    p = Path(path)
    if not p.exists():
        logger.warning(f"Test file not found: {p}")
        return []
    queries: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    if num_samples > 0 and num_samples < len(queries):
        queries = queries[:num_samples]
    logger.info(f"Loaded {len(queries)} test queries from {p}")
    return queries


def run_single_query(
    agent: SearchAgent,
    query: str,
    answer_titles: list[str],
) -> dict:
    """对一条查询运行 agent 并返回预测标题列表。"""
    start = time.perf_counter()
    try:
        result = agent.run(query)
    except Exception as e:
        logger.error(f"Query failed: {query[:60]}... error: {e}")
        return {"error": str(e), "time": time.perf_counter() - start}
    elapsed = time.perf_counter() - start
    pred_titles = []
    for p in result.get("high_relevant", []):
        pred_titles.append(p.get("title", ""))
    for p in result.get("partial_relevant", []):
        pred_titles.append(p.get("title", ""))
    return {
        "pred_titles": pred_titles,
        "answer_titles": answer_titles,
        "time": elapsed,
        "total_retrieved": result.get("total_retrieved", 0),
        "high_count": len(result.get("high_relevant", [])),
        "cost_tracker": result.get("cost_tracker", {}),
    }


def evaluate(
    test_file: str = "",
    num_samples: int = 5,
) -> dict:
    """完整评测流程。

    Args:
        test_file: 测试集路径（默认 data/AutoScholarQuery/test.jsonl）。
        num_samples: 评测样本数。

    Returns:
        包含各项指标均值的字典。
    """
    if not test_file:
        test_file = str(settings.data_dir / "AutoScholarQuery" / "test.jsonl")

    queries = load_test_queries(test_file, num_samples)
    if not queries:
        logger.error("No test queries loaded, aborting evaluation.")
        return {}

    agent = SearchAgent()
    all_metrics: list[dict] = []

    for i, q in enumerate(queries):
        question = q.get("question", "")
        answers = q.get("answer", [])
        qid = q.get("qid", f"q_{i}")

        logger.info(f"\n[{i + 1}/{len(queries)}] {qid}: {question[:80]}...")
        result = run_single_query(agent, question, answers)

        if "error" in result:
            all_metrics.append({
                "qid": qid, "precision@20": 0.0, "recall@20": 0.0,
                "f1": 0.0, "ndcg@20": 0.0, "mrr": 0.0,
                "time": result.get("time", 0), "retrieved": 0, "high_count": 0,
            })
            continue

        pred = result["pred_titles"]
        ans = result["answer_titles"]
        p20 = calc_precision_at_k(pred, ans, 20)
        r20 = calc_recall_at_k(pred, ans, 20)
        metrics = {
            "qid": qid,
            "precision@20": p20,
            "recall@20": r20,
            "f1": calc_f1(p20, r20),
            "ndcg@20": calc_ndcg(pred, ans, 20),
            "mrr": calc_mrr(pred, ans),
            "time": result["time"],
            "retrieved": result["total_retrieved"],
            "high_count": result["high_count"],
            "llm_calls": result.get("cost_tracker", {}).get("total_llm_calls", 0),
            "tokens": result.get("cost_tracker", {}).get("total_tokens", 0),
        }
        all_metrics.append(metrics)

        # 每轮输出
        logger.info(f"  P@20={p20:.4f} R@20={r20:.4f} F1={metrics['f1']:.4f}")

        time.sleep(3)  # API 限流

    # 汇总
    summary = _summarize_metrics(all_metrics)
    _log_summary(summary)

    # 保存
    output_path = settings.output_dir / "eval_results.json"
    output_path.write_text(
        json.dumps({"summary": summary, "per_query": all_metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Report saved to {output_path}")
    return summary


def _summarize_metrics(metrics: list[dict]) -> dict:
    n = len(metrics)
    if n == 0:
        return {}
    return {
        "num_queries": n,
        "precision@20": round(sum(m["precision@20"] for m in metrics) / n, 4),
        "recall@20": round(sum(m["recall@20"] for m in metrics) / n, 4),
        "f1": round(sum(m["f1"] for m in metrics) / n, 4),
        "ndcg@20": round(sum(m["ndcg@20"] for m in metrics) / n, 4),
        "mrr": round(sum(m["mrr"] for m in metrics) / n, 4),
        "avg_time": round(sum(m["time"] for m in metrics) / n, 2),
        "avg_retrieved": round(sum(m["retrieved"] for m in metrics) / n, 1),
        "avg_high_count": round(sum(m["high_count"] for m in metrics) / n, 1),
        "avg_llm_calls": round(sum(m.get("llm_calls", 0) for m in metrics) / n, 1),
        "avg_tokens": round(sum(m.get("tokens", 0) for m in metrics) / n, 1),
    }


def _log_summary(summary: dict) -> None:
    if not summary:
        return
    logger.info(f"\n{'=' * 50}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Queries:      {summary['num_queries']}")
    logger.info(f"  Precision@20: {summary['precision@20']:.4f}")
    logger.info(f"  Recall@20:    {summary['recall@20']:.4f}")
    logger.info(f"  F1:           {summary['f1']:.4f}")
    logger.info(f"  NDCG@20:      {summary['ndcg@20']:.4f}")
    logger.info(f"  MRR:          {summary['mrr']:.4f}")
    logger.info(f"  Avg time:     {summary['avg_time']:.2f}s")
    logger.info(f"  Avg LLM calls: {summary.get('avg_llm_calls', '?'):.1f}")
    logger.info(f"  Avg tokens:    {summary.get('avg_tokens', '?'):.1f}")


if __name__ == "__main__":
    import sys
    n = 5
    for i, a in enumerate(sys.argv):
        if a == "--num_samples" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    evaluate(num_samples=n)
