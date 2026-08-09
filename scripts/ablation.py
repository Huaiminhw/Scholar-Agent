"""消融实验：6 组配置验证各模块贡献。"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from scholar_agent.utils.logger import logger

# 动态修改配置的 hook 函数
ORIGINAL_CONFIG: dict[str, Any] = {}


def apply_config(overrides: dict[str, bool]) -> None:
    """将 overrides 写入 settings 模块的全局变量。"""
    from scholar_agent import settings as s

    for key, val in overrides.items():
        setattr(s, key, val)


def restore_config() -> None:
    """恢复被覆盖的配置项。"""
    from scholar_agent import settings as s

    for key in ORIGINAL_CONFIG:
        setattr(s, key, ORIGINAL_CONFIG[key])


def run_evaluation(
    test_queries: list[dict],
    config_overrides: dict[str, bool],
    label: str = "",
    max_queries: int = 10,
) -> dict:
    """用指定配置运行评测，返回指标均值。

    为避免实际调用 LLM 产生费用，此处返回模拟指标。
    正式评测时，将 _simulate 改为 _evaluate_real。
    """
    from evaluate import calc_precision_at_k, calc_recall_at_k, calc_f1, calc_ndcg, calc_mrr
    from scholar_agent.agent import SearchAgent
    REAL_EVAL = True

    # 备份配置
    backup: dict[str, Any] = {}
    from scholar_agent import settings as s
    for key in config_overrides:
        backup[key] = getattr(s, key, None)

    # 应用配置
    for key, val in config_overrides.items():
        setattr(s, key, val)

    results: list[dict] = []
    start = time.perf_counter()

    try:
        agent = SearchAgent() if REAL_EVAL else None

        for i, q in enumerate(test_queries[:max_queries]):
            question = q.get("question", "")
            answers = q.get("answer", [])
            qid = q.get("qid", f"q_{i}")
            logger.info(f"  [{label}] [{i + 1}/{max_queries}] {qid}")

            if REAL_EVAL and agent:
                from evaluate import run_single_query
                result = run_single_query(agent, question, answers)
                if "error" in result:
                    continue
                pred = result["pred_titles"]
            else:
                # 模拟评测（确保脚本可运行）
                time.sleep(0.05)
                pred = [a for a in answers[:2]]  # 模拟命中前 2 个答案

            ans = answers
            results.append({
                "qid": qid,
                "precision@20": calc_precision_at_k(pred, ans, 20),
                "recall@20": calc_recall_at_k(pred, ans, 20),
                "f1": calc_f1(
                    calc_precision_at_k(pred, ans, 20),
                    calc_recall_at_k(pred, ans, 20),
                ),
                "ndcg@20": calc_ndcg(pred, ans, 20),
                "mrr": calc_mrr(pred, ans),
            })
    finally:
        # 恢复配置
        for key, val in backup.items():
            setattr(s, key, val)

    elapsed = time.perf_counter() - start
    n = len(results)
    if n == 0:
        return {}

    def avg(key: str) -> float:
        return sum(r[key] for r in results) / n

    summary = {
        "label": label,
        "queries": n,
        "time": round(elapsed, 2),
        "precision@20": round(avg("precision@20"), 4),
        "recall@20": round(avg("recall@20"), 4),
        "f1": round(avg("f1"), 4),
        "ndcg@20": round(avg("ndcg@20"), 4),
        "mrr": round(avg("mrr"), 4),
    }
    logger.info(f"  [{label}] F1={summary['f1']:.4f} R@20={summary['recall@20']:.4f} P@20={summary['precision@20']:.4f}")
    return summary


def main() -> None:
    """执行 6 组消融实验。"""
    from scholar_agent import settings as s

    # 保存原始配置
    ORIGINAL_CONFIG.update({
        "use_single_crawler": getattr(s, "use_single_crawler", False),
        "use_score_batch": getattr(s, "use_score_batch", False),
        "skip_full_text": getattr(s, "skip_full_text", False),
        "skip_memory": getattr(s, "skip_memory", False),
        "use_original_pipeline": getattr(s, "use_original_pipeline", False),
    })

    # 加载测试集
    test_path = Path("data/AutoScholarQuery/test.jsonl")
    if not test_path.exists():
        logger.error(f"Test file not found: {test_path}")
        return

    queries: list[dict] = []
    with test_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    max_q = min(10, len(queries))  # 每轮跑 10 条，避免耗时长
    logger.info(f"Ablation: {len(queries)} queries available, using {max_q} per experiment")

    experiments = [
        ("Full System (v5.0)", {}),
        ("-MultiCrawler", {"use_single_crawler": True}),
        ("-ClaimLevel", {"use_score_batch": True}),
        ("-FullText", {"skip_full_text": True}),
        ("-Memory", {"skip_memory": True}),
        ("Baseline (original draft)", {"use_original_pipeline": True}),
    ]

    results: list[dict] = []
    for name, config in experiments:
        # Clear memory to avoid cross-experiment pollution
        from scholar_agent.memory import SearchMemory
        import json
        mem_path = Path("data/search_memory.json")
        if mem_path.exists():
            mem_path.unlink()
        from scholar_agent.retrieval.retriever import _search_cache
        _search_cache.clear()
        from scholar_agent.retrieval.citation import _citation_cache
        _citation_cache.clear()
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Experiment: {name}")
        logger.info(f"{'=' * 50}")
        summary = run_evaluation(queries, config, label=name, max_queries=max_q)
        if summary:
            results.append(summary)

    # 输出对比表
    logger.info(f"\n{'=' * 60}")
    logger.info("ABLATION RESULTS")
    logger.info(f"{'=' * 60}")
    header = f"{'System':<30} {'F1':<8} {'R@20':<8} {'P@20':<8} {'NDCG':<8} {'MRR':<8}"
    logger.info(header)
    logger.info("-" * 60)
    for r in results:
        logger.info(
            f"{r['label']:<30} {r['f1']:<8.4f} {r['recall@20']:<8.4f} "
            f"{r['precision@20']:<8.4f} {r['ndcg@20']:<8.4f} {r['mrr']:<8.4f}"
        )

    # 保存结果
    output_path = Path("output/ablation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Ablation results saved to {output_path}")


if __name__ == "__main__":
    main()
