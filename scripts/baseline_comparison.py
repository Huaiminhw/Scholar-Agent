"""外部基线对比：与 PaSa 论文报告的指标比较。"""
from __future__ import annotations

import json
from pathlib import Path

from scholar_agent.utils.logger import logger

# PaSa 论文 Table 5 报告的基线指标
BASELINES = [
    {
        "system": "Google + GPT-4o",
        "recall@20": 0.385,
        "note": "PaSa Table 5",
    },
    {
        "system": "PaSa-7B",
        "recall@20": 0.524,
        "note": "PaSa published",
    },
]


def run_comparison() -> list[dict]:
    """运行我们的系统并与基线比较。

    返回包含所有系统指标的比较列表。
    """
    results = list(BASELINES)

    # 尝试验证我们的指标
    our_metrics_file = Path("output/ablation_results.json")
    if our_metrics_file.exists():
        try:
            ablation_data = json.loads(our_metrics_file.read_text(encoding="utf-8"))
            full_system = None
            for exp in ablation_data:
                if exp.get("label") == "Full System (v5.0)":
                    full_system = exp
                    break
            if full_system:
                results.append({
                    "system": "Scholar Agent v5.0 (ours)",
                    "recall@20": full_system.get("recall@20", "?"),
                    "precision@20": full_system.get("precision@20", "?"),
                    "f1": full_system.get("f1", "?"),
                    "ndcg@20": full_system.get("ndcg@20", "?"),
                    "mrr": full_system.get("mrr", "?"),
                    "note": "实测结果",
                })
            else:
                results.append({
                    "system": "Scholar Agent v5.0 (ours)",
                    "recall@20": "?",
                    "precision@20": "?",
                    "f1": "?",
                    "ndcg@20": "?",
                    "mrr": "?",
                    "note": "待评测",
                })
        except Exception as e:
            logger.warning(f"Failed to load ablation results: {e}")
            results.append({
                "system": "Scholar Agent v5.0 (ours)",
                "recall@20": "?",
                "note": "待评测",
            })
    else:
        logger.info("No ablation results found; run ablation.py first for real metrics.")
        results.append({
            "system": "Scholar Agent v5.0 (ours)",
            "recall@20": "?",
            "precision@20": "?",
            "f1": "?",
            "ndcg@20": "?",
            "mrr": "?",
            "note": "待评测",
        })

    return results


def print_table(results: list[dict]) -> None:
    """打印对比表。"""
    logger.info(f"\n{'=' * 70}")
    logger.info("BASELINE COMPARISON")
    logger.info(f"{'=' * 70}")
    header = f"{'System':<30} {'R@20':<10} {'P@20':<10} {'F1':<10} {'NDCG':<10} {'MRR':<10} {'Notes'}"
    logger.info(header)
    logger.info("-" * 70)
    for r in results:
        logger.info(
            f"{r['system']:<30} {str(r.get('recall@20', '?')):<10} "
            f"{str(r.get('precision@20', '-')):<10} {str(r.get('f1', '-')):<10} "
            f"{str(r.get('ndcg@20', '-')):<10} {str(r.get('mrr', '-')):<10} "
            f"{r.get('note', '')}"
        )
    logger.info(f"{'=' * 70}")
    logger.info("Note: Our metrics will be filled after running evaluate.py and ablation.py")


def main() -> None:
    results = run_comparison()
    print_table(results)

    # 保存 comparison 结果
    output_path = Path("output/baseline_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Baseline comparison saved to {output_path}")


if __name__ == "__main__":
    main()
