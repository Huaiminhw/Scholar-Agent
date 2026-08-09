"""评测回复结果的结构化程度"""
import json, sys
sys.path.insert(0, r"E:\华为杯论文检索项目\scholar_agent")
from scholar_agent.agent import SearchAgent
from scholar_agent.config import settings
from pathlib import Path

def score_structured_output(result: dict) -> tuple:
    """评价单次回复的结构化质量，返回 (总分, 明细)"""
    score = 0.0
    details = {}
    high = result.get("high_relevant", [])
    partial = result.get("partial_relevant", [])
    all_papers = high + partial

    # 1. 分层输出 (0-0.30)
    if high and partial:
        score += 0.30
        details["分层输出"] = 0.30
    elif high or partial:
        score += 0.15
        details["分层输出"] = 0.15
    else:
        details["分层输出"] = 0.0

    # If no papers in structured output, check if agent returned raw results
    if not all_papers:
        # Fall back to full paper list if available
        raw_papers = result.get("all_papers", [])
        if raw_papers:
            all_papers = raw_papers
            high = []
            partial = raw_papers
        else:
            return 0.0, {"错误": "无论文返回"}

    # 2. 评分覆盖率 (0-0.20)
    scored = sum(1 for p in all_papers if p.get("score") is not None) / len(all_papers)
    score += 0.20 * scored
    details["评分覆盖率"] = round(scored, 4)

    # 3. 理由覆盖率 (0-0.20)
    reasoned = sum(1 for p in all_papers if p.get("reason")) / len(all_papers)
    score += 0.20 * reasoned
    details["理由覆盖率"] = round(reasoned, 4)

    # 4. Claim 细粒度 (0-0.15)
    claim_cov = sum(1 for p in all_papers if p.get("claim_scores")) / len(all_papers)
    score += 0.15 * claim_cov
    details["Claim细粒度覆盖率"] = round(claim_cov, 4)

    # 5. 全文验证标记 (0-0.15)
    fv = sum(1 for p in all_papers if p.get("fulltext_verified_claims", 0) > 0) / len(all_papers)
    score += 0.15 * fv
    details["全文验证标记覆盖率"] = round(fv, 4)

    return round(score, 4), details


def evaluate_structure(test_file=None, num_samples=10):
    """在测试集上跑结构化评分"""
    if not test_file:
        test_file = str(settings.data_dir / "AutoScholarQuery" / "test.jsonl")
    
    with open(test_file, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]
    if num_samples > 0 and num_samples < len(queries):
        queries = queries[:num_samples]
    
    agent = SearchAgent()
    all_scores = []
    
    for i, q in enumerate(queries):
        question = q.get("question", "")
        qid = q.get("qid", f"q_{i}")
        print(f"[{i+1}/{len(queries)}] {qid}: {question[:60]}...")
        
        result = agent.run(question)
        s, details = score_structured_output(result)
        all_scores.append({"qid": qid, "score": s, "details": details})
        print(f"  结构化得分: {s:.4f}")
    
    avg = sum(s["score"] for s in all_scores) / len(all_scores) if all_scores else 0
    print(f"\n{'='*50}")
    print(f"平均结构化得分: {avg:.4f} / 1.0")
    print(f"="*50)
    
    out_path = settings.output_dir / "structure_scores.json"
    out_path.write_text(
        json.dumps({"avg_score": avg, "per_query": all_scores}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"结果保存到: {out_path}")
    return avg


if __name__ == "__main__":
    import sys
    n = 5
    for i, a in enumerate(sys.argv):
        if a == "--num_samples" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    evaluate_structure(num_samples=n)
