import json
from datetime import datetime
from pathlib import Path
from scholar_agent.config import settings
from scholar_agent.utils.logger import logger


def format_result(result: dict) -> str:
    """生成 Markdown 格式搜索结果报告。"""
    query = result['query']
    parsed = result.get('plan', result.get('parsed', {}))
    high: list[dict] = result.get('high_relevant', [])
    partial: list[dict] = result.get('partial_relevant', [])

    lines: list[str] = []
    lines.append("# 学术论文搜索报告")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**用户查询**: {query}")
    lines.append("")

    # 查询理解摘要
    lines.append("## 查询理解")
    if isinstance(parsed, dict):
        if parsed.get('core_topic'):
            lines.append(f"- **核心主题**: {parsed['core_topic']}")
        if parsed.get('keywords'):
            lines.append(f"- **关键词**: {', '.join(parsed['keywords'])}")
        if parsed.get('methods'):
            lines.append(f"- **方法论**: {', '.join(parsed['methods'])}")
        if parsed.get('year_from'):
            lines.append(f"- **时间范围**: {parsed['year_from']} 至今")
    lines.append("")

    # 检索统计
    lines.append("## 检索统计")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 总检索论文数 | {result.get('total_retrieved', 0)} |")
    lines.append(f"| 高度相关 | {len(high)} |")
    lines.append(f"| 部分相关 | {len(partial)} |")
    lines.append("")

    # 高度相关论文
    lines.append("## 高度相关论文")
    for i, p in enumerate(high, 1):
        lines.append(f"### {i}. {p['title']}")
        lines.append(f"- **相关性评分**: {p.get('score', '?')}/10")
        lines.append(f"- **年份**: {p.get('year', '?')}")
        lines.append(f"- **引用数**: {p.get('citations', '?')}")
        lines.append(f"- **发表 venue**: {p.get('venue', '未知')}")
        authors = p.get('authors', [])
        if authors:
            lines.append(f"- **作者**: {', '.join(authors[:3])}")
        if p.get('reason'):
            lines.append(f"- **相关原因**: {p['reason']}")
        if p.get('abstract'):
            lines.append(f"- **摘要**: {p['abstract'][:200]}...")
        lines.append("")

    # 部分相关论文（表格形式）
    if partial:
        lines.append("## 部分相关论文")
        lines.append("| # | 标题 | 年份 | 引用数 | 评分 |")
        lines.append("|---|------|------|--------|------|")
        for i, p in enumerate(partial, 1):
            title_short = (p['title'][:50] + '...') if len(p['title']) > 50 else p['title']
            lines.append(f"| {i} | {title_short} | {p.get('year', '?')} | {p.get('citations', '?')} | {p.get('score', '?')} |")
        lines.append("")

    # 年份分布
    all_papers = high + partial
    if all_papers:
        lines.append("## 年份分布")
        year_count: dict[int | str, int] = {}
        for p in all_papers:
            y = p.get('year')
            if y:
                year_count[y] = year_count.get(y, 0) + 1
        lines.append("| 年份 | 论文数 |")
        lines.append("|------|--------|")
        for y in sorted(year_count.keys(), reverse=True):
            lines.append(f"| {y} | {year_count[y]} |")

    return '\n'.join(lines)


def save_markdown(result: dict, path: str | Path | None = None) -> str:
    """保存 Markdown 报告到文件。"""
    if path is None:
        path = settings.output_dir / 'report.md'
    md = format_result(result)
    Path(path).write_text(md, encoding='utf-8')
    logger.info(f"Markdown report saved: {path}")
    return md


def save_json(result: dict, path: str | Path | None = None) -> None:
    """保存 JSON 结果到文件。"""
    if path is None:
        path = settings.output_dir / 'result.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON result saved: {path}")
