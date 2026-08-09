# Scholar Agent — API 文档

## 概述

`SearchAgent` 是系统的核心入口类。封装了从查询理解到结果输出的全流程。

---

## 快速开始

```python
from scholar_agent import SearchAgent

agent = SearchAgent()
result = agent.run("Transformer-based medical image segmentation methods")
```

---

## `SearchAgent`

### 构造函数

```python
agent = SearchAgent()
```

内部自动初始化：
| 组件 | 类 | 作用 |
|------|---|------|
| `dimension_analyzer` | `DimensionAnalyzer` | 查询维度分解 |
| `crawler` | `Crawler` | 搜索策略生成与演化 |
| `selector` | `Selector` | 相关性评分 |
| `merger` | `ResultMerger` | 去重与融合 |
| `claim_extractor` | `ClaimExtractor` | Claim 级细粒度分解 |
| `memory` | `SearchMemory` | 搜索经验记忆 |

### `run(user_query: str) -> dict`

执行完整搜索流水线。

**参数：**
- `user_query` (str): 自然语言学术查询

**返回结构：**

```python
{
    "query": str,                # 原始查询
    "plan": dict,                # 维度分析结果
    "total_retrieved": int,      # 累计检索论文数
    "high_relevant": [           # 高相关论文列表 (score >= 7)
        {
            "id": str,           # 论文唯一 ID
            "title": str,        # 论文标题
            "citations": int,    # 引用数
            "score": float,      # 0-10 相关性得分
            "claim_scores": [    # 各 Claim 细粒度得分
                {
                    "claim_text": str,
                    "satisfied": bool,
                    "confidence": float,
                    "fulltext_verified": bool,  # 仅 W4+
                    "snippet": str              # 仅 W4+
                }
            ],
            "source_dimensions": [str],  # 来源搜索维度
        }
    ],
    "partial_relevant": [...],   # 部分相关论文列表 (4 <= score < 7)
    "timestamp": str,            # ISO 格式时间戳
}
```

---

## 组件 API

### `DimensionAnalyzer.analyze(query: str) -> dict`

将查询分解为多个搜索维度。

```python
{
    "dimensions": [
        {"name": "topic", "query": "...", "priority": "high", "reason": "..."},
        {"name": "method", "query": "...", "priority": "medium", "reason": "..."},
    ],
    "must_include_terms": ["term1", "term2"],
}
```

### `Crawler.plan_search(query: str) -> dict`

生成多策略搜索计划。当 LLM 维度分析无法提取有效关键词时作为后备。

### `Selector.score_batch(papers: list[dict], query: str) -> list[dict]`

批量打分模式（非 Claim 模式）。为每篇论文返回 0-10 分。

### `Selector.score_claims(query: str, papers: list[dict], claims: list[dict]) -> list[dict]`

Claim 级细粒度打分模式。每篇论文对每个 Claim 返回 satisfied / confidence。

### `ClaimExtractor.extract(query: str) -> list[dict]`

将查询分解为 3-6 个独立可验证的 Claim。

```python
[
    {"text": "...", "category": "method", "required": True, "weight": 0.3},
    {"text": "...", "category": "dataset", "required": False, "weight": 0.2},
]
```

### `FullTextAnalyzer`

| 方法 | 说明 |
|------|------|
| `fetch_full_text(paper) -> Path\|None` | 下载论文 PDF |
| `parse_sections(pdf_path) -> dict` | 解析章节结构 |
| `extract_references(sections) -> list[str]` | 提取参考文献标题 |
| `verify_claim_in_fulltext(paper, claim) -> dict` | 验证 Claim 证据 |

### `SearchMemory`

| 方法 | 说明 |
|------|------|
| `record(query_type, strategies, recall, precision)` | 记录搜索经验 |
| `recommend(user_query) -> list[str]` | 推荐历史最优策略 |
| `_classify(query) -> str` | 查询类型分类 |
| `clear()` | 清空记忆 |

---

## 配置项

| 环境变量 | 类型 | 默认值 | 说明 |
|----------|------|--------|------|
| `LLM_API_KEY` | str | `""` | LLM API 密钥 |
| `LLM_BASE_URL` | str | `https://api.deepseek.com/v1` | API 端点 |
| `LLM_MODEL` | str | `deepseek-chat` | 模型名称 |
| `SS_API_KEY` | str | `None` | Semantic Scholar API 密钥 |
| `USE_SEMANTIC_SCHOLAR` | bool | `True` | 启用 Semantic Scholar |
| `USE_OPENALEX` | bool | `True` | 启用 OpenAlex |
| `SEARCH_LIMIT` | int | `15` | 单次搜索返回数 |
| `TOP_K` | int | `20` | 最终输出论文数 |

---

## 输出格式

支持两种输出格式：

- **JSON**: `output/result.json` — 完整结构化结果
- **Markdown**: `output/result.md` — 人类可读报告

输出路径由 `settings.output_dir` 控制，默认为项目根目录下的 `output/`。
