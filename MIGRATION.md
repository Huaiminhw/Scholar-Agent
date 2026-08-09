# 迁移记录：初稿 → v1.0 重构

## 概述

从固定流水线系统重构为 Crawler/Selector 双角色 Agent 架构。

## 变更统计

| 指标 | 数值 |
|------|------|
| 新建文件 | 15 |
| 迁移文件 | 8 |
| 删除的旧代码 | 1（`config.py` 含泄露 Key） |
| 废弃的旧位置文件 | 7（保留未删） |

## 文件映射

| 旧位置 | 新位置 | 变更类型 | 说明 |
|--------|--------|----------|------|
| `config.py`（根目录） | `src/scholar_agent/config.py` | **废弃 + 重写** | pydantic-settings，API Key 移至 `.env` |
| — | `src/scholar_agent/crawler.py` | **新建** | 搜索策略 Agent |
| — | `src/scholar_agent/selector.py` | **新建** | 相关性评估 Agent |
| `src/agent.py` | `src/scholar_agent/agent.py` | **重写** | 整合 Crawler + Selector |
| `src/query_parser.py` | `src/scholar_agent/query/parser.py` | **迁移** | 标准化改动 |
| `src/query_evolver.py` | `src/scholar_agent/query/evolver.py` | **迁移** | 核心逻辑移入 Crawler |
| `src/retriever.py` | `src/scholar_agent/retrieval/retriever.py` | **迁移** | SemanticScholar 改单例 |
| — | `src/scholar_agent/retrieval/citation.py` | **拆分** | 从 retriever 独立引文函数 |
| `src/ranker.py` | `src/scholar_agent/ranking/ranker.py` | **重写** | 薄封装，委托给 Selector |
| `src/formatter.py` | `src/scholar_agent/presentation/formatter.py` | **迁移** | 路径改用 settings |
| — | `src/scholar_agent/utils/logger.py` | **新建** | loguru 封装 |
| `main.py` | `scripts/run_demo.py` | **重写** | 调用 SearchAgent |
| `evaluate.py` | `scripts/evaluate.py` | **迁移** | 导入改为 SearchAgent |
| `test.py` | `scripts/inspect_data.py` | **迁移** | 路径改用 settings |

## 配置变更

| 项目 | 旧 | 新 |
|------|----|----|
| API Key 管理 | 硬编码在 `config.py` | `.env` + pydantic-settings |
| 路径 | 硬编码 `/workspace/scholar_agent/` | `settings.output_dir` 自动推导 |
| 数据源开关 | 全局常量 | `settings.use_openalex` / `use_semantic_scholar` |

## 架构变更

| 旧架构 | 新架构 |
|--------|--------|
| 固定 8 步流水线 | Crawler（策略） + Selector（评估）双角色 |
| 通用 LLM 负责所有任务 | 独立角色分工 |
| `print()` 日志 | loguru 结构化日志 |
| 每次调用 new OpenAI() | 模块级单例 |

## 工程规范变更

| 项目 | 旧 | 新 |
|------|----|----|
| 依赖管理 | 无 | `pyproject.toml` |
| 版本管理 | 无 | `__version__ = '0.1.0'` |
| 测试 | 无 | `pytest tests/` |
| 文档 | 空 README | README + architecture.md + MIGRATION.md |
