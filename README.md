# Scholar Agent V2

面向学术论文检索的智能 Agent。当前仓库只保留 V2 实现，V1 源码、脚本和文档已经移除。

## 当前能力

- 查询理解：把自然语言需求解析为结构化约束，并进行校验与必要的修复。
- 查询生成：根据查询理解结果规划、扩展、校验并去重检索词。
- 多源检索：支持冻结的本地 PaSa、OpenAlex、Semantic Scholar、arXiv 和 Google Search，包含路线构建、并发调度和调用指标。
- 自适应检索：使用本地 BGE-M3 计算语义特征，并根据边际收益追加检索预算。
- 自适应召回深度：综合查询先按每路 28 条召回，仅将取满且候选收益足够的路线追加到 50 条。
- 候选池：执行跨来源去重、字段融合、约束过滤、信息补全和轻量重排。
- `selector`、`ranking`、`retrieval_planning` 目前仍是后续阶段的开发目录。

## 项目结构

```text
src/scholar_agent_v2/
├── query_understanding/   # 查询理解、Schema、Prompt 与校验
├── query_generation/      # 查询规划、生成、校验与去重
├── retrieval/             # 路线构建、检索调度和数据源适配器
│   ├── adaptive/          # BGE-M3 语义特征与自适应路由
│   └── sources/           # PaSa Local、OpenAlex、Semantic Scholar、arXiv、Google Search
├── candidate_pool/        # 去重、融合、过滤、补全与预排序
├── retrieval_planning/    # 后续检索规划模块
├── selector/              # 后续复杂条件选择模块
├── ranking/               # 后续最终排序模块
└── main.py                # V2 入口占位文件

tests/                     # V2 单元测试和显式启用的真实 API 测试
data/                      # 本地数据集，不提交 Git
models/                    # 本地模型，不提交 Git
```

## 环境要求

- Python 3.11 或更高版本
- 建议使用项目根目录下的 `.venv`
- 自适应语义检索需要本地 BGE-M3 模型

在 PowerShell 中安装：

```powershell
cd "D:scholar_agent"
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

复制环境变量模板并填写自己的密钥：

```powershell
Copy-Item .env.example .env
```

`.env` 包含密钥且已被 Git 忽略，不要提交到仓库。

## 配置

主要环境变量如下：

| 变量 | 用途 |
| --- | --- |
| `LLM_API_KEY` | OpenAI 兼容接口密钥 |
| `LLM_BASE_URL` | OpenAI 兼容接口地址 |
| `LLM_MODEL` | 模型名称 |
| `RETRIEVAL_PROFILE` | `frozen_local` 或 `competition_online`，默认后者 |
| `OPENALEX_API_KEY` | OpenAlex API 密钥 |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API 密钥，可选 |
| `GOOGLE_SEARCH_API_KEY` | Google Custom Search JSON API 密钥 |
| `GOOGLE_SEARCH_ENGINE_ID` | Google Programmable Search Engine ID（`cx`） |
| `GOOGLE_SEARCH_QUERY_SUFFIX` | 追加到 Google 查询末尾的可选限定词 |
| `USE_OPENALEX` | 是否启用 OpenAlex |
| `USE_SEMANTIC_SCHOLAR` | 是否启用 Semantic Scholar |
| `USE_ARXIV` | 是否启用 arXiv |
| `USE_GOOGLE_SEARCH` | 是否启用 Google Search，默认关闭 |
| `PASA_BM25_INDEX_PATH` | 冻结 PaSa SQLite FTS5 索引路径 |
| `PASA_QUERY_CACHE_PATH` | 本地冻结评测使用的查询缓存 |
| `PASA_CITATION_GRAPH_PATH` | 从 PaSa `sections` 构建的冻结引用图 |
| `QUERY_EXECUTION_MODE` | `frozen` 读取缓存；`live` 运行当前查询理解与扩展 |
| `BGE_M3_MODEL_PATH` | 本地 BGE-M3 模型目录 |
| `USE_CITATION_SNOWBALL` | 是否启用一跳 Forward/Backward 引用扩展 |
| `CITATION_CHANNEL_WEIGHT` | Citation 通道候选的排序权重，当前默认 `0.70` |

本地冻结检索和比赛开放源共用 RouteBuilder、Orchestrator、CandidatePool、
Hard Filter 和 Prerank，只切换 Retrieval Profile：

比赛在线交互测试（真实 LLM + 开放源 + 结构化 JSON）：

```powershell
.\.venv\Scripts\python.exe -u scripts\比赛在线测试.py
```

```powershell
# 完全离线、固定语料库
$env:RETRIEVAL_PROFILE="frozen_local"
$env:QUERY_EXECUTION_MODE="frozen"
python scripts\eval_pasa_formal_pipeline.py --start 0 --limit 5

# 比赛开放源；具体来源继续由 USE_* 控制
$env:RETRIEVAL_PROFILE="competition_online"
$env:QUERY_EXECUTION_MODE="live"
```

业务入口可统一通过
`ScholarRetrievalPipeline.from_environment()` 构建 Pipeline；Profile 只决定
Adapter 集合，不改变 RouteBuilder、CandidatePool、Filter 或 Prerank。

综合查询的默认深度预算集中定义在 `retrieval/route_builder.py` 的
`MODE_BUDGETS`，正式 PaSa 评测也可在命令行覆盖：

```powershell
python scripts\eval_pasa_formal_pipeline.py `
  --cache results\pasa_v2_query_cache\realscholarquery_v3_queries.jsonl `
  --start 0 --limit 50 `
  --first-round-limit 28 `
  --max-route-limit 50 `
  --min-new-candidate-rate 0.15
```

终端和 JSON 报告都会记录每轮候选新增数、新增率、重复率、耗时、
停止原因，以及 Pool Recall、Recall@40 和 Recall@100。

## Citation Snowball

PaSa 原始论文记录把引用保存为 `sections` 中的论文标题。首次使用前构建
冻结的双向引用图：

```powershell
python scripts\build_pasa_citation_graph.py
```

正式本地评测通过开关启用一跳 Snowball：

```powershell
python scripts\eval_pasa_formal_pipeline.py `
  --cache results\pasa_v2_query_cache\realscholarquery_v3_queries.jsonl `
  --start 0 --limit 50 `
  --citation-snowball `
  --citation-seed-top-k 3 `
  --citation-forward-top-k 30 `
  --citation-backward-top-k 30
```

Pipeline 先对关键词候选进行 Prerank，只选择达到相关性阈值的 Top-K
种子；多个种子连接同一引用候选时累积得分，再对 Forward、Backward
分别做全局 Top-K。近期查询默认只执行 Forward。报告同时保存引用扩展
前后的 Pool Recall、Citation 新增 Gold、重复率、耗时与停止原因。

## Web 界面

项目包含一个不依赖 Node.js 的轻量 Web 界面，可运行完整论文检索，也可只查看问题理解结果：

```powershell
.\.venv\Scripts\python.exe web_app.py
```

然后访问 <http://127.0.0.1:8000>。完整检索沿用 `.env` 中的 LLM 和检索源配置；停止服务请在终端按 `Ctrl+C`。

## 测试

运行不访问真实外部服务的测试：

```powershell
python -m pytest -s
```

真实 LLM 测试默认跳过，需要显式启用：

```powershell
$env:RUN_REAL_LLM_TESTS="1"
python -m pytest -s tests\test_analyzer.py tests\test_query_generation_real.py
```

真实检索测试默认跳过，需要显式启用：

```powershell
$env:RUN_REAL_RETRIEVAL_TESTS="1"
python -m pytest -s tests\test_retrieval_real.py
```

端到端真实检索流水线使用独立开关：

```powershell
$env:RUN_REAL_RETRIEVAL_PIPELINE_TESTS="1"
python -m pytest -s tests\test_retrieval_pipeline_real.py
```

这些真实测试会产生 API 调用、网络等待和相应费用。

## 开发状态

当前 V2 的查询理解、查询生成、论文召回和候选池已有独立测试；统一命令行入口尚未实现，目前请通过对应模块的测试验证功能。
