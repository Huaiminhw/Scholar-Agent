# Scholar Agent V2

面向学术论文检索的智能 Agent。当前仓库只保留 V2 实现，V1 源码、脚本和文档已经移除。

## 当前能力

- 查询理解：把自然语言需求解析为结构化约束，并进行校验与必要的修复。
- 查询生成：根据查询理解结果规划、扩展、校验并去重检索词。
- 多源检索：支持 OpenAlex、Semantic Scholar 和 arXiv，包含路线构建、并发调度和调用指标。
- 自适应检索：使用本地 BGE-M3 计算语义特征，并根据边际收益追加检索预算。
- 候选池：执行跨来源去重、字段融合、约束过滤、信息补全和轻量重排。
- `selector`、`ranking`、`retrieval_planning` 目前仍是后续阶段的开发目录。

## 项目结构

```text
src/scholar_agent_v2/
├── query_understanding/   # 查询理解、Schema、Prompt 与校验
├── query_generation/      # 查询规划、生成、校验与去重
├── retrieval/             # 路线构建、检索调度和数据源适配器
│   ├── adaptive/          # BGE-M3 语义特征与自适应路由
│   └── sources/           # OpenAlex、Semantic Scholar、arXiv
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
cd "D:\桌面\草稿\比赛\scholar_agent_师弟"
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
| `OPENALEX_API_KEY` | OpenAlex API 密钥 |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API 密钥，可选 |
| `USE_OPENALEX` | 是否启用 OpenAlex |
| `USE_SEMANTIC_SCHOLAR` | 是否启用 Semantic Scholar |
| `USE_ARXIV` | 是否启用 arXiv |
| `BGE_M3_MODEL_PATH` | 本地 BGE-M3 模型目录 |

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
