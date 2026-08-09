# Scholar Agent

LLM Agent for Comprehensive Academic Paper Search

**第八届中国研究生人工智能创新大赛（华为赛题三）参赛项目**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![GitHub tag](https://img.shields.io/github/v/tag/libotao2423139622-creator/scholar-agent)](https://github.com/libotao2423139622-creator/scholar-agent)

## 项目简介

构建一个端到端的学术论文智能搜索 Agent。用户输入自然语言学术查询，系统自动完成**查询理解 → 多策略检索 → 细粒度打分 → 全文验证 → 结果归纳**全流程。

### 四大创新点

1. **Multi-Crawler 并行搜索** — 多维度解耦检索，提升召回率
2. **Claim-level 细粒度打分** — 逐断言对齐，提升精确率和可解释性
3. **全文解析 + 深层引文追踪** — 从正文提取隐藏关联论文
4. **搜索经验记忆** — 跨查询策略复用，越用越聪明

### 架构

```
用户查询 → DimensionAnalyzer → Multi-Crawler → ResultMerger
         → Selector (Claim-level) → Gap Analysis → FullTextAnalyzer
         → Memory Record → 分层输出 (High/Partial)
```

## 快速开始

### 1. 安装

```bash
git clone https://github.com/libotao2423139622-creator/scholar-agent.git
cd scholar-agent
pip install -e .
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env` 填入你的 API Key：

```ini
LLM_API_KEY=your_deepseek_key_here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
USE_SEMANTIC_SCHOLAR=false
USE_OPENALEX=true
```

### 3. 运行

```python
from scholar_agent import SearchAgent

agent = SearchAgent()
result = agent.run("基于Transformer的医学图像分割方法与U-Net的对比分析")
```

或使用演示脚本：

```bash
python scripts/run_demo.py
```

### 4. 运行测试

```bash
pytest tests/ -v
```

## 项目结构

```
scholar-agent/
├── pyproject.toml
├── .env.example
├── Dockerfile
├── README.md
├── docs/
│   ├── project_doc.md         # 完整项目文档
│   ├── api.md                 # API 使用文档
│   └── experiments.md         # 评测+消融实验报告
├── src/
│   └── scholar_agent/
│       ├── agent.py           # SearchAgent 主控
│       ├── crawler.py         # 搜索策略智能体
│       ├── selector.py        # 相关性评估智能体
│       ├── claim_extractor.py # Claim 级查询分解
│       ├── dimension_analyzer.py # 维度分析
│       ├── result_merger.py   # 结果去重融合
│       ├── full_text.py       # 全文解析+PDF验证
│       ├── memory.py          # 搜索经验记忆
│       ├── config.py          # pydantic-settings 配置
│       ├── query/             # 查询解析+演化
│       ├── retrieval/         # 双数据源检索+引文
│       ├── presentation/      # 报告输出
│       └── utils/             # 日志等工具
├── scripts/
│   ├── run_demo.py
│   ├── evaluate.py            # 评测脚本
│   ├── ablation.py            # 消融实验
│   └── baseline_comparison.py # 基线对比
├── tests/                     # 单元测试
└── data/                      # 数据集（gitignored）
```

## 评测结果

| System | Recall@20 | Precision@20 | F1 | NDCG@20 | MRR |
|--------|-----------|-------------|----|---------|-----|
| Google + GPT-4o | 0.385 | — | — | — | — |
| PaSa-7B | 0.524 | — | — | — | — |
| **Scholar Agent (ours)** | *待评测* | *待评测* | *待评测* | *待评测* | *待评测* |

```bash
# 运行评测
python scripts/evaluate.py
python scripts/ablation.py
python scripts/baseline_comparison.py
```

## 依赖

- Python >= 3.11
- openai >= 1.0
- semanticscholar >= 0.7
- pydantic-settings >= 2.0
- loguru >= 0.7
- requests >= 2.28
- PyMuPDF (fitz) >= 1.23

## 许可

仅供参赛使用。


使用方式：
1. 进入项目激活环境：
.\.venv\Scripts\Activate.ps1

2. 测试
先测试1条，确认流程和费用：python scripts\evaluate.py --num_samples 1

测试前10条：python scripts\evaluate.py --num_samples 10

如果希望同时保存控制台日志：
python scripts\evaluate.py --num_samples 10 2>&1 |
    Tee-Object -FilePath output\eval_10.log

完成后结果保存在：output\eval_results.json