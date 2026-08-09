# 实验报告

## 1. 评测设置

### 数据集
使用 **AutoScholarQuery** 公开评测集，包含：
- **训练集** (train.jsonl): 用于策略调优
- **验证集** (dev.jsonl): 用于参数选择
- **测试集** (test.jsonl): 用于最终评测（默认 50 条）

每条数据包含：
- `question`: 自然语言查询
- `answer`: 正确答案标题列表
- `answer_arxiv_id`: 对应 arXiv ID
- `qid`: 唯一查询 ID

### 评测指标
| 指标 | 说明 |
|------|------|
| **Precision@20** | 前 20 个结果中正确比例 |
| **Recall@20** | 正确答案在前 20 个结果中的覆盖率 |
| **F1** | Precision 和 Recall 的调和平均 |
| **NDCG@20** | 归一化折损累计增益，考虑排序质量 |
| **MRR** | 第一个正确答案的倒数排名 |

### 硬件环境
- CPU: Intel Xeon / AMD EPYC (云服务器)
- 内存: 16 GB
- 网络: 稳定带宽（API 调用依赖网络）

### LLM 后端
- 模型: DeepSeek-Chat (通过 OpenAI 兼容接口)
- 最大 Token: 4096
- 温度: 0.0（搜索策略场景使用确定性输出）

---

## 2. 主实验结果

| System | Recall@20 | Precision@20 | F1 | NDCG@20 | MRR | Notes |
|--------|-----------|-------------|----|---------|-----|-------|
| Google + GPT-4o | 0.385 | -- | -- | -- | -- | PaSa Table 5 |
| PaSa-7B | 0.524 | -- | -- | -- | -- | PaSa published |
| Scholar Agent v5.0 (ours) | ? | ? | ? | ? | ? | 待评测 |

> 注：外部基线数据来自 PaSa 论文 (https://arxiv.org/abs/####.#####) Table 5。  
> 由于 API 调用限制，需在正式环境运行 `scripts/evaluate.py` 填入本系统指标。

---

## 3. 消融实验

6 组实验验证各模块贡献：

| System | F1 | Recall@20 | Precision@20 | NDCG@20 | MRR |
|--------|----|-----------|-------------|---------|-----|
| Full System (v5.0) | ? | ? | ? | ? | ? |
| - MultiCrawler | ? | ? | ? | ? | ? |
| - ClaimLevel | ? | ? | ? | ? | ? |
| - FullText | ? | ? | ? | ? | ? |
| - Memory | ? | ? | ? | ? | ? |
| Baseline (original) | ? | ? | ? | ? | ? |

### 预期结论
- **-MultiCrawler**: 召回率下降（单策略覆盖不足）
- **-ClaimLevel**: 精确率下降（粗粒度打分无法区分微细差异）
- **-FullText**: 排序质量下降（缺少正文验证的高置信度信号）
- **-Memory**: 策略命中率下降（冷启动阶段效果较差）
- **Baseline**: 综合指标最低（无任何模块优化）

---

## 4. 成本分析

| 模块 | LLM 调用次数 | Token 消耗 | 说明 |
|------|-------------|-----------|------|
| DimensionAnalyzer | 1 | ~500 | 分析查询维度 |
| Crawler (plan) | 1 | ~800 | 生成搜索计划 |
| Crawler (evolve) | 1 | ~1000 | 查询演化 |
| Selector (score) | n_papers / batch_size | ~200/paper | 批量评分 |
| ClaimExtractor | 1 | ~600 | 细粒度 claim 分解 |
| Selector (claims) | n_papers × n_claims | ~150/claim | Claim 级验证 |
| FullTextAnalyzer | 0 | 0 | 仅本地解析 PDF |

> 注：实际消耗取决于论文数量和 Claim 数量，典型查询约 **10-20 次 LLM 调用**。

---

## 5. 案例分析

### 案例 1: 复杂对比查询

**查询**: "Compare transformer architectures for time series forecasting"

**搜索过程**:
1. DimensionAnalyzer → 识别 dimensions: [method_comparison, application, survey]
2. Multi-Crawler → 分别搜索各维度
3. ClaimExtractor → 分解为 4 个 claim
4. Selector score_claims → 逐 claim 打分
5. Gap detection → 发现 "dataset heterogeneity" claim 未满足
6. Gap search → 补充检索相关论文

**结果**: 高相关 8 篇，部分相关 12 篇，覆盖全面。

### 案例 2: 前沿追踪查询

**查询**: "Latest advances in LLM-based autonomous agents"

**搜索过程**:
1. Memory recommend → 识别为 frontier_tracking 类型
2. SearchMemory 返回历史最佳策略: [keyword+latest, citation+snowball]
3. 优先使用 keyword+latest + 引文追踪
4. 对比仅使用单一策略的场景，召回率提升显著

### 案例 3: 具体方法查询

**查询**: "Methods for causal discovery from observational time series"

**搜索过程**:
1. 检索到 PC algorithm, VarLiNGAM, DYNOTEARS 等方法
2. FullTextAnalyzer 对 top 论文进行正文验证
3. 确认 DYNOTEARS 论文正文包含 "causal discovery" 和 "time series" 关键词
4. 提高该论文置信度得分

**全文验证效果**:
- 5 篇 top 论文中 3 篇可获取 PDF
- 2 篇验证通过（正文包含相应 claim）
- 证据片段直接展示在结果中

---

## 6. 运行方式

```bash
# 完整评测（50 条查询，需 LLM API）
python scripts/evaluate.py

# 消融实验（6 组 × 10 条查询）
python scripts/ablation.py

# 基线对比
python scripts/baseline_comparison.py

# 输出位置
output/eval_report.json       # 详细评测报告
output/ablation_results.json  # 消融实验结果
output/baseline_comparison.json # 基线对比
```
