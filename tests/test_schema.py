from scholar_agent_v2.query_understanding.schema import (
    QueryAnalysisResult,
    QueryType,
    QueryIntent,
    SearchTask,
    RetrievalMode,
    ConstraintSet,
    ConstraintItem,
    ConstraintType,
    ConstraintOperator,
    RelationConstraint,
    ConstraintLevel,
)


def test_normal_complex_query():

    result = QueryAnalysisResult(
        original_query=(
            "找2023年以来使用强化学习优化LLM学术搜索Agent，"
            "并在真实科研查询数据集上验证的论文，"
            "最好来自ACL、NeurIPS等会议。"
        ),

        query_type=QueryType.BROAD_BY_DESCRIPTION,

        intent=QueryIntent(
            task=SearchTask.PAPER_SEARCH,
            mode=RetrievalMode.COMPREHENSIVE,
            goal="寻找满足多个技术和实验条件的学术搜索Agent论文"
        ),

        content=(
            "reinforcement learning optimized "
            "LLM-based academic literature search agents "
            "evaluated on real scientific queries"
        ),

        constraints=ConstraintSet(
            hard=[
                ConstraintItem(
                    id="C1",
                    type=ConstraintType.YEAR,
                    operator=ConstraintOperator.GTE,
                    value=2023,
                    source_text="2023年以来"
                ),

                ConstraintItem(
                    id="C2",
                    type=ConstraintType.TECHNOLOGY,
                    value="LLM-based academic search agent",
                    source_text="LLM学术搜索Agent"
                ),

                ConstraintItem(
                    id="C3",
                    type=ConstraintType.EVALUATION,
                    value="real scientific query dataset",
                    source_text="真实科研查询数据集上验证"
                ),
            ],

            soft=[
                ConstraintItem(
                    id="P1",
                    type=ConstraintType.VENUE,
                    operator=ConstraintOperator.ANY_OF,
                    values=["ACL", "NeurIPS"],
                    source_text="最好来自ACL、NeurIPS等会议"
                )
            ],

            exclude=[]
        ),

        relations=[
            RelationConstraint(
                id="R1",
                subject="reinforcement learning",
                predicate="optimize",
                object="LLM academic search agent",
                level=ConstraintLevel.HARD,
                source_text="使用强化学习优化LLM学术搜索Agent"
            )
        ]
    )

    print(result.model_dump_json(indent=2))

    assert result.constraints.hard[0].value == 2023
    assert result.relations[0].predicate == "optimize"
    assert result.constraints.soft[0].values == ["ACL", "NeurIPS"]
