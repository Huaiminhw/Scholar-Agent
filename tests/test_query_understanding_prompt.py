from scholar_agent_v2.query_understanding.prompt import (
    PromptConfig,
    QUERY_ANALYSIS_SCHEMA_SUMMARY,
    build_query_analysis_system_prompt,
)


def test_entity_schema_summary_has_its_own_closed_enum() -> None:
    entity_block = QUERY_ANALYSIS_SCHEMA_SUMMARY.split(
        '"entities":', 1
    )[1].split('"domains":', 1)[0]

    assert '"topic|method|model|dataset|author|venue|paper|other"' in entity_block
    assert '"task|' not in entity_block
    assert "technology" not in entity_block
    assert "evaluation" not in entity_block


def test_main_prompt_defines_field_boundaries_and_relation_semantics() -> None:
    prompt = build_query_analysis_system_prompt(
        PromptConfig(schema_mode="none", current_year=2026)
    )

    assert "not a dataset-size constraint" in prompt
    assert "Level means retrieval necessity, not truth/confidence" in prompt
    assert "Never copy intent.task/ConstraintItem.type into EntityType" in prompt
    assert "Cross-role overlap is allowed only" in prompt
    assert (
        "BROAD_BY_DESCRIPTION: normal research-topic, method, benchmark, "
        "phenomenon, or scientific-relation search."
    ) in prompt


def test_boundary_example_is_optional_and_does_not_cost_default_tokens() -> None:
    default_prompt = build_query_analysis_system_prompt(
        PromptConfig(
            schema_mode="summary",
            include_examples=False,
            current_year=2026,
        )
    )
    example_prompt = build_query_analysis_system_prompt(
        PromptConfig(
            schema_mode="summary",
            include_examples=True,
            current_year=2026,
        )
    )

    assert "EXAMPLE 4 (boundary counterexample)" not in default_prompt
    assert "EXAMPLE 4 (boundary counterexample)" in example_prompt


def test_default_prompt_size_has_an_explicit_budget_guard() -> None:
    compact = build_query_analysis_system_prompt(
        PromptConfig(schema_mode="none", current_year=2026)
    )
    default = build_query_analysis_system_prompt(
        PromptConfig(schema_mode="summary", current_year=2026)
    )

    assert len(compact) <= 7_500
    assert len(default) <= 10_000
