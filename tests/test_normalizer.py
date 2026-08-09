from scholar_agent_v2.retrieval.normalizer import (
    normalize_arxiv_id,
    normalize_doi,
    reconstruct_openalex_abstract,
)


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1000/XYZ") == "10.1000/xyz"


def test_normalize_arxiv_version():
    assert normalize_arxiv_id("arXiv:2401.12345v2") == "2401.12345"


def test_openalex_abstract_reconstruction():
    inv = {"hello": [0], "world": [1]}
    assert reconstruct_openalex_abstract(inv) == "hello world"
