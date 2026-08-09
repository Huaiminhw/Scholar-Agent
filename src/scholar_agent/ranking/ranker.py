from scholar_agent.selector import Selector


class Ranker:
    """
    精排器（薄封装）
    委托给 Selector.score_batch() 做相关性打分。
    """

    def __init__(self) -> None:
        self._selector = Selector()

    def rank(self, query: str, papers: list[dict]) -> list[dict]:
        return self._selector.score_batch(query, papers)
