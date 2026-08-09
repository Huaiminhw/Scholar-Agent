from semanticscholar import SemanticScholar
import requests
import time
from scholar_agent.config import settings
from scholar_agent.utils.logger import logger

OA_BASE = "https://api.openalex.org"
OA_HEADERS = {"User-Agent": "ScholarAgent/1.0 (mailto:your_email@example.com)"}

_ss_client: SemanticScholar | None = None


def _oa_params(params: dict | None = None) -> dict:
    """为 OpenAlex 请求统一附加查询参数和 API Key。"""
    result = dict(params or {})
    if settings.openalex_api_key:
        result["api_key"] = settings.openalex_api_key
    return result


def _get_ss_client() -> SemanticScholar | None:
    """懒加载 SemanticScholar 单例。"""
    global _ss_client
    if _ss_client is None and settings.use_semantic_scholar:
        kwargs = {"timeout": 30}
        if settings.ss_api_key:
            kwargs["api_key"] = settings.ss_api_key
        _ss_client = SemanticScholar(**kwargs)
    return _ss_client if settings.use_semantic_scholar else None


# ─────────────────────────────────────────
# Semantic Scholar
# ─────────────────────────────────────────

def ss_search(query: str, limit: int | None = None) -> list[dict]:
    """Semantic Scholar 论文检索。"""
    client = _get_ss_client()
    if client is None:
        return []
    limit = limit or settings.search_limit
    time.sleep(0.5)
    for attempt in range(3):
        try:
            results = client.search_paper(query, limit=limit, fields=[
                'paperId', 'title', 'abstract', 'year',
                'citationCount', 'authors', 'venue',
            ])
            papers = []
            for p in results:
                papers.append({
                    'id': f"ss_{p.paperId}",
                    'ss_id': p.paperId,
                    'title': p.title or '',
                    'abstract': p.abstract or '',
                    'year': p.year,
                    'citations': p.citationCount or 0,
                    'authors': [a['name'] for a in (p.authors or [])],
                    'venue': p.venue or '',
                    'source': 'SemanticScholar',
                })
            logger.info(f"[SS] '{query}' -> {len(papers)} papers")
            return papers
        except Exception as e:
            logger.warning(f"[SS] attempt {attempt + 1} failed: {e}")
            time.sleep(min(2 ** attempt, 15))
    return []


# ─────────────────────────────────────────
# OpenAlex
# ─────────────────────────────────────────

def oa_search(query: str, limit: int | None = None) -> list[dict]:
    """OpenAlex 论文检索。"""
    if not settings.use_openalex:
        return []
    limit = limit or settings.search_limit
    time.sleep(0.5)
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{OA_BASE}/works",
                params=_oa_params({
                    "search": query,
                    "per-page": limit,
                    "select": "id,title,abstract_inverted_index,publication_year,cited_by_count,authorships,primary_location,open_access",
                }),
                headers=OA_HEADERS,
                timeout=20,
            )
            if resp.status_code == 429:
                logger.warning(f"[OA] Rate limited (429), waiting before retry")
                retry_after = int(resp.json().get("retryAfter", 60))
                time.sleep(min(retry_after, 30))
                continue
            if resp.status_code != 200:
                logger.warning(f"[OA] HTTP {resp.status_code}, retrying...")
                time.sleep(min(2 ** attempt, 10))
                continue
            data = resp.json()
            papers = []
            for w in data.get('results', []):
                abstract = _oa_reconstruct_abstract(w.get('abstract_inverted_index'))
                loc = w.get('primary_location') or {}
                src = loc.get('source') or {}
                venue = src.get('display_name', '')
                authors = [
                    a['author']['display_name']
                    for a in (w.get('authorships') or [])[:5]
                    if a.get('author')
                ]
                papers.append({
                    'id': f"oa_{w['id'].split('/')[-1]}",
                    'oa_id': w['id'],
                    'title': w.get('title', '') or '',
                    'abstract': abstract,
                    'year': w.get('publication_year'),
                    'citations': w.get('cited_by_count', 0),
                    'authors': authors,
                    'venue': venue,
                    'source': 'OpenAlex',
                })
            logger.info(f"[OA] '{query}' -> {len(papers)} papers")
            return papers
        except Exception as e:
            logger.warning(f"[OA] attempt {attempt + 1} failed: {e}")
            time.sleep(min(2 ** attempt, 10))
    return []


def _oa_reconstruct_abstract(inverted_index: dict | None) -> str:
    """从 OpenAlex 的倒排索引重建摘要文本。"""
    if not inverted_index:
        return ''
    word_pos = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_pos.append((pos, word))
    word_pos.sort(key=lambda x: x[0])
    return ' '.join(w for _, w in word_pos)


# ─────────────────────────────────────────
# 统一接口
# ─────────────────────────────────────────

# Simple in-memory cache to avoid repeated API calls
_search_cache: dict[str, list[dict]] = {}

def search_papers(query: str, limit: int | None = None) -> list[dict]:
    """双数据源统一检索接口，自动去重。"""
    # Check cache first
    cache_key = f"{query}|{limit}"
    if cache_key in _search_cache:
        logger.debug(f"[Cache] hit for '{query[:50]}'")
        return _search_cache[cache_key]
    
    all_papers: dict[str, dict] = {}
    if settings.use_openalex:
        for p in oa_search(query, limit):
            all_papers[p['id']] = p
    if settings.use_semantic_scholar:
        for p in ss_search(query, limit):
            if p['id'] not in all_papers:
                all_papers[p['id']] = p
    
    result = list(all_papers.values())
    _search_cache[cache_key] = result
    if len(_search_cache) > 100:
        _search_cache.clear()
    return result
