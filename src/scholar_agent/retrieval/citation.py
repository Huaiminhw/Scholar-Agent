from semanticscholar import SemanticScholar
import requests
import time
from scholar_agent.config import settings
from scholar_agent.utils.logger import logger
from scholar_agent.retrieval.retriever import OA_BASE, OA_HEADERS, _oa_params, _oa_reconstruct_abstract, _get_ss_client

# ─────────────────────────────────────────
# Semantic Scholar 引文追踪
# ─────────────────────────────────────────

def ss_get_references(ss_id: str, limit: int = 10) -> list[dict]:
    """Semantic Scholar 引文追踪。"""
    client = _get_ss_client()
    if client is None:
        return []
    time.sleep(0.5)
    for attempt in range(3):
        try:
            paper = client.get_paper(ss_id, fields=['references'])
            refs = []
            for ref in (paper.references or [])[:limit]:
                if ref.get('paperId'):
                    refs.append({
                        'id': f"ss_{ref['paperId']}",
                        'ss_id': ref['paperId'],
                        'title': ref.get('title', ''),
                        'abstract': ref.get('abstract', ''),
                        'year': ref.get('year'),
                        'citations': ref.get('citationCount', 0),
                        'authors': [],
                        'venue': ref.get('venue', ''),
                        'source': 'SemanticScholar',
                    })
            logger.info(f"[SS refs] -> {len(refs)} references")
            return refs
        except Exception as e:
            logger.warning(f"[SS refs] attempt {attempt + 1} failed: {e}")
            time.sleep(min(2 ** attempt, 15))
    return []


# ─────────────────────────────────────────
# OpenAlex 引文追踪
# ─────────────────────────────────────────

def oa_get_references(oa_id: str, limit: int = 10) -> list[dict]:
    """OpenAlex 引文追踪。"""
    time.sleep(0.5)
    for attempt in range(3):
        try:
            work_id = oa_id.split('/')[-1] if '/' in oa_id else oa_id
            resp = requests.get(
                f"{OA_BASE}/works/{work_id}",
                params=_oa_params({"select": "referenced_works"}),
                headers=OA_HEADERS,
                timeout=20,
            )
            data = resp.json()
            ref_ids = (data.get('referenced_works') or [])[:limit]

            if not ref_ids:
                return []

            refs = []
            for ref_url in ref_ids:
                ref_work_id = ref_url.split('/')[-1]
                try:
                    ref_resp = requests.get(
                        f"{OA_BASE}/works/{ref_work_id}",
                        params=_oa_params({"select": "id,title,abstract_inverted_index,publication_year,cited_by_count,primary_location"}),
                        headers=OA_HEADERS,
                        timeout=20,
                    )
                    w = ref_resp.json()
                    if not w.get('title'):
                        continue
                    abstract = _oa_reconstruct_abstract(w.get('abstract_inverted_index'))
                    loc = w.get('primary_location') or {}
                    src = loc.get('source') or {}
                    refs.append({
                        'id': f"oa_{w['id'].split('/')[-1]}",
                        'oa_id': w['id'],
                        'title': w.get('title', '') or '',
                        'abstract': abstract,
                        'year': w.get('publication_year'),
                        'citations': w.get('cited_by_count', 0),
                        'authors': [],
                        'venue': src.get('display_name', ''),
                        'source': 'OpenAlex',
                    })
                    time.sleep(0.3)
                except Exception:
                    continue

            logger.info(f"[OA refs] -> {len(refs)} references")
            return refs
        except Exception as e:
            logger.warning(f"[OA refs] attempt {attempt + 1} failed: {e}")
            time.sleep(min(2 ** attempt, 10))
    return []


# ─────────────────────────────────────────
# 统一接口
# ─────────────────────────────────────────

# Simple in-memory cache for citation tracking
_citation_cache: dict[str, list[dict]] = {}

def get_references(paper_id: str, limit: int = 10) -> list[dict]:
    """根据论文来源调用对应的引文追踪。"""
    cache_key = f"{paper_id}|{limit}"
    if cache_key in _citation_cache:
        logger.debug(f"[Cache] citation hit for '{paper_id[:20]}'")
        return _citation_cache[cache_key]
    
    if paper_id.startswith('ss_'):
        result = ss_get_references(paper_id[3:], limit)
    elif paper_id.startswith('oa_'):
        result = oa_get_references(paper_id[3:], limit)
    else:
        return []
    
    _citation_cache[cache_key] = result
    if len(_citation_cache) > 200:
        _citation_cache.clear()
    return result
