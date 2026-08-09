import fitz
import requests
import re
from pathlib import Path
from scholar_agent.config import settings
from scholar_agent.utils.logger import logger


SECTION_PATTERNS = [
    (r'introduction|背景|引言', 'introduction'),
    (r'method|方法|approach|proposed|model architecture|system', 'method'),
    (r'experiment|实验|evaluation|result|performance|benchmark', 'experiments'),
    (r'related work|相关工作|related work', 'related_work'),
    (r'references|参考文献|bibliography', 'references'),
]


class FullTextAnalyzer:
    """
    FullTextAnalyzer: 全文分析器

    职责：
    1. 下载论文 PDF（从 OpenAlex / Semantic Scholar 获取 OA URL）
    2. 解析 PDF 章节结构
    3. 从参考文献章节提取引用标题
    4. 验证 claim 在正文中的证据
    """

    def __init__(self) -> None:
        self.cache_dir = settings.project_root / 'data' / '.fulltext_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_full_text(self, paper: dict) -> Path | None:
        """下载论文 PDF，返回本地缓存路径。"""
        # 检查缓存
        paper_id = paper.get('id', 'unknown')
        cached = self.cache_dir / f"{paper_id}.pdf"
        if cached.exists():
            logger.debug(f"FullText: cache hit for {paper_id}")
            return cached

        # 尝试获取 OA URL
        oa_url = paper.get('open_access_url') or self._find_oa_url(paper)
        if not oa_url:
            logger.debug(f"FullText: no OA URL for {paper_id}")
            return None

        try:
            resp = requests.get(oa_url, timeout=30, headers={"User-Agent": "ScholarAgent/1.0"})
            if resp.status_code == 200:
                cached.write_bytes(resp.content)
                logger.info(f"FullText: downloaded PDF for {paper_id}")
                return cached
        except Exception as e:
            logger.warning(f"FullText: download failed for {paper_id}: {e}")

        return None

    def _find_oa_url(self, paper: dict) -> str | None:
        """从 paper dict 查找 OA 链接。"""
        # OpenAlex 格式
        oa = paper.get('open_access', {})
        if isinstance(oa, dict):
            url = oa.get('oa_url') or oa.get('url')
            if url:
                return url
        # 备用字段
        for key in ['pdf_url', 'oa_url', 'open_access_url']:
            url = paper.get(key)
            if url:
                return url
        return None

    def parse_sections(self, pdf_path: Path) -> dict[str, str]:
        """解析 PDF 章节，返回 {section_name: text}。"""
        sections: dict[str, str] = {}
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            logger.warning(f"FullText: failed to parse {pdf_path}: {e}")
            return {'full_text': ''}

        # 按章节标题切分
        lines = full_text.split('\n')
        current_section = 'full_text'
        section_texts: dict[str, list[str]] = {current_section: []}

        for line in lines:
            line_stripped = line.strip()
            matched = False
            for pattern, section_name in SECTION_PATTERNS:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    current_section = section_name
                    if current_section not in section_texts:
                        section_texts[current_section] = []
                    matched = True
                    break
            section_texts.setdefault(current_section, []).append(line_stripped)

        for sec_name, sec_lines in section_texts.items():
            text = '\n'.join(sec_lines).strip()
            if text:
                sections[sec_name] = text[:5000]  # 截断防止超长

        if not sections:
            sections['full_text'] = full_text[:5000]

        logger.info(f"FullText: parsed {len(sections)} sections from {pdf_path.name}")
        return sections

    def extract_references(self, sections: dict[str, str]) -> list[str]:
        """从参考文献章节提取论文标题。"""
        refs_text = ''
        for key in ['references', 'related_work']:
            if key in sections:
                refs_text = sections[key]
                break

        if not refs_text:
            return []

        # 尝试提取引用标题（多种格式）
        refs = []
        lines = refs_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line or len(line) < 20:
                continue
            # 去掉编号前缀
            cleaned = re.sub(r'^\[\d+\]\s*', '', line)
            cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
            # 取句号前的部分作为标题（引用格式通常是 "Title. Authors. Journal."）
            title_part = cleaned.split('.')[0] if '.' in cleaned else cleaned[:100]
            if len(title_part) > 15:
                refs.append(title_part.strip())

        refs = refs[:50]
        logger.info(f"FullText: extracted {len(refs)} references")
        return refs

    def verify_claim_in_fulltext(self, paper: dict, claim_text: str) -> dict:
        """在正文中搜索 claim 关键词，验证是否有证据支持。"""
        pdf_path = self.fetch_full_text(paper)
        if not pdf_path:
            return {"found": False, "location": "", "snippet": ""}

        sections = self.parse_sections(pdf_path)

        # 优先在 method/experiments 章节搜索
        search_sections = ['method', 'experiments', 'introduction']
        keywords = claim_text.lower().split()[:5]

        for sec_name in search_sections:
            if sec_name not in sections:
                continue
            text = sections[sec_name].lower()
            # 检查关键词是否出现在同一段落（前后50字符内）
            for kw in keywords:
                if len(kw) < 3:
                    continue
                idx = text.find(kw)
                if idx >= 0:
                    start = max(0, idx - 50)
                    end = min(len(text), idx + len(kw) + 50)
                    snippet = sections[sec_name][start:end]
                    return {
                        "found": True,
                        "location": sec_name,
                        "snippet": snippet,
                    }

        return {"found": False, "location": "", "snippet": ""}
