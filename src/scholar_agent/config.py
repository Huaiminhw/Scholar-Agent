from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # 数据源开关
    use_semantic_scholar: bool = True
    use_openalex: bool = True

    # API Keys（从 .env 读取）
    ss_api_key: str | None = None
    openalex_api_key: str | None = None
    llm_api_key: str = ''
    llm_base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    llm_model: str = 'qwen-plus'

    # 消融实验配置开关
    use_single_crawler: bool = False
    use_score_batch: bool = False
    skip_full_text: bool = False
    skip_memory: bool = False
    use_original_pipeline: bool = False

    # 检索参数
    search_limit: int = 15
    max_rounds: int = 3
    top_k: int = 20

    # 项目路径（自动推导，不硬编码）
    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def output_dir(self) -> Path:
        d = self.project_root / 'output'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def data_dir(self) -> Path:
        return self.project_root / 'data'

    model_config = {'env_file': '.env', 'env_file_encoding': 'utf-8'}


settings = Settings()
