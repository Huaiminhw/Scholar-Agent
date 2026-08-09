FROM python:3.11-slim

WORKDIR /app

# 系统依赖（OCR / PDF 工具仅在需要时安装）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖以利用 Docker 缓存
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# 复制代码
COPY src/ src/
COPY scripts/ scripts/
COPY docs/ docs/
COPY .env.example .env.example

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from scholar_agent import SearchAgent; print('OK')" || exit 1

# 数据卷
VOLUME ["/app/data", "/app/output"]

# 默认入口：演示脚本
CMD ["python", "scripts/run_demo.py"]
