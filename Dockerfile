FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/platform/hf_cache XDG_CACHE_HOME=/app/data/platform/cache \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false SHOPASSIST_REQUIRE_LOGIN=true
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-ocr.txt ./
RUN pip install --no-cache-dir -r requirements-ocr.txt
RUN useradd --uid 10001 --create-home shopassist
COPY --chown=shopassist:shopassist . .
RUN mkdir -p data/platform vector_store models/speech \
    && chown -R shopassist:shopassist data/platform vector_store models/speech
USER shopassist
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["python", "scripts/serve.py"]
