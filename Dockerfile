FROM python:3.11-slim
WORKDIR /app

# ffmpeg for decoding mp3/m4a if you enable them
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files first (better Docker caching)
COPY pyproject.toml poetry.lock* /app/

# Install dependencies into the container Python (no venv inside Docker)
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Copy application code
COPY app /app/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]