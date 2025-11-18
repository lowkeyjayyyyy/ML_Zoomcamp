FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock /app/
COPY src/ /app/src/

RUN apt-get update && \
    apt-get install -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

RUN uv sync

EXPOSE 9696

ENTRYPOINT ["uv", "run", "gunicorn", "--chdir", "src", "--bind=0.0.0.0:9696", "predict:app"]