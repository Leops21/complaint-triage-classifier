FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .

CMD ["python", "scripts/00_profile_raw_data.py", "--help"]
