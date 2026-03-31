FROM python:3.12-slim

WORKDIR /app

# System deps for trafilatura
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .
COPY guardrails.py .
COPY config.yaml .

# Reports written to /reports (mounted volume)
RUN mkdir -p /reports

EXPOSE 8888

CMD ["python", "agent.py"]
