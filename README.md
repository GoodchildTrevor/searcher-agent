# MultiStep RAG Agent

A Retrieval-Augmented Generation (RAG) agent that iteratively selects and queries multiple search tools to produce accurate answers.

## Features

- Multi-tool support (vector search)
- Intelligent routing to decide whether to search more or answer
- Query expansion to generate better retrieval queries
- Iterative refinement with configurable iteration limits
- Async-first, production-ready API

## Quick start

1. Copy environment variables:
   cp .env.example .env
   Edit .env and set API_KEY (required). Optionally enable tracing.

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run locally:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8050 --reload
```

Or with Docker Compose:
```bash
docker-compose up --build
```

## Environment / Configuration

Key variables in .env (see .env.example):
- SEARCH_URL — vector search service URL
- OLLAMA_URL, OLLAMA_MODEL — LLM backend
- COLLECTIONS — comma-separated collections
- API_KEY — required for /agent-query
- AGENT_TRACE_ENABLED — set to `1` to enable JSONL trace logging
- AGENT_TRACE_LOG_PATH — trace file path (default: /app/logs/traces.jsonl)

## API

POST /agent-query (requires header X-API-KEY)

Example (Python requests):
```python
import requests

headers = {"X-API-KEY": "your-secret"}
payload = {
    "query": "What is the capital of France?",
    "max_iterations": 3,
    "expansion_count": 3,
    "confidence_threshold": 0.7,
    "should_answer": True,
    "tools": []
}

resp = requests.post("http://localhost:8050/agent-query", json=payload, headers=headers)
print(resp.status_code, resp.json())
```

Curl example:
```bash
curl -X POST http://localhost:8050/agent-query \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-secret" \
  -d '{"query":"Who invented the telephone?","max_iterations":2,"expansion_count":2,"confidence_threshold":0.5,"should_answer":true}'
```

## Tracing

Enable prompt/response tracing for prompt-tuning or debugging:
- Set AGENT_TRACE_ENABLED=1
- Traces are written as one JSON object per line to AGENT_TRACE_LOG_PATH

## Security

The /agent-query endpoint requires API_KEY (X-API-KEY header). Do not expose this service publicly without proper network controls.

## License

See LICENSE file.