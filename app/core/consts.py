import os
from dotenv import load_dotenv

load_dotenv()

API_KEY: str = os.getenv("API_KEY")
TRACE_ENABLED: bool = os.getenv("AGENT_TRACE_ENABLED", "0") == "1"
TRACE_LOG_PATH: str  = os.getenv("AGENT_TRACE_LOG_PATH", "/app/logs/traces.jsonl")

SEARCH_URL: str = os.getenv("SEARCH_URL")

OLLAMA_URL: str = os.getenv("OLLAMA_URL")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL")

_collections_raw: str = os.getenv("COLLECTIONS")
COLLECTIONS: list[str] = [c.strip() for c in _collections_raw.split(",") if c.strip()]
