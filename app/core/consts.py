import os
from dotenv import load_dotenv

load_dotenv()

SEARCH_URL: str = os.getenv("SEARCH_URL")

OLLAMA_URL: str = os.getenv("OLLAMA_URL")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL")

_collections_raw: str = os.getenv("COLLECTIONS")
COLLECTIONS: list[str] = [c.strip() for c in _collections_raw.split(",") if c.strip()]
