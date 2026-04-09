import os
from dotenv import load_dotenv

load_dotenv()

SEARCH_URL:str = os.getenv("SEARCH_URL", "http://search_qdrant:8033/vector_search")

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "ministral-3:14b") #"nemotron-cascade-2:30b"

COLLECTIONS = ["mech", "docs"]
