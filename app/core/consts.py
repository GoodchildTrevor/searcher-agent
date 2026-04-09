import os
from dotenv import load_dotenv

load_dotenv()

SEARCH_URL:str = os.getenv("SEARCH_URL")

OLLAMA_URL: str = os.getenv("OLLAMA_URL")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL") 

COLLECTIONS = ["mech", "docs"]
