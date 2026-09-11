# Searcher Agent Project

An advanced AI agent designed to search and retrieve information from vector databases using natural language queries. This project implements a sophisticated agentic workflow for intelligent document retrieval, analysis, and reasoning.

## 🚀 Features

- **Vector Database Integration**: Connects with local vector stores (Qdrant, Chroma) for semantic search
- **AI Agent Framework**: Built on LangGraph/Superagent architecture enabling multi-step reasoning capabilities  
- **Natural Language Processing**: Converts user queries into optimized database searches and generates human-readable answers
- **Docker Support**: Containerized deployment ready with Docker Compose

## 🛠 Tech Stack

- **Python 3.x** - Core application logic
- **LangGraph / Superagent** - Agent orchestration framework
- **Ollama + Llama Models** - Local Large Language Model integration
- **Qdrant / ChromaDB** - Vector database backends for semantic search
- **Docker & Docker Compose** - Containerization and environment management

## 📂 Project Structure

```text
searcher-agent/
├── app/                  # Application source code
│   ├── agent/            # Agent logic (nodes, main runner)
│   │   ├── nodes.py      # Graph node definitions
│   │   └── main_agent.py 
│   ├── core/             # Core configuration & models
│   │   ├── agent_settings.py
│   │   ├── llm.py        # LLM initialization
│   │   └── prompts.py    # System prompts templates
│   ├── tools/            # Agent tool definitions (e.g., vector search)
│   │   └── vector_search.py
│   └── main.py           # Entry point
├── Dockerfile            # Container build instructions
├── docker-compose.yml    # Service orchestration
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variable template
```

## ⚙️ Getting Started

### Prerequisites
- Docker & Docker Compose installed on your machine.
- (Optional) Ollama running locally if not included in the container setup.

### 1. Clone Repository
```bash
git clone https://github.com/GoodchildTrevor/searcher-agent.git
cd searcher-agent
```

### 2. Configuration
Copy the example environment file and adjust settings as needed:
```bash
cp .env.example .env
```
Edit `.env` to configure your LLM provider, Vector Store connection details (e.g., Qdrant host/port), and API keys.

### 3. Run with Docker Compose
Build the image and start all services in detached mode:
```bash
docker-compose up --build -d
```

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

*Built with LangGraph, Ollama, and modern AI agent frameworks.*