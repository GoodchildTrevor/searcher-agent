# MultiStep RAG Agent

A sophisticated Retrieval-Augmented Generation (RAG) agent that dynamically selects and uses multiple search tools to answer queries through an iterative refinement process.

## 🚀 Features

- **Multi-Tool Support**: Seamlessly integrates multiple search tools (vector, keyword, etc.)
- **Intelligent Routing**: Evaluates context sufficiency and decides whether to search more or answer
- **Query Expansion**: Generates multiple query variations for better retrieval
- **Iterative Refinement**: Performs multiple search cycles until confident enough to answer
- **Async-First Design**: Efficient concurrent operations

```http
response = requests.post(
    "http://localhost:8050/agent-query",
    json={
        "query": "What is the capital of France?",
        "max_iterations": 3,
        "expansion_count": 3,
        "confidence_threshold": 0.7,
        "should_answer": True
    }
)
```