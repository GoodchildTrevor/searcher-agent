from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Optional


class AgentConfigRequest(BaseModel):
    """
    Configuration parameters for the RAG agent.
    All fields are optional and will use default values if not provided.
    """
    query: str = Field(..., min_length=1, description="User's question or query")
    tools: list[str] = Field(
        default_factory=list,
        description="List of tool names to use for search"
    )
    chat_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Previous conversation history"
    )
    max_iterations: int = Field(default=3, ge=1, le=10, description="Maximum search cycles")
    expansion_count: int = Field(default=3, ge=1, le=5, description="Number of query variations")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold for answering")
    should_answer: bool = Field(default=False, description="Whether to generate final answer or just return info")
    raw_search: bool = Field(default=False, description="Whether to search just user query or start with agent expanded queries")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How to configure OAuth in the application?",
                "chat_history": [{"role": "user", "content": "Hello"}],
                "max_iterations": 3,
                "expansion_count": 3,
                "confidence_threshold": 0.7,
                "should_answer": True,
                "raw_search": False
            }
        }
    )


class SourceDocument(BaseModel):
    """
    Retrieved source document model.
    """
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Document source identifier")
    relevance: Optional[float] = Field(default=None, description="Relevance score")
    tool: Optional[str] = Field(default=None, description="Tool that retrieved this document")
    page_start: int = Field(default=1, description="Start page in the source document")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "OAuth configuration guide...",
                "source": "docs/oauth.md",
                "relevance": 0.92,
                "tool": "vector_search"
            }
        }
    )


class AgentResponse(BaseModel):
    """
    Response model for agent execution.
    """
    answer: str = Field(..., description="Generated answer or info message")
    iterations: int = Field(..., description="Number of search iterations performed")
    final_confidence: Optional[float] = Field(default=None, description="Final confidence score")
    sources: list[SourceDocument] = Field(default_factory=list, description="Retrieved source documents")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    tools_used: list[str] = Field(default_factory=list, description="Names of tools used during execution")
    tool_queries: dict[str, list[str]] = Field(default_factory=dict, description="Queries sent to each tool")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "To configure OAuth, follow these steps...",
                "iterations": 2,
                "final_confidence": 0.85,
                "sources": [
                    {
                        "content": "OAuth configuration guide...",
                        "source": "docs/oauth.md",
                        "relevance": 0.92,
                        "tool": "vector_search_docs"
                    }
                ],
                "tools_used": ["vector_search_docs"],
                "tool_queries": {
                    "vector_search_docs": [
                        "How to configure OAuth?",
                        "OAuth setup guide"
                    ]
                },
                "metadata": {
                    "router_decision": "sufficient_context (confidence=0.85)",
                    "retrieved_count": 5
                },
                "metrics": {
                    "llm_calls": 4,
                    "llm_total_time": 3.2,
                    "search_calls": 2,
                    "search_total_time": 0.8,
                    "docs_retrieved": 5,
                    "total_time": 4.5
                }
            }
        }
    )


class HealthResponse(BaseModel):
    """
    Health check response model.
    """
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0"
            }
        }
    )
