
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Optional, Protocol


@dataclass
class AgentSettings:
    """Configuration settings for the RAG agent"""

    max_iterations: int = 3
    expansion_count: int = 3
    confidence_default: float = 0.5
    confidence_threshold: float = 0.7
    should_answer: bool = True
    raw_search: bool = False
    expansion_timeout: float = 30.0
    loop_guard_multiplier: int = 5
    loop_guard_buffer: int = 10

class AgentState(Enum):
    """Possible states of the RAG agent."""
    
    ROUTING = "routing"
    """Agent evaluates context sufficiency."""
    
    TOOL_SELECTION = "tool_selection"
    """Agent selects tools based on query and context."""
    
    TOOL_PROCESSING = "tool_processing"
    """Agent iterates through selected tools."""
    
    QUERY_EXPANSION = "query_expansion"
    """Agent generates query variations for current tool."""
    
    RETRIEVAL = "retrieval"
    """Agent retrieves documents using current tool."""

    RERANK = "rerank"
    
    ANSWERING = "answering"
    """Agent generates final answer."""
    
    GIVE_INFO = "give_info"
    """Agent returns context info without answering."""
    
    DONE = "done"
    """Agent execution completed."""


@dataclass
class AgentContext:
    """
    State object that holds all data throughout the agent execution cycle.
    
    Note: This is a mutable dataclass. State is modified in place during
    agent execution for performance reasons.
    """
    
    query: str
    """User's original query."""
    
    chat_history: list[dict[str, str]] = field(default_factory=list)
    """Conversation history for context awareness."""
    
    current_context: list[dict[str, Any]] = field(default_factory=list)
    """Accumulated documents from all tools."""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """Execution metadata and logs."""
    
    selected_tools: list[str] = field(default_factory=list)
    """List of selected tool names."""
    
    tool_specific_queries: dict[str, list[str]] = field(default_factory=dict)
    """Mapping of tool names to their specific queries."""
    
    tool_results: dict[str, list[dict]] = field(default_factory=dict)
    """Raw results from each tool."""
    
    current_tool_index: int = 0
    """Index of currently processing tool."""
    
    confidence_score: Optional[float] = None
    """Current confidence score in context sufficiency."""
    
    current_iterations: int = 0
    """Number of completed search iterations."""
    
    max_iterations: int = 3
    """Maximum allowed search iterations."""
    
    final_answer: Optional[str] = None
    """Final generated answer."""

    metrics: dict[str, Any] = field(
        default_factory=lambda: {
            "llm_calls": 0,
            "llm_total_time": 0.0,
            "search_calls": 0,
            "search_total_time": 0.0,
            "docs_retrieved": 0,
            "tool_calls": {},
        }
    )
    
    def can_search_more(self) -> bool:
        """
        Check if the agent is allowed to perform another retrieval cycle.
        
        :return: True if iteration limit not reached.
        :rtype: bool
        """
        return self.current_iterations < self.max_iterations


class AgentNode(Protocol):
    """
    Protocol defining the interface for agent processing nodes.
    
    Each node represents a single responsibility in the agent workflow
    and operates on the shared AgentContext.
    """
    
    async def process(self, ctx: AgentContext) -> AgentContext | AgentState:
        """
        Process the context and return updated context or next state.
        
        :param ctx: Current agent context.
        :return: Updated context or next agent state.
        """
        ...


class BaseNode(ABC):
    """
    Base class for agent nodes with common utilities.
    
    Provides JSON extraction, logging helpers, and metrics tracking.
    """
    
    def __init__(self, llm: Any, name: str):
        """
        Initialize base node.
        
        :param llm: LLM provider implementing BaseLLM.
        :param name: Human-readable node name for logging.
        """
        self.llm = llm
        self.name = name
    
    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON from text, handling markdown wrappers and comments.
        
        :param text: Input text potentially containing JSON.
        :return: Cleaned JSON string or None if not found.
        """
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            text = match.group(1).strip()
        
        text = re.sub(r'(\}|\])(\s*#.*)$', r'\1', text.strip(), flags=re.MULTILINE)
        
        text = text.strip()
        if (text.startswith('[') and text.endswith(']')) or \
           (text.startswith('{') and text.endswith('}')):
            return text
        
        return None
    
    def _track_llm_metric(self, ctx: AgentContext, elapsed: float, call_type: str):
        """
        Update LLM-related metrics in context.
        
        :param ctx: Agent context to update.
        :param elapsed: Time taken for the LLM call.
        :param call_type: Type of call for metric key (e.g., 'router', 'expansion').
        """
        ctx.metrics["llm_calls"] += 1
        ctx.metrics["llm_total_time"] += elapsed
        metric_key = f"llm_{call_type}_calls"
        ctx.metrics[metric_key] = ctx.metrics.get(metric_key, 0) + 1
