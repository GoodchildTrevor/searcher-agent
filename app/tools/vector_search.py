from typing import Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
import asyncio
import logging
import aiohttp

from app.core.consts import SEARCH_URL

logger = logging.getLogger(__name__)


@runtime_checkable
class SearchToolProtocol(Protocol):
    """
    Protocol for search/retrieval tool compatibility.
    
    Any tool implementing this protocol can be used by the RAG agent
    to retrieve documents based on text queries.
    """
    
    name: str
    """Unique identifier for the tool."""
    
    description: str
    """Human-readable description of the tool's purpose and capabilities."""
    
    async def search_async(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        Search for documents given a query.
        
        :param query: Search query string.
        :param method: Search method to use (e.g., "hybrid", "vector", "keyword").
        :param top_k: Maximum number of results to return.
        :return: List of formatted document dictionaries.
        """
        ...
    
    async def batch_search_async(
        self,
        queries: list[str],
        method: str = "hybrid",
        top_k: int = 3
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Batch search across multiple queries.
        
        :param queries: List of search queries.
        :param method: Search method to use.
        :param top_k: Maximum number of results per query.
        :return: Dictionary mapping queries to document lists.
        """
        ...


@dataclass
class SearchConfig:
    """
    Configuration for the search tool.
    """
    
    base_url: str = SEARCH_URL
    """Base URL of the search API endpoint."""
    
    default_method: str = "hybrid"
    """Default search method to use."""
    
    default_collection: str = "docs"
    """Default collection name to search."""
    
    timeout: int = 15
    """Request timeout in seconds."""
    
    max_retries: int = 2
    """Number of retry attempts on failure."""


class SearchTool:
    """
    HTTP-based search tool implementation for the RAG agent.
    
    This tool communicates with an external search API via HTTP POST requests
    and supports both single and batch query execution with concurrency control.
    
    Must be used as an async context manager (or have __aenter__/__aexit__ called
    explicitly) so that the shared aiohttp.ClientSession is properly created and
    closed.
    
    :cvar name: Tool identifier used by the agent for selection.
    :cvar description: Human-readable description for LLM-based tool selection.
    """
    
    name: str = "vector_search"
    description: str = (
        "Semantic vector search over document collection. "
        "Use for finding relevant documents by meaning, not just keywords. "
        "Supports hybrid search combining vector and keyword matching."
    )
    
    def __init__(
        self,
        collection_name: str,
        config: Optional[SearchConfig] = None,
        max_concurrent: int = 10,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Initialize the search tool.
        :param collection_name: Name of qdrant's collection
        :type_ collection_name: str
        :param config: Search configuration object.
        :type config: Optional[SearchConfig]
        :param max_concurrent: Maximum number of concurrent requests in batch search.
        :type max_concurrent: int
        :param name: Optional override for tool name (for multiple instances).
        :type name: Optional[str]
        :param description: Optional override for tool description.
        :type description: Optional[str]
        """
        self.collection_name = collection_name
        self.config = config or SearchConfig()
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent)
        
        if name:
            self.name = name
        if description:
            self.description = description
        
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "SearchTool":
        """
        Async context manager entry to create session.
        
        :return: Self instance.
        :rtype: SearchTool
        """
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any]
    ) -> None:
        """
        Close session on context exit.
        
        :param exc_type: Exception type if raised.
        :param exc_val: Exception value if raised.
        :param exc_tb: Exception traceback if raised.
        """
        if self._session:
            await self._session.close()
            self._session = None

    def _format_document(self, doc: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Format a document to a unified structure for the agent.
        
        :param doc: Raw document from the search engine.
        :type doc: dict[str, Any]
        :return: Formatted document dictionary or None if invalid.
        :rtype: Optional[dict[str, Any]]
        """
        try:
            payload = doc.get("payload", {})
            text = payload.get("document", "").strip()
            source = payload.get("name", "unknown")
            score = doc.get("score", 0)
            page_start = payload.get("page_start", 1)

            if not text:
                return None

            return {
                "content": text,
                "source": source,
                "score": score,
                "page_start": page_start,
                "id": doc.get("id", str(hash(text)))
            }
        except Exception as e:
            logger.error(f"Error formatting document: {e}")
            return None

    async def search_async(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Asynchronous document search (protocol implementation).

        Uses the shared aiohttp.ClientSession created in __aenter__.
        Falls back to a temporary session if called outside context manager.
        
        :param query: Search query string.
        :type query: str
        :param method: Search method to use.
        :type method: str
        :param top_k: Maximum number of results to return.
        :type top_k: int
        :return: List of formatted document dictionaries.
        :rtype: list[dict[str, Any]]
        """
        payload = {
            "text": query,
            "method": method,
            "collection_name": self.collection_name,
            "limit": top_k
        }

        # Use shared session when available; create a temporary one otherwise.
        owned_session: Optional[aiohttp.ClientSession] = None
        if self._session is None:
            logger.warning(
                "search_async called outside context manager — creating a temporary session. "
                "Prefer using SearchTool as an async context manager."
            )
            owned_session = aiohttp.ClientSession()

        session = owned_session or self._session

        try:
            for attempt in range(self.config.max_retries + 1):
                try:
                    async with session.post(  # type: ignore[union-attr]
                        self.config.base_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()

                        documents = data.get("documents", [])

                        if not documents:
                            logger.warning(f"No results found for query: '{query}'")
                            return []

                        formatted_docs = [
                            fmt
                            for doc in documents
                            if (fmt := self._format_document(doc)) is not None
                        ]

                        return formatted_docs[:top_k]

                except asyncio.TimeoutError:
                    logger.error(f"Search timeout (attempt {attempt + 1}/{self.config.max_retries + 1})")
                    if attempt == self.config.max_retries:
                        raise
                    await asyncio.sleep(1)

                except aiohttp.ClientError as e:
                    logger.error(f"Search error: {e}")
                    if attempt == self.config.max_retries:
                        raise
                    await asyncio.sleep(1)
        finally:
            if owned_session is not None:
                await owned_session.close()

        return []

    async def batch_search_async(
        self,
        queries: list[str],
        method: str = "hybrid",
        top_k: int = 3
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Batch search across multiple queries with concurrency control.
        
        :param queries: List of search queries.
        :type queries: list[str]
        :param method: Search method to use.
        :type method: str
        :param top_k: Maximum number of results per query.
        :type top_k: int
        :return: Dictionary mapping queries to document lists.
        :rtype: dict[str, list[dict[str, Any]]]
        """

        async def bounded_search(query: str) -> list[dict[str, Any]]:
            async with self._semaphore:
                return await self.search_async(query, method, top_k)

        tasks = [bounded_search(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_dict = {}
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                logger.error(f"Search error for query '{query}': {result}")
                result_dict[query] = []
            else:
                result_dict[query] = result

        return result_dict


class BaseTool:
    """
    Base class for all agent tools.
    
    Provides common attributes and methods that all tools should implement.
    New tool types should inherit from this class or implement ToolProtocol directly.
    """
    
    name: str = "base_tool"
    description: str = "Base tool implementation."
    
    def __init__(self, name: Optional[str] = None, description: Optional[str] = None):
        """
        Initialize base tool.
        
        :param name: Optional tool name override.
        :param description: Optional description override.
        """
        if name:
            self.name = name
        if description:
            self.description = description


class KeywordSearchTool(BaseTool):
    """
    Example keyword-based search tool.
    
    This is a template for adding a second tool type in the future.
    Implement actual search logic in search_async and batch_search_async.
    """
    
    name: str = "keyword_search"
    description: str = (
        "Exact keyword matching search. "
        "Use for finding documents containing specific terms or phrases. "
        "Does not perform semantic matching."
    )
    
    def __init__(
        self,
        index_path: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Initialize keyword search tool.
        
        :param index_path: Path to the keyword index.
        :param name: Optional name override.
        :param description: Optional description override.
        """
        super().__init__(name, description)
        self.index_path = index_path
    
    async def search_async(
        self,
        query: str,
        method: str = "exact",
        top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        Keyword-based document search.
        
        :param query: Search query string.
        :param method: Search method (ignored for keyword search).
        :param collection_name: Collection name (ignored for keyword search).
        :param top_k: Maximum results to return.
        :return: List of matching documents.
        TODO: Implement actual keyword search logic
        """
        logger.info(f"Keyword search for: {query}")
        return []
    
    async def batch_search_async(
        self,
        queries: list[str],
        method: str = "exact",
        top_k: int = 3
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Batch keyword search.
        
        :param queries: List of queries.
        :param method: Search method.
        :param collection_name: Collection name.
        :param top_k: Results per query.
        :return: Query-to-results mapping.
        """
        results = {}
        for query in queries:
            results[query] = await self.search_async(query, method, top_k)
        return results
