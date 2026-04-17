import asyncio
import logging
from typing import Annotated, Any, Optional

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware

from app.agent.main_agent import MultiStepRAGAgent
from app.tools.vector_search import SearchTool
from app.core.llm import OllamaLLM
from app.core.prompts import (
    ROUTER_PROMPT, 
    EXPANSION_PROMPT, 
    RERANKER_PROMPT,
    ANSWER_PROMPT,
    TOOL_SELECTION_PROMPT,
)
from app.core.agent_settings import AgentSettings
from app.core.consts import OLLAMA_URL, OLLAMA_MODEL, SEARCH_URL, COLLECTIONS, API_KEY
from app.core.trace import log_trace
from app.core.models import AgentConfigRequest, AgentResponse, SourceDocument, HealthResponse

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging.getLogger("app.agent_settings").setLevel(logging.DEBUG)
logging.getLogger("app.tools").setLevel(logging.INFO) 


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializing"""
    logger.info("Initializing RAG Agent service...")
    tools = []
    try:
        for collection in COLLECTIONS:
            tool = SearchTool(
                collection_name=collection,
                max_concurrent=10,
                name=f"vector_search_{collection}",
                description=f"Semantic vector search over '{collection}' document collection."
            )
            await tool.__aenter__()
            tools.append(tool)

        llm = OllamaLLM(
            base_url=OLLAMA_URL,
            model=OLLAMA_MODEL,
            timeout=30.0
        )

        app.state.tools = tools
        app.state.llm = llm
        logger.info("RAG Agent service initialized successfully")
        yield
    finally:
        logger.info("Shutting down RAG Agent service...")
        for tool in tools:
            await tool.__aexit__(None, None, None)
        logger.info("RAG Agent service shut down complete")

app = FastAPI(
    lifespan=lifespan,
    title="MultiStep RAG Agent API",
    description="API for intelligent RAG-based question answering with query expansion",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_tools() -> list[SearchTool]:
    return app.state.tools


def get_llm() -> OllamaLLM:
    return app.state.llm


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Check service health status.
    
    :return: Health status with version information.
    :rtype: HealthResponse
    """
    return HealthResponse(status="healthy", version="1.0.0")


def require_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/agent-query", response_model=AgentResponse, tags=["Agent"], dependencies=[Depends(require_api_key)])
async def process_query(
    request: AgentConfigRequest,
    tools: Annotated[list[SearchTool], Depends(get_tools)],
    llm: Annotated[OllamaLLM, Depends(get_llm)],
) -> AgentResponse:
    """
    Process a user query through the MultiStep RAG Agent.

    :param request: Query request with configuration parameters.
    :return: Agent response with answer, sources, and metadata.
    :raises HTTPException: If agent initialization fails or execution errors occur.
    """
    try:
        logger.info("Processing query: %s...", request.query[:100])

        # Select tools
        if request.tools:
            requested_names = set(request.tools)
            available_names = {t.name for t in tools}

            invalid_tools = requested_names - available_names
            if invalid_tools:
                logger.warning("Invalid tool names requested: %s", invalid_tools)

            selected_tools = [t for t in tools if t.name in requested_names]
            if not selected_tools:
                logger.warning(
                    "No valid tools requested: %s, using all available", request.tools
                )
                selected_tools = tools
        else:
            selected_tools = tools

        logger.info("Using tools: %s", [t.name for t in selected_tools])
        log_trace("user_request", query=request.query, tools=request.tools, history_turns=len(request.chat_history))

        agent = MultiStepRAGAgent(
            llm=llm,
            tools=selected_tools,
            tool_selection_prompt=TOOL_SELECTION_PROMPT,
            router_prompt=ROUTER_PROMPT,
            expansion_prompt=EXPANSION_PROMPT,
            reranker_prompt=RERANKER_PROMPT,
            answer_prompt=ANSWER_PROMPT,
            settings=AgentSettings(
                max_iterations=request.max_iterations,
                expansion_count=request.expansion_count,
                confidence_threshold=request.confidence_threshold,
                should_answer=request.should_answer,
                raw_search=request.raw_search,
            ),
        )

        result = await agent.execute(
            query=request.query,
            chat_history=request.chat_history,
        )
        log_trace("agent_response",
            query=request.query,
            answer=result.get("answer", ""),
            sources=[s["source"] for s in result.get("sources", [])],
            iterations=result.get("iterations"),
            confidence=result.get("final_confidence"),
        )

        logger.info(
            "Query completed: iterations=%s, confidence=%s",
            result.get("iterations"),
            result.get("final_confidence"),
        )

        sources: list[SourceDocument] = []
        for src in result.get("sources", []):
            src_payload: dict[str, Any] = {
                "content": src["content"],
                "source": src["source"],
                "tool": src.get("tool"),
                "relevance": src.get("relevance"),
            }
            if "page_start" in src and src["page_start"] is not None:
                src_payload["page_start"] = src["page_start"]

            sources.append(SourceDocument(**src_payload))

        response = AgentResponse(
            answer=result.get("answer", ""),
            iterations=result.get("iterations", 0),
            final_confidence=result.get("final_confidence"),
            sources=sources,
            metadata=result.get("metadata", {}),
            metrics=result.get("metrics", {}),
            tools_used=result.get("tools_used", []),
            tool_queries=result.get("tool_queries", {}),
        )

        return response

    except asyncio.TimeoutError as e:
        logger.error("Query timeout: %s", e)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Query processing timed out. Please try again.",
        )
    except Exception as e:
        logger.error("Query processing error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during query processing.",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8050,
        reload=True,
        log_level="info"
    )
