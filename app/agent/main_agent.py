from typing import Any, Optional
import asyncio
import logging

from app.core.agent_settings import AgentContext, AgentState, AgentSettings
from app.core.llm import BaseLLM
from app.agent.nodes import (
    RouterNode,
    ToolSelectionNode,
    ExpansionNode,
    RetrievalNode,
    AnswerNode,
    GiveInfoNode,
    RerankerNode,
)

logger = logging.getLogger(__name__)


class MultiStepRAGAgent:
    """
    Facade orchestrating the multi-step RAG workflow.
    """

    name: str = "multi_step_rag_agent"

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[Any],
        tool_selection_prompt: str,
        router_prompt: str,
        expansion_prompt: str,
        reranker_prompt: str,
        answer_prompt: str,
        settings: Optional[AgentSettings] = None,
        rerank_top_k: int = 10,
    ):
        """
        Initialize the RAG agent facade.
        """
        self.settings = settings or AgentSettings()
        self.llm = llm

        # Build tools registry for dependency injection into nodes
        self._tools_registry = {tool.name: tool for tool in tools}

        # Initialize specialized nodes with injected dependencies
        self._router = RouterNode(
            llm=llm,
            router_prompt=router_prompt,
            confidence_threshold=self.settings.confidence_threshold,
            confidence_default=self.settings.confidence_default,
            should_answer=self.settings.should_answer,
        )

        self._tool_selector = ToolSelectionNode(
            llm=llm,
            tool_selection_prompt=tool_selection_prompt,
            available_tools=tools,
        )

        self._expander = ExpansionNode(
            llm=llm,
            expansion_prompt=expansion_prompt,
            tools_registry=self._tools_registry,
            expansion_count=self.settings.expansion_count,
            timeout=self.settings.expansion_timeout,
        )

        self._retriever = RetrievalNode(tools_registry=self._tools_registry)

        self._reranker = None  # RerankerNode(
        #     llm=llm,
        #     rerank_prompt=reranker_prompt,
        #     num_docs=rerank_top_k,
        # )

        self._answerer = AnswerNode(llm=llm, answer_prompt=answer_prompt)
        self._info_provider = GiveInfoNode()

    async def execute(
        self,
        query: str,
        chat_history: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """
        Execute the agent workflow for a given query.
        """
        overall_start = asyncio.get_running_loop().time()

        logger.info(
            "[%s] Execute start: query_len=%d, history_turns=%d, raw_search=%s",
            self.name, len(query), len(chat_history or []), self.settings.raw_search
        )

        ctx = AgentContext(
            query=query,
            chat_history=chat_history or [],
            max_iterations=self.settings.max_iterations,
        )

        if self.settings.raw_search:
            logger.info("[%s] Raw search mode enabled", self.name)
            ctx = await self._execute_raw_search(ctx)
            if ctx.current_context:
                next_state = await self._router.process(ctx)
                logger.info(
                    "[%s] Raw search router decision: %s, confidence=%s",
                    self.name, next_state.value, ctx.confidence_score
                )
                if next_state in (AgentState.ANSWERING, AgentState.GIVE_INFO):
                    ctx = await self._handle_terminal_state(ctx, next_state)
                    result = self._build_response(ctx, overall_start)
                    logger.info(
                        "[%s] Execute done (raw_search fast-path): iterations=%d, "
                        "final_confidence=%s, total_time=%.2fs",
                        self.name, result["iterations"],
                        result["final_confidence"],
                        result["metrics"].get("total_time", 0)
                    )
                    return result
            state = AgentState.ROUTING if not ctx.current_context else next_state
        else:
            state = AgentState.ROUTING

        confidence_history: list[dict[str, Any]] = []
        max_loop_guard = (
            self.settings.max_iterations * self.settings.loop_guard_multiplier
            + self.settings.loop_guard_buffer
        )
        loop_counter = 0

        while state != AgentState.DONE:
            loop_counter += 1
            if loop_counter > max_loop_guard:
                ctx.metrics["loop_guard_triggered"] = True
                logger.error(
                    "[%s] Loop guard triggered after %d steps (max=%d, iterations=%d)",
                    self.name, loop_counter, max_loop_guard, ctx.current_iterations
                )
                raise RuntimeError(
                    f"Agent loop exceeded safety limit of {max_loop_guard} iterations"
                )

            ctx.metadata["current_state"] = state.value
            logger.debug(
                "[%s] State=%s, iteration=%d, tool_index=%d, context_docs=%d",
                self.name, state.value, ctx.current_iterations,
                ctx.current_tool_index, len(ctx.current_context)
            )

            state = await self._process_state(ctx, state, confidence_history)

        result = self._build_response(ctx, overall_start)
        logger.info(
            "[%s] Execute done: iterations=%d, final_confidence=%s, total_time=%.2fs",
            self.name, result["iterations"],
            result["final_confidence"],
            result["metrics"].get("total_time", 0)
        )
        return result

    async def _execute_raw_search(self, ctx: AgentContext) -> AgentContext:
        """
        Execute direct search across all tools in parallel without query expansion.
        """
        raw_search_start = asyncio.get_running_loop().time()
        logger.info(
            "[%s] Raw search across %d tools for query_len=%d",
            self.name, len(self._tools_registry), len(ctx.query)
        )

        async def search_with_tool(tool: Any) -> tuple[str, list[dict], Optional[Exception]]:
            try:
                results = await tool.search_async(ctx.query, top_k=5)
                for doc in results:
                    doc["tool"] = tool.name
                    doc["iteration"] = 0
                return tool.name, results, None
            except Exception as e:
                logger.error("[%s] Raw search failed for %s: %s", self.name, tool.name, e)
                return tool.name, [], e

        search_tasks = [search_with_tool(tool) for tool in self._tools_registry.values()]
        results = await asyncio.gather(*search_tasks, return_exceptions=False)

        seen_ids: set[str] = set()
        total_docs = 0
        error_count = 0

        for tool_name, docs, error in results:
            if error:
                error_count += 1
                ctx.metrics["raw_search_errors"] = ctx.metrics.get("raw_search_errors", 0) + 1
                ctx.metadata[f"raw_search_error_{tool_name}"] = str(error)
                continue

            ctx.metrics.setdefault("tool_calls", {})
            ctx.metrics["tool_calls"][tool_name] = (
                ctx.metrics["tool_calls"].get(tool_name, 0) + 1
            )

            new_docs_count = 0
            for doc in docs:
                doc_id = doc.get("id")
                if doc_id and doc_id in seen_ids:
                    continue

                ctx.current_context.append(doc)
                if doc_id:
                    seen_ids.add(doc_id)

                new_docs_count += 1
                total_docs += 1

            ctx.metrics["docs_retrieved"] += new_docs_count
            logger.debug(
                "[%s] Raw search tool %s: %d new documents",
                self.name, tool_name, new_docs_count
            )

        ctx.metrics["raw_search_time"] = (
            asyncio.get_running_loop().time() - raw_search_start
        )

        logger.info(
            "[%s] Raw search completed: %d documents from %d tools in %.2fs (errors: %d)",
            self.name, total_docs, len(self._tools_registry),
            ctx.metrics["raw_search_time"], error_count
        )

        return ctx

    async def _handle_terminal_state(
        self,
        ctx: AgentContext,
        state: AgentState,
    ) -> AgentContext:
        """
        Process terminal states (ANSWERING or GIVE_INFO) with a single rerank before them.
        """
        logger.info(
            "[%s] Handle terminal state: %s, docs=%d, reranker=%s",
            self.name, state.value, len(ctx.current_context),
            'on' if self._reranker else 'off'
        )

        # Single rerank step before final answer / info, if configured
        if self._reranker is not None and ctx.current_context:
            ctx = await self._reranker.process(ctx)

        if state == AgentState.ANSWERING:
            return await self._answerer.process(ctx)
        else:
            return await self._info_provider.process(ctx)

    async def _process_state(
        self,
        ctx: AgentContext,
        state: AgentState,
        confidence_history: list[dict],
    ) -> AgentState:
        """
        Process a single state in the agent workflow.
        """
        if state == AgentState.ROUTING:
            next_state = await self._router.process(ctx)
            confidence_history.append(
                {
                    "iteration": ctx.current_iterations,
                    "confidence": ctx.confidence_score,
                }
            )
            logger.info(
                "[%s] Routing result: next_state=%s, confidence=%s, iteration=%d",
                self.name, next_state.value, ctx.confidence_score, ctx.current_iterations
            )
            if next_state in (AgentState.ANSWERING, AgentState.GIVE_INFO):
                ctx = await self._handle_terminal_state(ctx, next_state)
                return AgentState.DONE
            return next_state

        elif state == AgentState.TOOL_SELECTION:
            ctx = await self._tool_selector.process(ctx)
            if ctx.selected_tools:
                logger.info(
                    "[%s] Tools selected (iteration %d): %s",
                    self.name, ctx.current_iterations, ctx.selected_tools
                )
                ctx.current_tool_index = 0
                return AgentState.TOOL_PROCESSING
            else:
                logger.warning(
                    "[%s] No tools selected (iteration %d)",
                    self.name, ctx.current_iterations
                )
                ctx.metadata["tool_selection_empty"] = True
                ctx.metrics["empty_tool_selections"] = (
                    ctx.metrics.get("empty_tool_selections", 0) + 1
                )
                next_state = (
                    AgentState.QUERY_EXPANSION if ctx.can_search_more()
                    else AgentState.ANSWERING
                )
                if next_state in (AgentState.ANSWERING, AgentState.GIVE_INFO):
                    ctx = await self._handle_terminal_state(ctx, next_state)
                    return AgentState.DONE
                return next_state

        elif state == AgentState.TOOL_PROCESSING:
            if ctx.current_tool_index < len(ctx.selected_tools):
                logger.debug(
                    "[%s] Processing tool %s (%d/%d)",
                    self.name, ctx.selected_tools[ctx.current_tool_index],
                    ctx.current_tool_index + 1, len(ctx.selected_tools)
                )
                return AgentState.QUERY_EXPANSION
            else:
                ctx.current_iterations += 1
                logger.info(
                    "[%s] Completed tool pass, moving to ROUTING, iterations=%d",
                    self.name, ctx.current_iterations
                )
                return AgentState.ROUTING

        elif state == AgentState.QUERY_EXPANSION:
            ctx = await self._expander.process(ctx)
            return AgentState.RETRIEVAL

        elif state == AgentState.RETRIEVAL:
            ctx = await self._retriever.process(ctx)
            ctx.current_tool_index += 1
            return AgentState.TOOL_PROCESSING

        elif state in (AgentState.ANSWERING, AgentState.GIVE_INFO):
            logger.info(
                "[%s] Terminal state reached explicitly: %s", self.name, state.value
            )
            ctx = await self._handle_terminal_state(ctx, state)
            return AgentState.DONE

        else:
            raise RuntimeError(f"Unhandled agent state: {state}")

    def _build_response(
        self,
        ctx: AgentContext,
        overall_start: float,
    ) -> dict[str, Any]:
        """
        Build final response dictionary from agent context.
        """
        if ctx.metrics.get("llm_calls", 0) > 0:
            ctx.metrics["avg_llm_time"] = (
                ctx.metrics["llm_total_time"] / ctx.metrics["llm_calls"]
            )

        if ctx.metrics.get("search_calls", 0) > 0:
            ctx.metrics["avg_search_time"] = (
                ctx.metrics["search_total_time"] / ctx.metrics["search_calls"]
            )

        ctx.metrics["total_time"] = asyncio.get_running_loop().time() - overall_start
        ctx.metrics["total_iterations"] = ctx.current_iterations

        logger.info(
            "[%s] Build response: iterations=%d, docs=%d, tools_used=%s, total_time=%.2fs",
            self.name, ctx.current_iterations, len(ctx.current_context),
            list(ctx.tool_results.keys()), ctx.metrics["total_time"]
        )

        return {
            "answer": ctx.final_answer if ctx.final_answer else "",
            "iterations": ctx.current_iterations,
            "final_confidence": ctx.confidence_score,
            "sources": [
                {
                    "content": d.get("content", ""),
                    "source": d.get("source", "unknown"),
                    "tool": d.get("tool", "unknown"),
                    "page_start": d.get("page_start", 1),
                    "relevance": d.get("score", None),
                }
                for d in ctx.current_context
            ],
            "tools_used": list(ctx.tool_results.keys()),
            "tool_queries": ctx.tool_specific_queries,
            "metadata": ctx.metadata,
            "metrics": ctx.metrics,
        }
