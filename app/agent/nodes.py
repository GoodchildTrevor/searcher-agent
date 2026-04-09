import asyncio
import json
import logging
import re
from typing import Any, Optional

from app.core.agent_settings import AgentContext, AgentState, AgentNode, BaseNode

logger = logging.getLogger(__name__)


class RouterNode(BaseNode):
    """
    Node responsible for evaluating context sufficiency.
    Uses binary decision (answer_found: true/false) rather than
    a float confidence score, which is more reliable on small models.
    When answer_found is True, additionally calls evaluate_confidence_async
    on the LLM (if supported) to get a real float score.
    """

    def __init__(
        self,
        llm: Any,
        router_prompt: str,
        confidence_threshold: float = 0.5,
        confidence_default: float = 0.0,
        should_answer: bool = True
    ):
        super().__init__(llm, "router")
        self.router_prompt = router_prompt
        self.confidence_threshold = confidence_threshold
        self.confidence_default = confidence_default
        self.should_answer = should_answer

    async def process(self, ctx: AgentContext) -> AgentState:
        context_text = "\n".join(d.get("content", "") for d in ctx.current_context)

        if not context_text.strip():
            ctx.confidence_score = 0.0
            ctx.metadata["router_decision"] = "no_context"
            ctx.metrics["routing_decisions"] = ctx.metrics.get("routing_decisions", 0) + 1
            logger.info("[%s] No context available, routing → TOOL_SELECTION", self.name)
            return AgentState.TOOL_SELECTION

        prompt = self.router_prompt.format(
            query=ctx.query,
            context=context_text[:3000]
        )

        logger.debug("[%s] Prompt: %s...", self.name, prompt[:500])
        logger.info(
            "[%s] Request: query_len=%d, context_docs=%d, context_len=%d",
            self.name, len(ctx.query), len(ctx.current_context), len(context_text)
        )

        answer_found = False
        start_time = asyncio.get_running_loop().time()

        try:
            response = await self.llm.generate_async(
                prompt,
                options={"temperature": 0.1, "num_predict": 50}
            )

            elapsed = asyncio.get_running_loop().time() - start_time
            self._track_llm_metric(ctx, elapsed, "router")

            logger.info("[%s] Response: len=%d, time=%.2fs", self.name, len(response), elapsed)
            logger.debug("[%s] Raw response: %s", self.name, response.strip())

            try:
                json_str = self._extract_json(response) or response.strip()
                parsed = json.loads(json_str)
                answer_found = bool(parsed.get("answer_found", False))
                ctx.metrics["router_successful_parses"] = (
                    ctx.metrics.get("router_successful_parses", 0) + 1
                )
                logger.info("[%s] Parsed answer_found=%s", self.name, answer_found)

            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                logger.warning(
                    "[%s] Parse error, treating as answer_found=False: %s. Raw: %s",
                    self.name, e, response.strip()[:100]
                )
                answer_found = False
                ctx.metrics["router_parse_errors"] = (
                    ctx.metrics.get("router_parse_errors", 0) + 1
                )

        except Exception as e:
            elapsed = asyncio.get_running_loop().time() - start_time
            logger.error("[%s] LLM error after %.2fs: %s", self.name, elapsed, e)
            ctx.metadata["router_error"] = str(e)
            ctx.metrics["router_llm_errors"] = (
                ctx.metrics.get("router_llm_errors", 0) + 1
            )
            answer_found = False

        # When answer is found, get a real float confidence via evaluate_confidence_async.
        # Fall back to 1.0/0.0 for LLMs that don't implement the method.
        if answer_found:
            try:
                ctx.confidence_score = await self.llm.evaluate_confidence_async(
                    context=context_text[:2000],
                    query=ctx.query,
                )
                logger.info(
                    "[%s] Float confidence score: %.3f", self.name, ctx.confidence_score
                )
            except AttributeError:
                ctx.confidence_score = 1.0
                logger.debug(
                    "[%s] LLM does not support evaluate_confidence_async, using 1.0",
                    self.name
                )
            except Exception as e:
                ctx.confidence_score = 1.0
                logger.warning(
                    "[%s] evaluate_confidence_async failed (%s), using 1.0", self.name, e
                )
        else:
            ctx.confidence_score = 0.0

        ctx.metadata["router_raw_response"] = (
            response if "response" in locals() else ""
        )
        ctx.metrics["routing_decisions"] = (
            ctx.metrics.get("routing_decisions", 0) + 1
        )

        if answer_found:
            if self.should_answer:
                ctx.metadata["router_decision"] = "sufficient_context → ANSWERING"
                ctx.metrics["high_confidence_routes"] = (
                    ctx.metrics.get("high_confidence_routes", 0) + 1
                )
                logger.info("[%s] Decision: sufficient_context → ANSWERING", self.name)
                return AgentState.ANSWERING
            else:
                ctx.metadata["router_decision"] = "sufficient_context → GIVE_INFO"
                ctx.metrics["info_only_routes"] = (
                    ctx.metrics.get("info_only_routes", 0) + 1
                )
                logger.info("[%s] Decision: sufficient_context → GIVE_INFO", self.name)
                return AgentState.GIVE_INFO

        if ctx.can_search_more():
            ctx.metadata["router_decision"] = (
                f"need_more_info (iteration {ctx.current_iterations}) → TOOL_SELECTION"
            )
            ctx.metrics["search_routes"] = ctx.metrics.get("search_routes", 0) + 1
            logger.info(
                "[%s] Decision: need_more_info (iteration %d/%d) → TOOL_SELECTION",
                self.name, ctx.current_iterations, ctx.max_iterations
            )
            return AgentState.TOOL_SELECTION

        ctx.metadata["router_decision"] = (
            f"max_iterations_reached ({ctx.current_iterations}) → ANSWERING"
        )
        ctx.metrics["forced_answer_routes"] = (
            ctx.metrics.get("forced_answer_routes", 0) + 1
        )
        logger.warning(
            "[%s] Decision: max_iterations_reached (%d/%d) → ANSWERING (forced)",
            self.name, ctx.current_iterations, ctx.max_iterations
        )
        return AgentState.ANSWERING


class ToolSelectionNode(BaseNode):
    """
    Node responsible for selecting appropriate search tools.
    """

    def __init__(
        self,
        llm: Any,
        tool_selection_prompt: str,
        available_tools: list[Any]
    ):
        super().__init__(llm, "tool_selection")
        self.tool_selection_prompt = tool_selection_prompt
        self.available_tools = {tool.name: tool for tool in available_tools}

    async def process(self, ctx: AgentContext) -> AgentContext:
        context_text = "\n".join(d.get("content", "") for d in ctx.current_context)
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in ctx.chat_history)

        tools_desc = "\n".join(
            f"- {name}: {tool.description}"
            for name, tool in self.available_tools.items()
        )

        prompt = self.tool_selection_prompt.format(
            query=ctx.query,
            context=context_text[:2000],
            history=history_text[-500:],
            tools=tools_desc,
        )

        logger.debug("[%s] Prompt: %s...", self.name, prompt[:500])
        logger.info(
            "[%s] Request: query_len=%d, available_tools=%s, context_docs=%d",
            self.name, len(ctx.query), list(self.available_tools.keys()), len(ctx.current_context)
        )

        start_time = asyncio.get_running_loop().time()
        try:
            response = await self.llm.generate_async(
                prompt,
                options={"temperature": 0.1, "num_predict": 100}
            )

            elapsed = asyncio.get_running_loop().time() - start_time
            self._track_llm_metric(ctx, elapsed, "tool_selection")
            logger.info("[%s] Response: len=%d, time=%.2fs", self.name, len(response), elapsed)
            logger.debug("[%s] Raw response: %s", self.name, response.strip())

            json_str = self._extract_json(response)
            if json_str:
                try:
                    tools_list = json.loads(json_str)
                    if isinstance(tools_list, list):
                        ctx.selected_tools = [
                            t for t in tools_list
                            if t in self.available_tools
                        ]
                        unknown = [t for t in tools_list if t not in self.available_tools]
                        if unknown:
                            logger.warning(
                                "[%s] LLM requested unknown tools, skipping: %s",
                                self.name, unknown
                            )
                        ctx.metrics["tool_selection_success"] = ctx.metrics.get("tool_selection_success", 0) + 1
                        ctx.metrics["tools_selected_count"] = ctx.metrics.get("tools_selected_count", 0) + len(ctx.selected_tools)
                        logger.info("[%s] Selected tools: %s", self.name, ctx.selected_tools)
                    else:
                        ctx.selected_tools = []
                        ctx.metrics["tool_selection_invalid_format"] = ctx.metrics.get("tool_selection_invalid_format", 0) + 1
                        logger.warning("[%s] Invalid format (expected list), no tools selected", self.name)
                except json.JSONDecodeError as e:
                    ctx.selected_tools = []
                    ctx.metrics["tool_selection_json_error"] = ctx.metrics.get("tool_selection_json_error", 0) + 1
                    logger.warning("[%s] JSON decode error: %s. Raw: %s", self.name, e, json_str[:100])
            else:
                ctx.selected_tools = []
                ctx.metrics["tool_selection_no_json"] = ctx.metrics.get("tool_selection_no_json", 0) + 1
                logger.warning("[%s] No JSON found in response, no tools selected", self.name)

            ctx.metadata["selected_tools"] = ctx.selected_tools
            ctx.current_tool_index = 0

        except Exception as e:
            elapsed = asyncio.get_running_loop().time() - start_time
            logger.error("[%s] Error after %.2fs: %s", self.name, elapsed, e)
            ctx.selected_tools = []
            ctx.metadata["tool_selection_error"] = str(e)
            ctx.metrics["tool_selection_llm_errors"] = ctx.metrics.get("tool_selection_llm_errors", 0) + 1

        return ctx


class ExpansionNode(BaseNode):
    """
    Node responsible for generating query variations.
    Requires tools_registry to pass the real tool description into the prompt.
    """

    def __init__(
        self,
        llm: Any,
        expansion_prompt: str,
        tools_registry: dict[str, Any],
        expansion_count: int = 3,
        timeout: float = 30.0
    ):
        super().__init__(llm, "expansion")
        self.expansion_prompt = expansion_prompt
        self.tools_registry = tools_registry
        self.expansion_count = expansion_count
        self.timeout = timeout

    async def process(self, ctx: AgentContext) -> AgentContext:
        if ctx.current_tool_index >= len(ctx.selected_tools):
            logger.warning(
                "[%s] current_tool_index=%d >= selected_tools=%d, skipping expansion",
                self.name, ctx.current_tool_index, len(ctx.selected_tools)
            )
            return ctx

        current_tool_name = ctx.selected_tools[ctx.current_tool_index]
        current_tool = self.tools_registry.get(current_tool_name)
        tool_description = (
            getattr(current_tool, "description", None) or "Search tool"
        )

        logger.info(
            "[%s] Expanding query for tool '%s' (description: %s) (tool %d/%d), count=%d",
            self.name, current_tool_name, tool_description,
            ctx.current_tool_index + 1, len(ctx.selected_tools),
            self.expansion_count
        )

        previous_queries = []
        for queries in ctx.tool_specific_queries.values():
            previous_queries.extend(queries)
        previous_queries_str = ", ".join(f'"{q}"' for q in previous_queries[-10:])

        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in ctx.chat_history[-5:])

        start_time = asyncio.get_running_loop().time()
        try:
            response = await asyncio.wait_for(
                self.llm.generate_async(
                    self.expansion_prompt.format(
                        count=self.expansion_count,
                        query=ctx.query,
                        tool_name=current_tool_name,
                        tool_description=tool_description,
                        previous_queries=previous_queries_str,
                        history=history_text if history_text else "History empty.",
                        iteration=ctx.current_iterations,
                        max_iterations=ctx.max_iterations
                    ),
                    options={"temperature": 0.8, "top_p": 0.9, "num_predict": 300}
                ),
                timeout=self.timeout
            )

            elapsed = asyncio.get_running_loop().time() - start_time
            self._track_llm_metric(ctx, elapsed, "expansion")
            logger.info("[%s] Response: len=%d, time=%.2fs", self.name, len(response), elapsed)
            logger.debug("[%s] Raw response: %s", self.name, response.strip())

            json_str = self._extract_json(response)
            if json_str:
                try:
                    queries = json.loads(json_str)
                    if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                        if ctx.query not in queries:
                            queries.insert(0, ctx.query)
                        ctx.tool_specific_queries[current_tool_name] = queries[:self.expansion_count + 1]
                        ctx.metrics["expansion_success"] = ctx.metrics.get("expansion_success", 0) + 1
                        ctx.metrics["queries_generated"] = ctx.metrics.get("queries_generated", 0) + len(queries)
                        logger.info(
                            "[%s] Generated %d queries for '%s': %s",
                            self.name, len(ctx.tool_specific_queries[current_tool_name]),
                            current_tool_name, ctx.tool_specific_queries[current_tool_name]
                        )
                    else:
                        ctx.tool_specific_queries[current_tool_name] = [ctx.query]
                        ctx.metrics["expansion_invalid_format"] = ctx.metrics.get("expansion_invalid_format", 0) + 1
                        logger.warning(
                            "[%s] Invalid expansion format for '%s', falling back to original query",
                            self.name, current_tool_name
                        )
                except json.JSONDecodeError as e:
                    ctx.tool_specific_queries[current_tool_name] = [ctx.query]
                    ctx.metrics["expansion_json_error"] = ctx.metrics.get("expansion_json_error", 0) + 1
                    logger.warning(
                        "[%s] JSON decode error for '%s': %s, falling back to original query",
                        self.name, current_tool_name, e
                    )
            else:
                ctx.tool_specific_queries[current_tool_name] = [ctx.query]
                ctx.metrics["expansion_no_json"] = ctx.metrics.get("expansion_no_json", 0) + 1
                logger.warning(
                    "[%s] No JSON found in response for '%s', falling back to original query",
                    self.name, current_tool_name
                )

        except asyncio.TimeoutError:
            elapsed = asyncio.get_running_loop().time() - start_time
            logger.error("[%s] Timeout for %s after %.2fs", self.name, current_tool_name, elapsed)
            ctx.tool_specific_queries[current_tool_name] = [ctx.query]
            ctx.metadata[f"expansion_timeout_{current_tool_name}"] = True
            ctx.metrics["expansion_timeouts"] = ctx.metrics.get("expansion_timeouts", 0) + 1

        except Exception as e:
            elapsed = asyncio.get_running_loop().time() - start_time
            logger.error("[%s] Error for %s after %.2fs: %s", self.name, current_tool_name, elapsed, e)
            ctx.tool_specific_queries[current_tool_name] = [ctx.query]
            ctx.metadata[f"expansion_error_{current_tool_name}"] = str(e)
            ctx.metrics["expansion_errors"] = ctx.metrics.get("expansion_errors", 0) + 1

        return ctx


class RetrievalNode:
    """
    Node responsible for executing document retrieval.
    """

    def __init__(self, tools_registry: dict[str, Any]):
        self.tools_registry = tools_registry

    async def process(self, ctx: AgentContext) -> AgentContext:
        if ctx.current_tool_index >= len(ctx.selected_tools):
            logger.warning(
                "[retrieval] current_tool_index=%d >= selected_tools=%d, skipping retrieval",
                ctx.current_tool_index, len(ctx.selected_tools)
            )
            return ctx

        current_tool_name = ctx.selected_tools[ctx.current_tool_index]
        current_tool = self.tools_registry.get(current_tool_name)

        if not current_tool:
            logger.error("Tool %s not found in registry", current_tool_name)
            ctx.metrics["retrieval_tool_not_found"] = ctx.metrics.get("retrieval_tool_not_found", 0) + 1
            return ctx

        queries = ctx.tool_specific_queries.get(current_tool_name, [ctx.query])
        logger.info(
            "[retrieval] Starting retrieval with tool '%s', queries=%d, top_k=5",
            current_tool_name, len(queries)
        )
        logger.debug("[retrieval] Queries: %s", queries)

        start_time = asyncio.get_running_loop().time()
        try:
            results = await current_tool.batch_search_async(queries=queries, top_k=5)
            elapsed = asyncio.get_running_loop().time() - start_time
            logger.info(
                "[retrieval] Tool '%s' returned results for %d queries in %.2fs",
                current_tool_name, len(results), elapsed
            )

            ctx.metrics["search_calls"] += len(queries)
            ctx.metrics["search_total_time"] += elapsed

            tool_metrics = ctx.metrics.setdefault("tool_performance", {}).setdefault(
                current_tool_name, {"calls": 0, "total_time": 0.0, "docs_retrieved": 0}
            )
            tool_metrics["calls"] += len(queries)
            tool_metrics["total_time"] += elapsed

            ctx.tool_results[current_tool_name] = []
            for docs in results.values():
                ctx.tool_results[current_tool_name].extend(docs)

            seen_ids = {d.get("id") for d in ctx.current_context if d.get("id")}
            new_docs_count = 0

            for doc in ctx.tool_results[current_tool_name]:
                doc_id = doc.get("id")
                doc_copy = doc.copy()
                doc_copy["tool"] = current_tool_name

                if doc_id and doc_id in seen_ids:
                    logger.debug("[retrieval] Skipping duplicate doc id=%s", doc_id)
                    continue

                ctx.current_context.append(doc_copy)
                if doc_id:
                    seen_ids.add(doc_id)

                ctx.metrics["docs_retrieved"] += 1
                new_docs_count += 1
                tool_metrics["docs_retrieved"] += 1

            ctx.metadata[f"retrieved_count_{current_tool_name}"] = len(ctx.tool_results[current_tool_name])
            ctx.metadata[f"new_docs_{current_tool_name}"] = new_docs_count
            ctx.metrics["retrieval_success"] = ctx.metrics.get("retrieval_success", 0) + 1

            logger.info(
                "[retrieval] Tool '%s': total_raw=%d, new_unique=%d, total_context=%d",
                current_tool_name,
                len(ctx.tool_results[current_tool_name]),
                new_docs_count,
                len(ctx.current_context)
            )

        except Exception as e:
            elapsed = asyncio.get_running_loop().time() - start_time
            logger.error("Retrieval error for %s after %.2fs: %s", current_tool_name, elapsed, e)
            ctx.metadata[f"retrieval_error_{current_tool_name}"] = str(e)
            ctx.metrics["retrieval_errors"] = ctx.metrics.get("retrieval_errors", 0) + 1

        ctx.metadata["total_retrieved_count"] = len(ctx.current_context)
        return ctx


class AnswerNode(BaseNode):
    """
    Node responsible for generating the final answer.
    """

    def __init__(self, llm: Any, answer_prompt: str):
        super().__init__(llm, "answer")
        self.answer_prompt = answer_prompt

    async def process(self, ctx: AgentContext) -> AgentContext:
        context_parts = []
        for d in ctx.current_context:
            source = d.get("source", "unknown")
            tool = d.get("tool", "unknown")
            content = d.get("content", "")
            context_parts.append(f"[{source} ({tool})] {content}")

        context_text = "\n\n".join(context_parts)
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in ctx.chat_history)

        if not context_text:
            logger.warning("[%s] Answering with empty context", self.name)

        prompt = self.answer_prompt.format(
            context=context_text if context_text else "Context not found.",
            history=history_text if history_text else "Dialog history is empty.",
            query=ctx.query,
            confidence_score=ctx.confidence_score if ctx.confidence_score is not None else "not evaluated"
        )

        logger.debug("[%s] Prompt: %s...", self.name, prompt[:500])
        logger.info(
            "[%s] Request: context_docs=%d, query_len=%d, confidence_score=%s",
            self.name, len(ctx.current_context), len(ctx.query), ctx.confidence_score
        )

        start_time = asyncio.get_running_loop().time()
        try:
            ctx.final_answer = await self.llm.generate_async(prompt, options={"temperature": 0.4})

            elapsed = asyncio.get_running_loop().time() - start_time
            self._track_llm_metric(ctx, elapsed, "answer")

            answer_len = len(ctx.final_answer) if ctx.final_answer else 0
            logger.info("[%s] Response: len=%d, time=%.2fs", self.name, answer_len, elapsed)

            if answer_len < 50:
                logger.warning("[%s] Unusually short answer (%d chars)", self.name, answer_len)
                ctx.metrics["short_answers"] = ctx.metrics.get("short_answers", 0) + 1
            elif answer_len > 5000:
                logger.warning("[%s] Unusually long answer (%d chars)", self.name, answer_len)
                ctx.metrics["long_answers"] = ctx.metrics.get("long_answers", 0) + 1

            ctx.metadata["answer_generated"] = True
            ctx.metrics["answer_success"] = ctx.metrics.get("answer_success", 0) + 1

        except Exception as e:
            elapsed = asyncio.get_running_loop().time() - start_time
            logger.error("[%s] Error after %.2fs: %s", self.name, elapsed, e)
            ctx.final_answer = "Sorry, an error occurred while generating the answer."
            ctx.metadata["answer_error"] = str(e)
            ctx.metrics["answer_errors"] = ctx.metrics.get("answer_errors", 0) + 1

        return ctx


class GiveInfoNode:
    """
    Node for returning context info without generating an answer.
    """

    async def process(self, ctx: AgentContext) -> AgentContext:
        logger.info(
            "[give_info] Returning %d docs without answer generation (info_only mode)",
            len(ctx.current_context)
        )
        ctx.metadata["info_only"] = True
        ctx.metadata["answer_generated"] = False
        ctx.metrics["info_only_responses"] = ctx.metrics.get("info_only_responses", 0) + 1
        return ctx


class RerankerNode(BaseNode):
    """
    Node responsible for reranking retrieved documents by relevance.
    Inherits BaseNode to reuse _track_llm_metric and _extract_json (#17).
    """

    def __init__(self, llm: Any, rerank_prompt: str, num_docs: int = 10):
        super().__init__(llm, "reranker")
        self.rerank_prompt = rerank_prompt
        self.num_docs = num_docs

    @staticmethod
    def _parse_indices(response: str) -> list[int]:
        """
        Robustly extract a list of integer indices from LLM output.
        Tries JSON array parse first; falls back to comma-separated integers (#18).
        """
        text = response.strip()
        if not text:
            return []

        # Try JSON array anywhere in the response
        json_match = re.search(r'\[[\d\s,]+\]', text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    return [int(x) for x in parsed]
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: collect all integers from the full text
        return [int(m) for m in re.findall(r'\d+', text)]

    async def process(self, ctx: AgentContext) -> AgentContext:
        docs = ctx.current_context
        if not docs:
            ctx.metrics["rerank_skipped_empty"] = ctx.metrics.get("rerank_skipped_empty", 0) + 1
            logger.info("[%s] No documents to rerank, skipping", self.name)
            return ctx

        logger.info(
            "[%s] Starting rerank: docs=%d, num_docs=%d, query_len=%d",
            self.name, len(docs), self.num_docs, len(ctx.query)
        )

        serialized_docs: list[dict[str, Any]] = []
        for i, d in enumerate(docs):
            serialized_docs.append(
                {
                    "id": d.get("id") or f"doc_{i}",
                    "content": d.get("content", "")[:3000],
                    "source": d.get("source", "unknown"),
                    "tool": d.get("tool", "unknown"),
                }
            )

        prompt = self.rerank_prompt.format(
            query=ctx.query,
            documents=json.dumps(serialized_docs, ensure_ascii=False),
            num_docs=self.num_docs,
        )

        logger.debug("[%s] Prompt: %s...", self.name, prompt[:500])
        logger.info(
            "[%s] Request: docs=%d, num_docs=%d, query_len=%d",
            self.name, len(serialized_docs), self.num_docs, len(ctx.query)
        )

        start_time = asyncio.get_running_loop().time()
        response = ""
        try:
            response = await self.llm.generate_async(
                prompt,
                options={"temperature": 0.1},
            )
            elapsed = asyncio.get_running_loop().time() - start_time
            self._track_llm_metric(ctx, elapsed, "reranker")

            logger.info("[%s] Response: len=%d, time=%.2fs", self.name, len(response), elapsed)
            logger.debug("[%s] Raw response: %s", self.name, response.strip()[:500])

            indices = self._parse_indices(response)

            if not indices:
                ctx.metrics["rerank_empty_selection"] = ctx.metrics.get("rerank_empty_selection", 0) + 1
                logger.warning(
                    "[%s] Rerank returned empty/invalid indices, keeping original context", self.name
                )
                return ctx

            seen: set[int] = set()
            cleaned: list[int] = []
            for i in indices:
                if 0 <= i < len(ctx.current_context) and i not in seen:
                    seen.add(i)
                    cleaned.append(i)
                if len(cleaned) >= self.num_docs:
                    break

            if not cleaned:
                ctx.metrics["rerank_empty_selection"] = ctx.metrics.get("rerank_empty_selection", 0) + 1
                logger.warning(
                    "[%s] Rerank indices out of range, keeping original context", self.name
                )
                return ctx

            new_context = [ctx.current_context[i] for i in cleaned]

            logger.info(
                "[%s] Kept %d / %d docs after rerank",
                self.name, len(new_context), len(ctx.current_context)
            )

            ctx.metadata["rerank_indices"] = cleaned
            ctx.metadata["rerank_before_count"] = len(ctx.current_context)
            ctx.current_context = new_context
            ctx.metadata["rerank_after_count"] = len(ctx.current_context)

            ctx.metrics["rerank_success"] = ctx.metrics.get("rerank_success", 0) + 1
            ctx.metrics["docs_after_rerank"] = len(ctx.current_context)

        except Exception as e:
            elapsed = asyncio.get_running_loop().time() - start_time
            logger.error("[%s] Error after %.2fs: %s", self.name, elapsed, e)
            logger.error("[%s] RAW OUTPUT (truncated): %r", self.name, response.strip()[:300])
            ctx.metadata["rerank_error"] = str(e)
            ctx.metrics["rerank_errors"] = ctx.metrics.get("rerank_errors", 0) + 1

        return ctx
