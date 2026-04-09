import asyncio
import json
import logging
import re
from typing import Any, Optional

from app.configs.agent_settings import AgentContext, AgentState, AgentNode, BaseNode

logger = logging.getLogger(__name__)


class RouterNode(BaseNode):
    """
    Node responsible for evaluating context sufficiency.
    Uses binary decision (answer_found: true/false) rather than
    a float confidence score, which is more reliable on small models.
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
            logger.info(f"[{self.name}] No context available, routing → TOOL_SELECTION")  # <<< NEW
            return AgentState.TOOL_SELECTION

        prompt = self.router_prompt.format(
            query=ctx.query,
            context=context_text[:3000]
        )

        logger.debug(f"[{self.name}] Prompt: {prompt[:500]}...")
        logger.info(
            f"[{self.name}] Request: query_len={len(ctx.query)}, "
            f"context_docs={len(ctx.current_context)}, "
            f"context_len={len(context_text)}"
        )

        answer_found = False
        start_time = asyncio.get_event_loop().time()

        try:
            response = await self.llm.generate_async(
                prompt,
                options={"temperature": 0.1, "num_predict": 50}
            )

            elapsed = asyncio.get_event_loop().time() - start_time
            self._track_llm_metric(ctx, elapsed, "router")

            logger.info(f"[{self.name}] Response: len={len(response)}, time={elapsed:.2f}s")
            logger.debug(f"[{self.name}] Raw response: {response.strip()}")

            try:
                json_str = self._extract_json(response) or response.strip()
                parsed = json.loads(json_str)
                answer_found = bool(parsed.get("answer_found", False))
                ctx.metrics["router_successful_parses"] = (
                    ctx.metrics.get("router_successful_parses", 0) + 1
                )
                logger.info(f"[{self.name}] Parsed answer_found={answer_found}")

            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                logger.warning(
                    f"[{self.name}] Parse error, treating as answer_found=False: {e}. "
                    f"Raw: {response.strip()[:100]}"
                )
                answer_found = False
                ctx.metrics["router_parse_errors"] = (
                    ctx.metrics.get("router_parse_errors", 0) + 1
                )

        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"[{self.name}] LLM error after {elapsed:.2f}s: {e}")
            ctx.metadata["router_error"] = str(e)
            ctx.metrics["router_llm_errors"] = (
                ctx.metrics.get("router_llm_errors", 0) + 1
            )
            answer_found = False

        ctx.confidence_score = 1.0 if answer_found else 0.0
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
                logger.info(f"[{self.name}] Decision: sufficient_context → ANSWERING")  # <<< NEW
                return AgentState.ANSWERING
            else:
                ctx.metadata["router_decision"] = "sufficient_context → GIVE_INFO"
                ctx.metrics["info_only_routes"] = (
                    ctx.metrics.get("info_only_routes", 0) + 1
                )
                logger.info(f"[{self.name}] Decision: sufficient_context → GIVE_INFO")  # <<< NEW
                return AgentState.GIVE_INFO

        if ctx.can_search_more():
            ctx.metadata["router_decision"] = (
                f"need_more_info (iteration {ctx.current_iterations}) → TOOL_SELECTION"
            )
            ctx.metrics["search_routes"] = ctx.metrics.get("search_routes", 0) + 1
            logger.info(  # <<< NEW
                f"[{self.name}] Decision: need_more_info "
                f"(iteration {ctx.current_iterations}/{ctx.max_iterations}) → TOOL_SELECTION"
            )
            return AgentState.TOOL_SELECTION

        ctx.metadata["router_decision"] = (
            f"max_iterations_reached ({ctx.current_iterations}) → ANSWERING"
        )
        ctx.metrics["forced_answer_routes"] = (
            ctx.metrics.get("forced_answer_routes", 0) + 1
        )
        logger.warning(  # <<< NEW (warning, not info — forced path is notable)
            f"[{self.name}] Decision: max_iterations_reached "
            f"({ctx.current_iterations}/{ctx.max_iterations}) → ANSWERING (forced)"
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

        logger.debug(f"[{self.name}] Prompt: {prompt[:500]}...")
        logger.info(  # <<< NEW
            f"[{self.name}] Request: query_len={len(ctx.query)}, "
            f"available_tools={list(self.available_tools.keys())}, "
            f"context_docs={len(ctx.current_context)}"
        )

        start_time = asyncio.get_event_loop().time()
        try:
            response = await self.llm.generate_async(
                prompt,
                options={"temperature": 0.1, "num_predict": 100}
            )

            elapsed = asyncio.get_event_loop().time() - start_time
            self._track_llm_metric(ctx, elapsed, "tool_selection")
            logger.info(f"[{self.name}] Response: len={len(response)}, time={elapsed:.2f}s")  # <<< NEW
            logger.debug(f"[{self.name}] Raw response: {response.strip()}")  # <<< NEW

            json_str = self._extract_json(response)
            if json_str:
                try:
                    tools_list = json.loads(json_str)
                    if isinstance(tools_list, list):
                        ctx.selected_tools = [
                            t for t in tools_list
                            if t in self.available_tools
                        ]
                        # <<< NEW: warn about tools requested but not available
                        unknown = [t for t in tools_list if t not in self.available_tools]
                        if unknown:
                            logger.warning(
                                f"[{self.name}] LLM requested unknown tools, skipping: {unknown}"
                            )
                        ctx.metrics["tool_selection_success"] = ctx.metrics.get("tool_selection_success", 0) + 1
                        ctx.metrics["tools_selected_count"] = ctx.metrics.get("tools_selected_count", 0) + len(ctx.selected_tools)
                        logger.info(f"[{self.name}] Selected tools: {ctx.selected_tools}")  # <<< NEW
                    else:
                        ctx.selected_tools = []
                        ctx.metrics["tool_selection_invalid_format"] = ctx.metrics.get("tool_selection_invalid_format", 0) + 1
                        logger.warning(f"[{self.name}] Invalid format (expected list), no tools selected")  # <<< NEW
                except json.JSONDecodeError as e:
                    ctx.selected_tools = []
                    ctx.metrics["tool_selection_json_error"] = ctx.metrics.get("tool_selection_json_error", 0) + 1
                    logger.warning(f"[{self.name}] JSON decode error: {e}. Raw: {json_str[:100]}")  # <<< NEW
            else:
                ctx.selected_tools = []
                ctx.metrics["tool_selection_no_json"] = ctx.metrics.get("tool_selection_no_json", 0) + 1
                logger.warning(f"[{self.name}] No JSON found in response, no tools selected")  # <<< NEW

            ctx.metadata["selected_tools"] = ctx.selected_tools
            ctx.current_tool_index = 0

        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"[{self.name}] Error after {elapsed:.2f}s: {e}")
            ctx.selected_tools = []
            ctx.metadata["tool_selection_error"] = str(e)
            ctx.metrics["tool_selection_llm_errors"] = ctx.metrics.get("tool_selection_llm_errors", 0) + 1

        return ctx


class ExpansionNode(BaseNode):
    """
    Node responsible for generating query variations.
    """

    def __init__(
        self,
        llm: Any,
        expansion_prompt: str,
        expansion_count: int = 3,
        timeout: float = 30.0
    ):
        super().__init__(llm, "expansion")
        self.expansion_prompt = expansion_prompt
        self.expansion_count = expansion_count
        self.timeout = timeout

    async def process(self, ctx: AgentContext) -> AgentContext:
        if ctx.current_tool_index >= len(ctx.selected_tools):
            logger.warning(  # <<< NEW
                f"[{self.name}] current_tool_index={ctx.current_tool_index} "
                f">= selected_tools={len(ctx.selected_tools)}, skipping expansion"
            )
            return ctx

        current_tool_name = ctx.selected_tools[ctx.current_tool_index]
        logger.info(  # <<< NEW
            f"[{self.name}] Expanding query for tool '{current_tool_name}' "
            f"(tool {ctx.current_tool_index + 1}/{len(ctx.selected_tools)}), "
            f"count={self.expansion_count}"
        )

        previous_queries = []
        for queries in ctx.tool_specific_queries.values():
            previous_queries.extend(queries)
        previous_queries_str = ", ".join(f'"{q}"' for q in previous_queries[-10:])

        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in ctx.chat_history[-5:])

        start_time = asyncio.get_event_loop().time()
        try:
            response = await asyncio.wait_for(
                self.llm.generate_async(
                    self.expansion_prompt.format(
                        count=self.expansion_count,
                        query=ctx.query,
                        tool_name=current_tool_name,
                        tool_description="Search tool",
                        previous_queries=previous_queries_str,
                        history=history_text if history_text else "History empty.",
                        iteration=ctx.current_iterations,
                        max_iterations=ctx.max_iterations
                    ),
                    options={"temperature": 0.8, "top_p": 0.9, "num_predict": 300}
                ),
                timeout=self.timeout
            )

            elapsed = asyncio.get_event_loop().time() - start_time
            self._track_llm_metric(ctx, elapsed, "expansion")
            logger.info(f"[{self.name}] Response: len={len(response)}, time={elapsed:.2f}s")  # <<< NEW
            logger.debug(f"[{self.name}] Raw response: {response.strip()}")  # <<< NEW

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
                        logger.info(  # <<< NEW
                            f"[{self.name}] Generated {len(ctx.tool_specific_queries[current_tool_name])} "
                            f"queries for '{current_tool_name}': "
                            f"{ctx.tool_specific_queries[current_tool_name]}"
                        )
                    else:
                        ctx.tool_specific_queries[current_tool_name] = [ctx.query]
                        ctx.metrics["expansion_invalid_format"] = ctx.metrics.get("expansion_invalid_format", 0) + 1
                        logger.warning(  # <<< NEW
                            f"[{self.name}] Invalid expansion format for '{current_tool_name}', "
                            f"falling back to original query"
                        )
                except json.JSONDecodeError as e:
                    ctx.tool_specific_queries[current_tool_name] = [ctx.query]
                    ctx.metrics["expansion_json_error"] = ctx.metrics.get("expansion_json_error", 0) + 1
                    logger.warning(  # <<< NEW
                        f"[{self.name}] JSON decode error for '{current_tool_name}': {e}, "
                        f"falling back to original query"
                    )
            else:
                ctx.tool_specific_queries[current_tool_name] = [ctx.query]
                ctx.metrics["expansion_no_json"] = ctx.metrics.get("expansion_no_json", 0) + 1
                logger.warning(  # <<< NEW
                    f"[{self.name}] No JSON found in response for '{current_tool_name}', "
                    f"falling back to original query"
                )

        except asyncio.TimeoutError:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"[{self.name}] Timeout for {current_tool_name} after {elapsed:.2f}s")
            ctx.tool_specific_queries[current_tool_name] = [ctx.query]
            ctx.metadata[f"expansion_timeout_{current_tool_name}"] = True
            ctx.metrics["expansion_timeouts"] = ctx.metrics.get("expansion_timeouts", 0) + 1

        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"[{self.name}] Error for {current_tool_name} after {elapsed:.2f}s: {e}")
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
            logger.warning(  # <<< NEW
                f"[retrieval] current_tool_index={ctx.current_tool_index} "
                f">= selected_tools={len(ctx.selected_tools)}, skipping retrieval"
            )
            return ctx

        current_tool_name = ctx.selected_tools[ctx.current_tool_index]
        current_tool = self.tools_registry.get(current_tool_name)

        if not current_tool:
            logger.error(f"Tool {current_tool_name} not found in registry")
            ctx.metrics["retrieval_tool_not_found"] = ctx.metrics.get("retrieval_tool_not_found", 0) + 1
            return ctx

        queries = ctx.tool_specific_queries.get(current_tool_name, [ctx.query])
        logger.info(  # <<< NEW
            f"[retrieval] Starting retrieval with tool '{current_tool_name}', "
            f"queries={len(queries)}, top_k=5"
        )
        logger.debug(f"[retrieval] Queries: {queries}")  # <<< NEW

        start_time = asyncio.get_event_loop().time()
        try:
            results = await current_tool.batch_search_async(queries=queries, top_k=5)
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(  # <<< NEW
                f"[retrieval] Tool '{current_tool_name}' returned results "
                f"for {len(results)} queries in {elapsed:.2f}s"
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
                    logger.debug(f"[retrieval] Skipping duplicate doc id={doc_id}")  # <<< NEW
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

            logger.info(  # <<< NEW
                f"[retrieval] Tool '{current_tool_name}': "
                f"total_raw={len(ctx.tool_results[current_tool_name])}, "
                f"new_unique={new_docs_count}, "
                f"total_context={len(ctx.current_context)}"
            )

        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"Retrieval error for {current_tool_name} after {elapsed:.2f}s: {e}")
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

        # <<< NEW: warn explicitly when answering with no context
        if not context_text:
            logger.warning(f"[{self.name}] Answering with empty context")

        prompt = self.answer_prompt.format(
            context=context_text if context_text else "Context not found.",
            history=history_text if history_text else "Dialog history is empty.",
            query=ctx.query,
            confidence_score=ctx.confidence_score if ctx.confidence_score is not None else "not evaluated"
        )

        logger.debug(f"[{self.name}] Prompt: {prompt[:500]}...")
        logger.info(
            f"[{self.name}] Request: context_docs={len(ctx.current_context)}, "
            f"query_len={len(ctx.query)}, "
            f"confidence_score={ctx.confidence_score}"  # <<< NEW: include confidence
        )

        start_time = asyncio.get_event_loop().time()
        try:
            ctx.final_answer = await self.llm.generate_async(prompt, options={"temperature": 0.4})

            elapsed = asyncio.get_event_loop().time() - start_time
            self._track_llm_metric(ctx, elapsed, "answer")

            answer_len = len(ctx.final_answer) if ctx.final_answer else 0
            logger.info(f"[{self.name}] Response: len={answer_len}, time={elapsed:.2f}s")

            if answer_len < 50:
                logger.warning(f"[{self.name}] Unusually short answer ({answer_len} chars)")
                ctx.metrics["short_answers"] = ctx.metrics.get("short_answers", 0) + 1
            elif answer_len > 5000:
                logger.warning(f"[{self.name}] Unusually long answer ({answer_len} chars)")
                ctx.metrics["long_answers"] = ctx.metrics.get("long_answers", 0) + 1

            ctx.metadata["answer_generated"] = True
            ctx.metrics["answer_success"] = ctx.metrics.get("answer_success", 0) + 1

        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"[{self.name}] Error after {elapsed:.2f}s: {e}")
            ctx.final_answer = "Sorry, an error occurred while generating the answer."
            ctx.metadata["answer_error"] = str(e)
            ctx.metrics["answer_errors"] = ctx.metrics.get("answer_errors", 0) + 1

        return ctx


class GiveInfoNode:
    """
    Node for returning context info without generating an answer.
    """

    async def process(self, ctx: AgentContext) -> AgentContext:
        logger.info(  # <<< NEW
            f"[give_info] Returning {len(ctx.current_context)} docs "
            f"without answer generation (info_only mode)"
        )
        ctx.metadata["info_only"] = True
        ctx.metadata["answer_generated"] = False
        ctx.metrics["info_only_responses"] = ctx.metrics.get("info_only_responses", 0) + 1
        return ctx


class RerankerNode:
    name = "reranker"

    def __init__(self, llm, rerank_prompt: str, num_docs: int = 10):
        self.llm = llm
        self.rerank_prompt = rerank_prompt
        self.num_docs = num_docs

    def _track_llm_metric(self, ctx, elapsed: float, kind: str) -> None:
        ctx.metrics["llm_calls"] = ctx.metrics.get("llm_calls", 0) + 1
        ctx.metrics["llm_total_time"] = ctx.metrics.get("llm_total_time", 0.0) + elapsed
        ctx.metrics[f"{kind}_time"] = elapsed

    async def process(self, ctx: AgentContext) -> AgentContext:
        docs = ctx.current_context
        if not docs:
            ctx.metrics["rerank_skipped_empty"] = ctx.metrics.get("rerank_skipped_empty", 0) + 1
            logger.info(f"[{self.name}] No documents to rerank, skipping")
            return ctx

        logger.info(
            f"[{self.name}] Starting rerank: docs={len(docs)}, "
            f"num_docs={self.num_docs}, query_len={len(ctx.query)}"
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

        import json
        prompt = self.rerank_prompt.format(
            query=ctx.query,
            documents=json.dumps(serialized_docs, ensure_ascii=False),
            num_docs=self.num_docs,
        )

        logger.debug(f"[{self.name}] Prompt: {prompt[:500]}...")
        logger.info(
            f"[{self.name}] Request: docs={len(serialized_docs)}, "
            f"num_docs={self.num_docs}, query_len={len(ctx.query)}"
        )

        start_time = asyncio.get_event_loop().time()
        response = ""
        try:
            response = await self.llm.generate_async(
                prompt,
                options={"temperature": 0.1},
            )
            elapsed = asyncio.get_event_loop().time() - start_time
            self._track_llm_metric(ctx, elapsed, "reranker")

            logger.info(f"[{self.name}] Response: len={len(response)}, time={elapsed:.2f}s")
            logger.debug(f"[{self.name}] Raw response: {response.strip()[:500]}")

            text = response.strip()
            if not text:
                raise ValueError("Empty reranker response")

            first_line = text.splitlines()[0]

            first_line = first_line.replace(";", ",")
            parts = [p.strip() for p in first_line.split(",")]

            indices: list[int] = []
            for p in parts:
                if not p:
                    continue
                try:
                    indices.append(int(p))
                except ValueError:
                    logger.debug(f"[{self.name}] Cannot cast {p!r} to int, skipping")

            if not indices:
                ctx.metrics["rerank_empty_selection"] = ctx.metrics.get(
                    "rerank_empty_selection", 0
                ) + 1
                logger.warning(
                    f"[{self.name}] Rerank returned empty/invalid indices, keeping original context"
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
                ctx.metrics["rerank_empty_selection"] = ctx.metrics.get(
                    "rerank_empty_selection", 0
                ) + 1
                logger.warning(
                    f"[{self.name}] Rerank indices out of range, keeping original context"
                )
                return ctx

            new_context = [ctx.current_context[i] for i in cleaned]

            logger.info(
                f"[{self.name}] Kept {len(new_context)} / {len(ctx.current_context)} docs "
                f"after rerank"
            )

            ctx.metadata["rerank_indices"] = cleaned
            ctx.metadata["rerank_before_count"] = len(ctx.current_context)
            ctx.current_context = new_context
            ctx.metadata["rerank_after_count"] = len(ctx.current_context)

            ctx.metrics["rerank_success"] = ctx.metrics.get("rerank_success", 0) + 1
            ctx.metrics["docs_after_rerank"] = len(ctx.current_context)

        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"[{self.name}] Error after {elapsed:.2f}s: {e}")
            logger.error(f"[{self.name}] RAW OUTPUT (truncated): {response.strip()[:300]!r}")
            ctx.metadata["rerank_error"] = str(e)
            ctx.metrics["rerank_errors"] = ctx.metrics.get("rerank_errors", 0) + 1

        return ctx
