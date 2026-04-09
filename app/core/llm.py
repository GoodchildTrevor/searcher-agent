from abc import ABC, abstractmethod
from typing import Optional, Any
import re
import logging

from ollama import AsyncClient

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.

    Defines the contract that all LLM implementations must follow
    to be compatible with the RAG agent nodes.
    """

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        options: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Generate text response from the LLM.

        :param prompt: Input prompt for generation.
        :param options: Optional generation parameters (temperature, top_p, etc.).
        :return: Generated text response.
        :raises Exception: If generation fails.
        """
        pass


class OllamaLLM(BaseLLM):
    """
    Ollama LLM client implementation.

    Provides async generation capabilities using the Ollama API.
    Compatible with the BaseLLM interface for dependency injection.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0
    ):
        """
        Initialize Ollama client.

        :param base_url: Ollama server URL (e.g., http://localhost:11434).
        :param model: Model name to use (e.g., 'llama3.1', 'qwen2.5').
        :param timeout: Request timeout in seconds.
        """
        self.client = AsyncClient(host=base_url, timeout=timeout)
        self.model = model
        self.request_timeout = timeout

    async def generate_async(
        self,
        prompt: str,
        options: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Generate text using Ollama API.

        :param prompt: The input prompt for generation.
        :param options: Optional generation parameters.
        :return: Generated text response.
        :raises Exception: If API call fails or response is invalid.
        """
        response = await self.client.generate(
            model=self.model,
            prompt=prompt,
            options=options or {},
            stream=False
        )
        return response["response"]

    async def evaluate_confidence_async(self, context: str, query: str) -> float:
        """
        Evaluate confidence that context answers the query.

        :param context: Retrieved context text.
        :param query: User query to evaluate.
        :return: Confidence score between 0.0 and 1.0.
        """
        prompt = f"""Analyze if the context sufficiently answers the query.

        Query: {query}
        Context: {context[:2000]}

        Rate confidence from 0.0 to 1.0 where 1.0 means fully answered.
        Output ONLY a decimal number between 0.0 and 1.0, nothing else.

        Confidence:"""

        try:
            response = await self.generate_async(
                prompt,
                options={"temperature": 0.1, "num_predict": 10}
            )
            numbers = re.findall(r"0?\.\d+|1\.0+|[01]", response.strip())
            if numbers:
                confidence = float(numbers[0])
                # Guard against model returning 0-10 scale despite instructions
                if confidence > 1.0:
                    confidence = confidence / 10.0
                return max(0.0, min(1.0, confidence))
            logger.warning("Confidence evaluation returned no parseable number: %s", response.strip())
            return 0.5
        except Exception as e:
            logger.warning("Confidence evaluation failed: %s", e)
            return 0.5
