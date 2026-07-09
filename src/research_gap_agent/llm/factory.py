"""LLM Factory for the Research Gap Discovery & Industry Alignment Agent.

This module provides a centralized factory for creating LLM wrapper instances.
It handles model instantiation, validation, and hides implementation details
from higher-level components.

The factory is designed to be used by:
- router.py (for model selection)
- generator.py (for text generation)
- future LangGraph nodes

This module does NOT contain:
- Model inference logic
- Prompt building
- Retrieval logic
- LangGraph logic
- Routing logic
- Pinecone calls
"""

from __future__ import annotations

import logging
from typing import Any

from research_gap_agent.llm.oss20B import OSS20B
from research_gap_agent.llm.oss120B import OSS120B

logger = logging.getLogger(__name__)


class LLMFactory:
    """Centralized factory for creating LLM wrapper instances.

    This factory provides a single point of entry for instantiating LLM models,
    ensuring consistent initialization and hiding implementation details from
    calling code.

    Example:
        >>> factory = LLMFactory()
        >>> llm = factory.get_llm("oss20b")
        >>> response = llm.generate("What are research gaps in AI?")
    """

    # Registry of available models
    _MODEL_REGISTRY: dict[str, type[Any]] = {
        "oss20b": OSS20B,
        "oss120b": OSS120B,
        "oss20B": OSS20B,
        "oss120B": OSS120B,
    }

    def __init__(self, api_keys_path: str = "api_keys.txt"):
        """Initialize the LLM Factory.

        Args:
            api_keys_path: Path to the API keys file. Defaults to "api_keys.txt".
                          This path will be passed to all model wrappers.
        """
        self._api_keys_path = api_keys_path
        logger.info("LLMFactory initialized with api_keys_path: %s", api_keys_path)

    def get_llm(self, model_name: str) -> Any:
        """Get an LLM wrapper instance for the specified model.

        Args:
            model_name: The name of the model to instantiate.
                       Supported values: "oss20b", "oss120b"

        Returns:
            An instantiated LLM wrapper instance.

        Raises:
            ValueError: If the model name is unsupported or invalid.
            RuntimeError: If model initialization fails.

        Example:
            >>> factory = LLMFactory()
            >>> llm = factory.get_llm("oss20b")
            >>> isinstance(llm, OSS20B)
            True
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("Model name must be a non-empty string.")

        model_name = model_name.strip().lower()

        if not model_name:
            raise ValueError("Model name must not be empty or whitespace only.")

        logger.info("LLMFactory get_llm - requested model: %s", model_name)

        # Check if model is supported
        if model_name not in self._MODEL_REGISTRY:
            available = ", ".join(self.available_models())
            error_msg = (
                f"Unsupported model '{model_name}'. "
                f"Available models: {available}"
            )
            logger.error("LLMFactory get_llm - %s", error_msg)
            raise ValueError(error_msg)

        # Get the model class from registry
        model_class = self._MODEL_REGISTRY[model_name]
        logger.debug(
            "LLMFactory get_llm - instantiating model class: %s", model_class.__name__
        )

        try:
            # Instantiate the model wrapper
            model_instance = model_class(api_keys_path=self._api_keys_path)
            logger.info(
                "LLMFactory get_llm - successfully instantiated %s",
                model_instance.model_name,
            )
            return model_instance

        except Exception as exc:
            logger.exception(
                "LLMFactory get_llm - failed to instantiate model %s", model_name
            )
            raise RuntimeError(
                f"Failed to initialize model '{model_name}': {exc}"
            ) from exc

    def available_models(self) -> list[str]:
        """Get a list of all available model names.

        Returns:
            A list of supported model names in alphabetical order.

        Example:
            >>> factory = LLMFactory()
            >>> models = factory.available_models()
            >>> "oss20b" in models
            True
        """
        models = sorted(self._MODEL_REGISTRY.keys())
        logger.debug("LLMFactory available_models - returning %d models", len(models))
        return models

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is supported by the factory.

        Args:
            model_name: The name of the model to check.

        Returns:
            True if the model is supported, False otherwise.

        Example:
            >>> factory = LLMFactory()
            >>> factory.is_model_available("oss20b")
            True
            >>> factory.is_model_available("unknown_model")
            False
        """
        if not model_name or not isinstance(model_name, str):
            return False

        model_name = model_name.strip().lower()
        is_available = model_name in self._MODEL_REGISTRY

        logger.debug(
            "LLMFactory is_model_available - model=%s, available=%s",
            model_name,
            is_available,
        )
        return is_available

    def __repr__(self) -> str:
        """Return a string representation of the LLMFactory instance.

        Returns:
            A descriptive string representation.
        """
        models = ", ".join(self.available_models())
        return f"LLMFactory(available_models=[{models}])"
