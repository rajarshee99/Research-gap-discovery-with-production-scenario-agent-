"""OSS 120B reasoning model wrapper for the Research Gap Discovery & Industry Alignment Agent.

This module provides a production-ready wrapper around the Groq OSS 120B reasoning model.
It handles authentication, request execution, and error handling while hiding
provider-specific implementation details.

The wrapper is designed to be used by higher-level components such as:
- generator.py (for text generation)
- router.py (for model selection)
- factory.py (for model instantiation)

This module does NOT contain:
- LangGraph logic
- Retrieval logic
- Routing logic
- Prompt engineering
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Retry configuration for rate limit errors
MAX_RETRIES = 5  # Increased retries for TPM limits
INITIAL_RETRY_DELAY = 5.0  # Longer initial delay for TPM limits
RETRY_BACKOFF_FACTOR = 2.0


class OSS120B:
    """Production-ready wrapper for the Groq OSS 120B reasoning model.

    This class provides a simple interface for interacting with the OSS 120B model
    while handling authentication, validation, and error management.

    Example:
        >>> model = OSS120B()
        >>> if model.is_available():
        ...     response = model.generate("What are research gaps in AI?")
        ...     print(response)
    """

    # Model identifier used by Groq API
    GROQ_MODEL_NAME = "openai/gpt-oss-120b"

    # Default generation parameters
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2048  # Reduced from 8192 to stay within TPM limits
    DEFAULT_TOP_P = 1.0
    DEFAULT_REASONING_EFFORT = "medium"

    def __init__(self, api_keys_path: str = "api_keys.txt"):
        """Initialize the OSS 120B model wrapper.

        Args:
            api_keys_path: Path to the API keys file. Defaults to "api_keys.txt".

        Raises:
            RuntimeError: If the API key file is missing or invalid.
        """
        self._api_key: Optional[str] = None
        self._client: Optional[Any] = None
        self._api_keys_path = api_keys_path
        self._initialized = False

        try:
            self._initialize()
        except Exception as exc:
            logger.error("Failed to initialize OSS120B: %s", exc)
            raise

    def _initialize(self) -> None:
        """Initialize the Groq client with API credentials.

        This method reads the API key from the api_keys.txt file and
        initializes the Groq client.

        Raises:
            RuntimeError: If API key is missing or invalid.
        """
        logger.info("Initializing OSS120B model wrapper")

        try:
            self._api_key = self._read_groq_api_key()
            logger.info("Successfully read Groq API key")

            # Import Groq client
            try:
                from groq import Groq  # type: ignore

                self._client = Groq(api_key=self._api_key)
                logger.info("Successfully initialized Groq client")
            except ImportError as exc:
                raise RuntimeError(
                    "Groq Python client is not installed. "
                    "Install it with: pip install groq"
                ) from exc

            self._initialized = True
            logger.info("OSS120B model wrapper initialized successfully")

        except Exception as exc:
            logger.exception("Failed to initialize OSS120B")
            self._initialized = False
            raise

    def _read_groq_api_key(self) -> str:
        """Read Groq API key from api_keys.txt.

        Expected line pattern:
            groq api key: <API_KEY>

        Returns:
            The Groq API key.

        Raises:
            RuntimeError: If the API key file is missing or the key is invalid.
        """
        path = Path(self._api_keys_path)
        if not path.exists():
            raise RuntimeError(
                f"Missing API credentials file: {self._api_keys_path}. "
                "Expected a line like 'groq api key: <API_KEY>'."
            )

        content = path.read_text(encoding="utf-8", errors="ignore")

        # Pattern to match Groq API key (case-insensitive)
        m = re.search(r"groq\s+api\s+key\s*:\s*(\S+)", content, flags=re.IGNORECASE)
        if not m:
            raise RuntimeError(
                f"Unable to find Groq API key in {self._api_keys_path}. "
                "Expected a line like 'groq api key: <API_KEY>'."
            )

        api_key = m.group(1).strip()
        if not api_key:
            raise RuntimeError(
                f"Groq API key found in {self._api_keys_path} but is empty."
            )

        return api_key

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using the OSS 120B reasoning model.

        Args:
            prompt: The input prompt for text generation.
            **kwargs: Additional generation parameters such as:
                - temperature (float): Sampling temperature (default: 0.7)
                - max_tokens (int): Maximum tokens to generate (default: 8192)
                - top_p (float): Nucleus sampling parameter (default: 1.0)
                - reasoning_effort (str): Reasoning effort level (default: "medium")
                - stream (bool): Whether to stream responses (default: False)

        Returns:
            The generated text as a string.

        Raises:
            ValueError: If the prompt is empty or invalid.
            RuntimeError: If the model is not initialized or API call fails.
            TimeoutError: If the request times out.
        """
        if not self._initialized or self._client is None:
            raise RuntimeError(
                "OSS120B model is not initialized. Call is_available() first."
            )

        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string.")

        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt must not be empty or whitespace only.")

        start_time = time.perf_counter()

        # Extract base generation parameters with defaults
        base_temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
        base_max_tokens = kwargs.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        top_p = kwargs.get("top_p", self.DEFAULT_TOP_P)
        reasoning_effort = kwargs.get("reasoning_effort", self.DEFAULT_REASONING_EFFORT)
        stream = kwargs.get("stream", False)

        # Retry logic for rate limit errors
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                # Adjust max_tokens for retries to stay within TPM limits
                max_tokens = base_max_tokens
                if attempt > 0:
                    original_max_tokens = max_tokens
                    max_tokens = max(512, max_tokens // 2)  # Halve max_tokens, minimum 512
                    logger.info(
                        "Retry attempt %d: reduced max_tokens from %d to %d",
                        attempt + 1, original_max_tokens, max_tokens
                    )
                
                estimated_tokens = len(prompt) // 4  # Rough estimate
                logger.info(
                    "OSS120B generate - attempt %d/%d, prompt length: %d characters (~%d tokens), max_tokens=%d",
                    attempt + 1,
                    MAX_RETRIES,
                    len(prompt),
                    estimated_tokens,
                    max_tokens
                )
                
                temperature = base_temperature

                # Log parameters (excluding sensitive data)
                logger.debug(
                    "OSS120B generate - temperature: %.2f, max_tokens: %d, top_p: %.2f, "
                    "reasoning_effort: %s, stream: %s",
                    temperature,
                    max_tokens,
                    top_p,
                    reasoning_effort,
                    stream,
                )

                # Make the API call
                completion = self._client.chat.completions.create(
                    model=self.GROQ_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    top_p=top_p,
                    reasoning_effort=reasoning_effort,
                    stream=stream,
                )

                # Handle streaming vs non-streaming responses
                if stream:
                    # Collect streamed chunks
                    generated_text = ""
                    for chunk in completion:
                        if chunk.choices and chunk.choices[0].delta.content:
                            generated_text += chunk.choices[0].delta.content
                else:
                    # Non-streaming response
                    if not completion.choices:
                        raise RuntimeError("API returned no choices in response.")
                    generated_text = completion.choices[0].message.content or ""

                if not generated_text:
                    logger.warning("OSS120B generate - empty response from API")
                    return ""

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                logger.info(
                    "OSS120B generate - response length: %d characters, elapsed_ms: %.2f",
                    len(generated_text),
                    elapsed_ms,
                )

                return generated_text

            except Exception as exc:
                last_exception = exc
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                
                # Check if this is a rate limit error
                error_str = str(exc).lower()
                is_rate_limit_error = (
                    "rate limit" in error_str or
                    "413" in error_str or
                    "tokens per minute" in error_str or
                    "tpm" in error_str
                )
                
                if is_rate_limit_error and attempt < MAX_RETRIES - 1:
                    # Calculate retry delay with exponential backoff
                    retry_delay = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_FACTOR ** attempt)
                    logger.warning(
                        "OSS120B generate - rate limit error on attempt %d/%d. "
                        "Retrying in %.2f seconds. Error: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        retry_delay,
                        exc
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    # Not a rate limit error or out of retries
                    logger.exception(
                        "OSS120B generate - API call failed after %.2f ms on attempt %d/%d",
                        elapsed_ms,
                        attempt + 1,
                        MAX_RETRIES
                    )
                    raise RuntimeError(f"OSS120B generation failed: {exc}") from exc

        # This should not be reached, but just in case
        raise RuntimeError(f"OSS120B generation failed after {MAX_RETRIES} attempts: {last_exception}")

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Generate text using the OSS 120B model with streaming.

        Args:
            prompt: The input prompt for text generation.
            **kwargs: Additional generation parameters such as:
                - temperature (float): Sampling temperature (default: 0.7)
                - max_tokens (int): Maximum tokens to generate (default: 2048)
                - top_p (float): Nucleus sampling parameter (default: 1.0)
                - reasoning_effort (str): Reasoning effort level (default: "medium")

        Yields:
            Text chunks as they are generated.

        Raises:
            ValueError: If the prompt is empty or invalid.
            RuntimeError: If the model is not initialized or API call fails.
        """
        if not self._initialized or self._client is None:
            raise RuntimeError(
                "OSS120B model is not initialized. Call is_available() first."
            )

        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string.")

        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt must not be empty or whitespace only.")

        # Force streaming to True
        kwargs["stream"] = True
        
        # Extract base generation parameters with defaults
        base_temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
        base_max_tokens = kwargs.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        top_p = kwargs.get("top_p", self.DEFAULT_TOP_P)
        reasoning_effort = kwargs.get("reasoning_effort", self.DEFAULT_REASONING_EFFORT)

        # Retry logic for rate limit errors
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                # Adjust max_tokens for retries to stay within TPM limits
                max_tokens = base_max_tokens
                if attempt > 0:
                    original_max_tokens = max_tokens
                    max_tokens = max(512, max_tokens // 2)  # Halve max_tokens, minimum 512
                    logger.info(
                        "Retry attempt %d: reduced max_tokens from %d to %d",
                        attempt + 1, original_max_tokens, max_tokens
                    )
                
                estimated_tokens = len(prompt) // 4  # Rough estimate
                logger.info(
                    "OSS120B generate_stream - attempt %d/%d, prompt length: %d characters (~%d tokens), max_tokens=%d",
                    attempt + 1,
                    MAX_RETRIES,
                    len(prompt),
                    estimated_tokens,
                    max_tokens
                )
                
                temperature = base_temperature

                # Make the streaming API call
                completion = self._client.chat.completions.create(
                    model=self.GROQ_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    top_p=top_p,
                    reasoning_effort=reasoning_effort,
                    stream=True,  # Force streaming
                )

                # Yield chunks as they arrive
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

                return  # Success, exit retry loop

            except Exception as exc:
                last_exception = exc
                
                # Check if this is a rate limit error
                error_str = str(exc).lower()
                is_rate_limit_error = (
                    "rate limit" in error_str or
                    "413" in error_str or
                    "tokens per minute" in error_str or
                    "tpm" in error_str
                )
                
                if is_rate_limit_error and attempt < MAX_RETRIES - 1:
                    # Calculate retry delay with exponential backoff
                    retry_delay = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_FACTOR ** attempt)
                    logger.warning(
                        "OSS120B generate_stream - rate limit error on attempt %d/%d. "
                        "Retrying in %.2f seconds. Error: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        retry_delay,
                        exc
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    # Not a rate limit error or out of retries
                    logger.exception(
                        "OSS120B generate_stream - API call failed on attempt %d/%d",
                        attempt + 1,
                        MAX_RETRIES
                    )
                    raise RuntimeError(f"OSS120B streaming generation failed: {exc}") from exc

        # This should not be reached, but just in case
        raise RuntimeError(f"OSS120B streaming generation failed after {MAX_RETRIES} attempts: {last_exception}")

    @property
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            The string identifier for this model: "oss120b"
        """
        return "oss120b"

    def is_available(self) -> bool:
        """Check if the model is correctly configured and ready for inference.

        This method validates that:
        - The API key file exists
        - The API key is valid
        - The Groq client can be initialized
        - The model is accessible

        Returns:
            True if the model is available, False otherwise.
        """
        if not self._initialized:
            logger.debug("OSS120B is_available - model not initialized")
            return False

        if self._client is None:
            logger.debug("OSS120B is_available - client is None")
            return False

        try:
            # Verify we can make a simple API call
            # Use a minimal request to check connectivity
            test_prompt = "Hello"
            _ = self.generate(test_prompt, max_tokens=10)
            logger.info("OSS120B is_available - model is accessible")
            return True
        except Exception as exc:
            logger.warning("OSS120B is_available - model not accessible: %s", exc)
            return False

    def __repr__(self) -> str:
        """Return a string representation of the OSS120B instance.

        Returns:
            A descriptive string representation.
        """
        status = "initialized" if self._initialized else "not initialized"
        return f"OSS120B(model_name={self.model_name}, status={status})"
