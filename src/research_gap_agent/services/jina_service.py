"""
Jina AI content extraction service for the Research Gap Discovery Agent.

This module provides content extraction capabilities using Jina AI,
which is used to extract key information from web pages found via Tavily search.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)


class JinaAIService:
    """Service for extracting key information from web content using Jina AI."""

    def __init__(self, api_keys_path: str = "api_keys.txt"):
        """Initialize Jina AI service with API key from file."""
        self._api_key = self._read_jina_api_key(api_keys_path)
        # Jina AI reader API doesn't need the base URL, we construct URLs dynamically
        self._base_url = "https://r.jina.ai"

    def _read_jina_api_key(self, api_keys_path: str) -> str:
        """Read Jina AI API key from api_keys.txt."""
        path = Path(api_keys_path)
        if not path.exists():
            logger.warning("API keys file not found: %s. Jina AI reader may work without API key for basic usage.", api_keys_path)
            return ""  # Return empty string - Jina AI reader often works without API key
        
        content = path.read_text(encoding="utf-8", errors="ignore")
        
        # Try multiple patterns for flexibility
        patterns = [
            r"Jina\s+AI\s+api\s+key\s*:\s*([A-Za-z0-9_\-]+)",
            r"jina\s+ai\s+api\s+key\s*:\s*([A-Za-z0-9_\-]+)",
            r"jina_[a-zA-Z0-9_\-]+"
        ]
        
        for pattern in patterns:
            m = re.search(pattern, content, flags=re.IGNORECASE)
            if m:
                api_key = m.group(1).strip() if m.group(1) else m.group(0).strip()
                if api_key:
                    logger.info("Found Jina AI API key in %s", api_keys_path)
                    return api_key

        logger.warning("Jina AI API key not found in %s. Jina AI reader may work without API key for basic usage.", api_keys_path)
        return ""  # Return empty string - Jina AI reader often works without API key

    def extract_key_info(
        self,
        urls: List[str],
        query: Optional[str] = None,
        max_length: int = 2000,
    ) -> Dict[str, Any]:
        """
        Extract key information from web pages using Jina AI.

        Args:
            urls: List of URLs to extract content from.
            query: Optional query to guide extraction (helps focus on relevant info).
            max_length: Maximum length of extracted content per URL (default: 2000).

        Returns:
            Dictionary containing extracted information:
            {
                "query": Optional[str],
                "extracted_content": [
                    {
                        "url": str,
                        "key_information": str,
                        "status": str
                    },
                    ...
                ],
                "total_extracted": int,
                "failed_urls": List[str]
            }

        Raises:
            RuntimeError: If API call fails or returns error.
        """
        if not urls:
            logger.warning("No URLs provided for Jina AI extraction")
            return {
                "query": query,
                "extracted_content": [],
                "total_extracted": 0,
                "failed_urls": [],
            }

        logger.info(
            "Jina AI extraction - URLs: %d, query: %s, max_length: %s",
            len(urls),
            query or "None",
            max_length
        )

        extracted_content = []
        failed_urls = []

        for url in urls:
            try:
                # Use Jina AI's reader API for content extraction
                # Fix URL construction to handle both http and https properly
                clean_url = url.replace('https://', '').replace('http://', '')
                reader_url = f"https://r.jina.ai/http://{clean_url}"
                
                headers = {}
                # Only add Authorization header if we have an API key
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"

                logger.debug("Extracting from URL: %s via Jina AI reader", url)
                
                response = requests.get(
                    reader_url,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()

                content = response.text
                
                # Check if we got valid content
                if not content or len(content) < 50:
                    logger.warning("Insufficient content extracted from: %s", url)
                    extracted_content.append({
                        "url": url,
                        "key_information": "",
                        "status": "insufficient content"
                    })
                    failed_urls.append(url)
                    continue

                # If content is too long, truncate it intelligently
                if len(content) > max_length:
                    # Try to truncate at sentence boundaries
                    content = content[:max_length]
                    last_period = content.rfind('.')
                    if last_period > max_length * 0.8:  # Only if we're not cutting too much
                        content = content[:last_period + 1]

                extracted_content.append({
                    "url": url,
                    "key_information": content,
                    "status": "success"
                })

                logger.debug("Successfully extracted content from: %s", url)

            except requests.exceptions.Timeout:
                logger.warning("Timeout extracting content from: %s", url)
                failed_urls.append(url)
                extracted_content.append({
                    "url": url,
                    "key_information": "",
                    "status": "timeout"
                })
            except requests.exceptions.RequestException as exc:
                logger.warning("Failed to extract content from %s: %s", url, exc)
                failed_urls.append(url)
                extracted_content.append({
                    "url": url,
                    "key_information": "",
                    "status": f"error: {str(exc)}"
                })
            except Exception as exc:
                logger.exception("Unexpected error extracting from: %s", url)
                failed_urls.append(url)
                extracted_content.append({
                    "url": url,
                    "key_information": "",
                    "status": f"unexpected error: {str(exc)}"
                })

        logger.info(
            "Jina AI extraction completed - extracted: %d, failed: %d",
            len(extracted_content) - len(failed_urls),
            len(failed_urls)
        )

        return {
            "query": query,
            "extracted_content": extracted_content,
            "total_extracted": len(extracted_content) - len(failed_urls),
            "failed_urls": failed_urls,
        }

    def extract_from_search_results(
        self,
        search_results: List[Dict[str, Any]],
        query: Optional[str] = None,
        max_results: int = 5,
        max_length: int = 2000,
    ) -> Dict[str, Any]:
        """
        Extract key information from Tavily search results.

        Args:
            search_results: List of search result dictionaries from Tavily.
            query: Optional query to guide extraction.
            max_results: Maximum number of results to process (default: 5).
            max_length: Maximum length of extracted content per URL (default: 2000).

        Returns:
            Dictionary containing extracted information from search results.
        """
        if not search_results:
            logger.warning("No search results provided for Jina AI extraction")
            return {
                "query": query,
                "extracted_content": [],
                "total_extracted": 0,
                "failed_urls": [],
            }

        # Extract URLs from search results
        urls = [result.get("url", "") for result in search_results[:max_results] if result.get("url")]
        
        logger.info(
            "Extracting from %d search results (limited to max_results=%d)",
            len(search_results),
            max_results
        )

        return self.extract_key_info(urls, query=query, max_length=max_length)


if __name__ == "__main__":
    # Self-test
    logging.basicConfig(level=logging.INFO)
    
    service = JinaAIService()
    
    test_urls = [
        "https://www.example.com",
        "https://www.wikipedia.org/wiki/Artificial_intelligence"
    ]
    
    print(f"Testing Jina AI extraction with {len(test_urls)} URLs")
    
    try:
        results = service.extract_key_info(test_urls, query="artificial intelligence")
        print(f"\nExtracted content from {results['total_extracted']} URLs:")
        for i, content in enumerate(results['extracted_content'], 1):
            print(f"\n{i}. URL: {content['url']}")
            print(f"   Status: {content['status']}")
            print(f"   Content preview: {content['key_information'][:200]}...")
        
        if results['failed_urls']:
            print(f"\nFailed URLs: {results['failed_urls']}")
    except Exception as e:
        print(f"Error: {e}")
