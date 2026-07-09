"""
Tavily web search service for the Research Gap Discovery Agent.

This module provides web search capabilities using the Tavily API,
which is used when the oss120B model needs additional context beyond
the PDF documents.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)


class TavilyService:
    """Service for performing web searches using Tavily API."""

    def __init__(self, api_keys_path: str = "api_keys.txt"):
        """Initialize Tavily service with API key from file."""
        self._api_key = self._read_tavily_api_key(api_keys_path)
        self._base_url = "https://api.tavily.com/search"

    def _read_tavily_api_key(self, api_keys_path: str) -> str:
        """Read Tavily API key from api_keys.txt."""
        path = Path(api_keys_path)
        if not path.exists():
            raise RuntimeError(
                f"Missing API keys file: {api_keys_path}. "
                "Expected a line like 'Tavily API key: <API_KEY>'."
            )

        content = path.read_text(encoding="utf-8", errors="ignore")
        
        # Try multiple patterns for flexibility
        patterns = [
            r"Tavily\s+API\s+key\s*:\s*([A-Za-z0-9_\-]+)",
            r"tavily\s+api\s+key\s*:\s*([A-Za-z0-9_\-]+)",
            r"tvly-([A-Za-z0-9_\-]+)"
        ]
        
        for pattern in patterns:
            m = re.search(pattern, content, flags=re.IGNORECASE)
            if m:
                api_key = m.group(1).strip()
                if api_key:
                    # Ensure it has the tvly prefix if not present
                    if not api_key.startswith("tvly-"):
                        api_key = f"tvly-{api_key}"
                    return api_key

        raise RuntimeError(
            f"Unable to find Tavily API key in {api_keys_path}. "
            "Expected a line like 'Tavily API key: <API_KEY>'."
        )

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a web search using Tavily API.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return (default: 5).
            search_depth: Search depth - "basic" or "advanced" (default: "basic").
            include_domains: List of domains to include in search.
            exclude_domains: List of domains to exclude from search.

        Returns:
            Dictionary containing search results with the following structure:
            {
                "query": str,
                "results": [
                    {
                        "title": str,
                        "url": str,
                        "content": str,
                        "score": float,
                        "published_date": Optional[str]
                    },
                    ...
                ],
                "total_results": int
            }

        Raises:
            RuntimeError: If API call fails or returns error.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        logger.info("Tavily search - query: %s, max_results: %s", query, max_results)

        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": True,  # Get Tavily's AI-generated answer as fallback
            "include_raw_content": True,  # Get raw content for better Jina processing
            "include_images": False,
            "include_image_descriptions": False,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        try:
            response = requests.post(
                self._base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # Extract and format results
            results = []
            raw_results = data.get("results", [])
            
            for result in raw_results:
                formatted_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "published_date": result.get("published_date"),
                }
                results.append(formatted_result)

            # Also capture Tavily's AI answer if available
            tavily_answer = data.get("answer", "")

            logger.info(
                "Tavily search completed - results: %d, has_answer: %s, query: %s",
                len(results),
                bool(tavily_answer),
                query
            )

            return {
                "query": query,
                "results": results,
                "total_results": len(results),
                "answer": tavily_answer,  # Include Tavily's AI-generated answer
            }

        except requests.exceptions.Timeout:
            logger.error("Tavily search timeout for query: %s", query)
            raise RuntimeError(f"Tavily search timeout for query: {query}")
        except requests.exceptions.RequestException as exc:
            logger.error("Tavily search failed: %s", exc)
            raise RuntimeError(f"Tavily search failed: {exc}")
        except Exception as exc:
            logger.exception("Unexpected error in Tavily search")
            raise RuntimeError(f"Tavily search error: {exc}") from exc


if __name__ == "__main__":
    # Self-test
    logging.basicConfig(level=logging.INFO)
    
    service = TavilyService()
    
    test_query = "latest research in artificial intelligence"
    print(f"Testing Tavily search with query: {test_query}")
    
    try:
        results = service.search(test_query, max_results=3)
        print(f"\nFound {results['total_results']} results:")
        for i, result in enumerate(results['results'], 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Score: {result['score']}")
            print(f"   Content: {result['content'][:200]}...")
    except Exception as e:
        print(f"Error: {e}")
