import requests
from typing import List, Dict, Any, Optional
import time


class TinyfishService:
    def __init__(self, api_key: str, base_url: str = "https://api.tinyfish.ai/v1", 
                 max_results: int = 20, timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url
        self.max_results = max_results
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    def search(self, query: str, source_types: List[str] = None) -> Dict[str, Any]:
        payload = {
            "query": query,
            "max_results": self.max_results,
            "source_types": source_types or ["web", "academic", "news"],
            "include_summaries": True
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/search",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "results": []}

    def scrape_url(self, url: str, extract_content: bool = True) -> Dict[str, Any]:
        payload = {
            "url": url,
            "extract_content": extract_content,
            "include_metadata": True
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/scrape",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "content": ""}

    def batch_search(self, queries: List[str], delay: float = 1.0) -> List[Dict[str, Any]]:
        results = []
        for query in queries:
            result = self.search(query)
            results.append({
                "query": query,
                "data": result
            })
            if delay > 0 and query != queries[-1]:
                time.sleep(delay)
        return results

    def research_field(self, field_name: str, include_publications: bool = True,
                      include_institutions: bool = True) -> str:
        queries = [
            f"{field_name} recent research developments 2024",
            f"{field_name} major debates controversies",
            f"{field_name} key publications conferences"
        ]
        
        if include_institutions:
            queries.append(f"{field_name} leading institutions research groups")
        
        if include_publications:
            queries.append(f"{field_name} top journals conferences")
        
        batch_results = self.batch_search(queries)
        
        combined_data = []
        for result in batch_results:
            query = result["query"]
            data = result["data"]
            combined_data.append(f"=== Query: {query} ===")
            if "results" in data:
                for item in data["results"][:5]:
                    combined_data.append(f"Title: {item.get('title', 'N/A')}")
                    combined_data.append(f"Summary: {item.get('summary', item.get('snippet', 'N/A'))}")
                    combined_data.append(f"URL: {item.get('url', 'N/A')}")
                    combined_data.append("")
        
        return "\n".join(combined_data)

    def get_academic_sources(self, topic: str, max_results: int = 10) -> List[Dict[str, Any]]:
        result = self.search(
            topic,
            source_types=["academic", "scholar"]
        )
        
        sources = []
        if "results" in result:
            for item in result["results"][:max_results]:
                sources.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", ""),
                    "summary": item.get("summary", item.get("snippet", "")),
                    "date": item.get("date", "")
                })
        
        return sources

    def get_current_discussions(self, topic: str) -> str:
        result = self.search(
            topic,
            source_types=["news", "social", "blog"]
        )
        
        discussions = []
        if "results" in result:
            for item in result["results"][:10]:
                discussions.append(f"- {item.get('title', 'N/A')}: {item.get('summary', item.get('snippet', 'N/A'))}")
        
        return "\n".join(discussions) if discussions else "No recent discussions found."
