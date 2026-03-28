import requests
from typing import Any, Dict, List, Optional


class TinyfishService:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://agent.tinyfish.ai/v1",
        timeout: int = 60,
        browser_profile: str = "lite",
        proxy_config: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.browser_profile = browser_profile
        self.proxy_config = proxy_config or {}
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        })

    def run_extraction(self, url: str, goal: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "url": url,
            "goal": goal,
            "browser_profile": self.browser_profile,
        }

        if self.proxy_config:
            payload["proxy_config"] = self.proxy_config

        response = self.session.post(
            f"{self.base_url}/automation/run",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def extract_from_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        extracted = []
        for source in sources:
            url = source.get("url", "")
            goal = source.get("extraction_goal", "")
            if not url or not goal:
                extracted.append({
                    "title": source.get("title", "Unknown source"),
                    "url": url,
                    "goal": goal,
                    "error": "Missing source URL or extraction goal.",
                    "result": None,
                })
                continue

            try:
                run = self.run_extraction(url, goal)
                extracted.append({
                    "title": source.get("title", "Unknown source"),
                    "url": url,
                    "rationale": source.get("rationale", ""),
                    "goal": goal,
                    "status": run.get("status"),
                    "result": run.get("result"),
                    "error": run.get("error"),
                })
            except requests.exceptions.RequestException as exc:
                extracted.append({
                    "title": source.get("title", "Unknown source"),
                    "url": url,
                    "rationale": source.get("rationale", ""),
                    "goal": goal,
                    "status": "FAILED",
                    "result": None,
                    "error": str(exc),
                })

        return extracted

    def build_research_corpus(self, sources: List[Dict[str, Any]], extracted: List[Dict[str, Any]]) -> str:
        corpus: List[str] = []
        for source, extraction in zip(sources, extracted):
            corpus.append(f"=== Source: {source.get('title', 'Unknown source')} ===")
            corpus.append(f"URL: {source.get('url', 'N/A')}")
            if source.get("rationale"):
                corpus.append(f"Why it was chosen: {source['rationale']}")
            corpus.append(f"Extraction goal: {source.get('extraction_goal', 'N/A')}")
            corpus.append(f"Status: {extraction.get('status', 'UNKNOWN')}")
            if extraction.get("error"):
                corpus.append(f"Error: {extraction['error']}")
            if extraction.get("result") is not None:
                corpus.append(f"Extracted result: {extraction['result']}")
            corpus.append("")
        return "\n".join(corpus)
