"""
agents/literature_ingest.py
============================
Weekly Literature Ingestion Agent

Role:
  Pull the latest arXiv papers on SSM / Liquid Neural Networks / Forward-Forward
  learning via the free Semantic Scholar API and feed summaries into Agent 1's
  prompt context (evolutionary_memory.json).

Algorithm:
  1. Query Semantic Scholar API for recent papers matching keywords:
     "state space model", "liquid neural network", "forward-forward learning",
     "mamba", "selective state space".
  2. Extract title, abstract snippet, year, citation count.
  3. Store top-N papers (by relevance) in evolutionary_memory.json under
     a "literature" key.
  4. Agent 1 can then incorporate these insights into its architecture proposals.

This runs weekly (every Sunday) via the literature-weekly.yml workflow.

API: Semantic Scholar Academic Graph API (free, no key required for basic use).
"""

import os
import sys
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MEMORY_PATH = "evolutionary_memory.json"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# Keywords to search for (relevant to CDLE architecture)
SEARCH_QUERIES = [
    "state space model sequence modelling",
    "liquid neural network time constant",
    "forward-forward learning algorithm",
    "mamba selective state space",
    "neural architecture search efficient",
]

MAX_PAPERS_PER_QUERY = 3
MAX_TOTAL_PAPERS = 10


def load_json(path: str) -> dict:
    """Load a JSON file, returning empty dict if missing."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(data: dict, path: str) -> None:
    """Save a dict as a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved → {path}")


def search_papers(query: str, limit: int = 3) -> list[dict]:
    """
    Search Semantic Scholar for papers matching a query.

    Args:
        query: Search string.
        limit: Maximum number of results per query.

    Returns:
        List of paper dicts with title, abstract, year, citationCount.
    """
    try:
        import requests
    except ImportError:
        log.warning("requests library not installed. Skipping literature search.")
        return []

    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,citationCount,url",
        "year": "2024-2026",  # Focus on recent papers
    }

    try:
        response = requests.get(
            SEMANTIC_SCHOLAR_API,
            params=params,
            timeout=30,
            headers={"User-Agent": "EvoArchitect/1.0 (research)"},
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        log.warning(f"Semantic Scholar API error for query '{query}': {e}")
        return []

    papers = []
    for paper in data.get("data", []):
        abstract = paper.get("abstract", "") or ""
        papers.append({
            "title": paper.get("title", "Unknown"),
            "abstract_snippet": abstract[:300] + "..." if len(abstract) > 300 else abstract,
            "year": paper.get("year"),
            "citations": paper.get("citationCount", 0),
            "url": paper.get("url", ""),
        })

    return papers


def ingest_literature() -> list[dict]:
    """
    Run all search queries and collect papers.

    Returns:
        Deduplicated list of paper dicts, sorted by citation count.
    """
    all_papers = []
    seen_titles = set()

    for query in SEARCH_QUERIES:
        log.info(f"Searching: '{query}'")
        papers = search_papers(query, limit=MAX_PAPERS_PER_QUERY)
        for p in papers:
            title_lower = p["title"].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                all_papers.append(p)

        # Be polite to the free API — 1 second between queries
        time.sleep(1)

    # Sort by citation count (highest first) and keep top N
    all_papers.sort(key=lambda p: p.get("citations", 0), reverse=True)
    all_papers = all_papers[:MAX_TOTAL_PAPERS]

    log.info(f"Collected {len(all_papers)} unique papers.")
    return all_papers


def main():
    log.info("=== Literature Ingestion Agent ===")

    # Collect papers
    papers = ingest_literature()

    if not papers:
        log.info("No papers found. Skipping memory update.")
        return

    # Update evolutionary memory
    memory = load_json(MEMORY_PATH)
    memory["literature"] = {
        "last_updated": time.strftime("%Y-%m-%d"),
        "papers": papers,
        "summary": (
            f"Found {len(papers)} recent papers on SSM, Liquid NNs, and FF learning. "
            f"Top paper: '{papers[0]['title']}' ({papers[0].get('citations', 0)} citations)."
        ),
    }

    save_json(memory, MEMORY_PATH)

    # Print summary
    log.info("=== Literature Summary ===")
    for i, p in enumerate(papers, 1):
        log.info(
            f"  {i}. [{p.get('year', '?')}] {p['title']} "
            f"(citations: {p.get('citations', 0)})"
        )
    log.info("=== Literature Ingestion complete. ===")


if __name__ == "__main__":
    main()
