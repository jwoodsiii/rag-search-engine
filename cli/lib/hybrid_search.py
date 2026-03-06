import os

import numpy as np
from google import genai

from .chunked_semantic_search import ChunkedSemanticSearch
from .kwsearch import InvertedIndex
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_K,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    get_gemini_client,
    load_movies,
)

LIMIT_SCALE = 500


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> dict[int, float]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha, limit: int = 5):
        bm_results = self._bm25_search(query, limit * LIMIT_SCALE)
        # # print(bm_results)
        # doc_map = {doc["id"]: doc for doc in self.documents}
        semantic_results = self.semantic_search.search_chunks(
            query, limit * LIMIT_SCALE
        )
        combined = combine_search_results(bm_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query: str, k, limit: int = 10) -> None:
        bm_results = self._bm25_search(query, limit * LIMIT_SCALE)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * LIMIT_SCALE
        )
        """
            Combine results using rrf
                - create dictionary mapping doc_ids to full documents and bm25 and semantic ranks
                - for each document, calc rrf_score
                - add score to each doc (if doc is in both searches sum rrf score)
                - return results sorted by rrf score in descending order
        """
        combined = rrf_rank(bm_results, semantic_results, k)
        return combined[:limit]


def normalize(scores: list) -> list:
    if scores == []:
        return scores

    min_score = min(scores)
    max_score = max(scores)
    if min(scores) == max(scores):
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank, k=DEFAULT_K):
    return 1 / (k + rank)


def weighted_search(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    movies = load_movies()
    hs = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = hs.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }


def rrf_search(
    query: str, k: int = DEFAULT_K, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    movies = load_movies()
    hs = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = hs.rrf_search(query, k, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "k": k,
        "results": results,
    }


def rrf_rank(
    bm25_results: list[dict], semantic_results: list[dict], k: int = DEFAULT_K
) -> list[dict]:
    combined_rank = {}

    for i, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id not in combined_rank:
            combined_rank[doc_id] = {
                "document": result["document"],
                "title": result["title"],
                "bm25_rank": i,
                "semantic_rank": None,
                "rrf_score": 0.0,
            }
        doc = combined_rank[doc_id]
        if doc.get("bm25_rank") is None:
            doc["bm25_rank"] = i
            doc["rrf_score"] += rrf_score(i, DEFAULT_K)

        if doc.get("bm25_rank") and doc.get("semantic_rank"):
            doc["rrf_score"] += rrf_score(doc["bm25_rank"], DEFAULT_K) + rrf_score(
                doc["semantic_rank"], DEFAULT_K
            )

    # print(semantic_results)
    for i, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in combined_rank:
            combined_rank[doc_id] = {
                "document": result["document"],
                "title": result["title"],
                "bm25_rank": None,
                "semantic_rank": i,
                "rrf_score": 0.0,
            }

        doc = combined_rank[doc_id]
        # print(f"semantic_results doc: {doc}")
        if doc.get("semantic_rank") is None:
            doc["semantic_rank"] = i
            doc["rrf_score"] += rrf_score(i, DEFAULT_K)

        if doc.get("bm25_rank") and doc.get("semantic_rank"):
            doc["rrf_score"] += rrf_score(doc["bm25_rank"], k) + rrf_score(
                doc["semantic_rank"], k
            )

    hybrid_ranks = []
    for doc_id, data in combined_rank.items():
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
            rrf_score=data["rrf_score"],
        )
        hybrid_ranks.append(result)

    return sorted(hybrid_ranks, key=lambda x: x["score"], reverse=True)


def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], alpha: float = DEFAULT_ALPHA
) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def enhance_query(query: str, method: str) -> dict:
    client = get_gemini_client()
    match method:
        case "spell":
            resp = client.models.generate_content(
                model="gemma-3-27b-it",
                contents=f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{query}"
                """,
            )
            return {"method": method, "query": query, "enhanced_query": resp.text}
        case "rewrite":
            resp = client.models.generate_content(
                model="gemma-3-27b-it",
                contents=f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query}"
                """,
            )
            return {
                "method": method,
                "query": query,
                "enhanced_query": resp.text.strip(),
            }
