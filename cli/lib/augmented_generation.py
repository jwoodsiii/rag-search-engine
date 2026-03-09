import json

from .hybrid_search import HybridSearch
from .search_utils import (
    DEFAULT_K,
    DEFAULT_SEARCH_LIMIT,
    get_gemini_client,
    load_movies,
)

client = get_gemini_client()
model = "gemma-3-27b-it"


def generate_answer(query: str, search_results, limit: int = 5) -> str:
    context = ""
    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    resp = client.models.generate_content(
        model=model,
        contents=f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {context}

        Provide a comprehensive answer that addresses the query:""",
    )
    ans_text = (resp.text or "").strip()
    return ans_text


def generate_summary(query: str, search_results, limit: int = 5) -> str:
    context = ""
    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    resp = client.models.generate_content(
        model=model,
        contents=f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
        Query: {query}
        Search Results:
        {context}
        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
        """,
    )
    ans_text = (resp.text or "").strip()
    return ans_text


def rag(query: str, gen_func, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hs = HybridSearch(movies)

    search_results = hs.rrf_search(query, k=DEFAULT_K, limit=limit)
    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    answer = gen_func(query, search_results)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "answer": answer,
    }


def rag_command(query, gen_func):
    return rag(query, gen_func)
