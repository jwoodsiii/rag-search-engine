import json
import os
import time

from dotenv import load_dotenv
from google import genai

from .search_utils import get_gemini_client

client = get_gemini_client()
model = "gemma-3-27b-it"


def llm_rerank_individual(
    query: str, results: list[dict], limit: int = 5
) -> list[dict]:
    output = []
    for doc in results:
        resp = client.models.generate_content(
            model=model,
            contents=f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Output ONLY the number in your response, no other text or explanation.

            Score:""",
        )

        score_text = (resp.text or "").strip()
        score = int(score_text)

        output.append({**doc, "score": score})
        time.sleep(4)
    return sorted(output, key=lambda x: float(x["score"]), reverse=True)[:limit]


def llm_rerank_batch(query: str, results: list[dict], limit: int = 5) -> list[dict]:
    if not results:
        return []

    doc_list_str = "\n".join(
        [
            f"ID: {doc.get('id', '')}, Title: {doc.get('title', '')}, Document: {doc.get('document', '')[:500]}"
            for doc in results
        ]
    )

    movie_lookup = {doc["id"]: doc for doc in results}
    # print(movie_lookup)

    resp = client.models.generate_content(
        model=model,
        contents=f"""Rank the movies listed below by relevance to the following search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

        For example:
        [75, 12, 34, 2, 1]

        Ranking:""",
    )
    ranking_text = (resp.text or "").strip()
    parsed_ids = json.loads(ranking_text)

    output = []
    for rank, id in enumerate(parsed_ids, start=1):
        if id in movie_lookup:
            output.append({**movie_lookup[id], "score": rank})

    return output[:limit]


def rerank(query: str, results: list[dict], method: str, limit: int = 5) -> list[dict]:
    match method:
        case "individual":
            return llm_rerank_individual(query, results, limit)
        case "batch":
            return llm_rerank_batch(query, results, limit)
        case _:
            return []
