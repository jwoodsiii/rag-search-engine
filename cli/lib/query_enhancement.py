import os
from typing import Optional

from dotenv import load_dotenv
from google import genai

from .search_utils import get_gemini_client

client = get_gemini_client()
model = "gemma-3-27b-it"


def enhance_query(query: str, method: str) -> dict:
    client = get_gemini_client()
    match method:
        case "spell":
            resp = client.models.generate_content(
                model=model,
                contents=f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{query}"
                """,
            )
            return {
                "method": method,
                "query": query,
                "enhanced_query": resp.text.strip(),
            }
        case "rewrite":
            resp = client.models.generate_content(
                model=model,
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
        case "expand":
            resp = client.models.generate_content(
                model="gemma-3-27b-it",
                contents=f"""Expand the user-provided movie search query below with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.

                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                User query: "{query}"
                """,
            )
            return {
                "method": method,
                "query": query,
                "enhanced_query": f"{query} {resp.text.strip()} i.q.",
            }
