import json

from .search_utils import get_gemini_client

client = get_gemini_client()
model = "gemma-3-27b-it"


def generate_answer(query: str, results: list[str]) -> str:
    resp = client.models.generate_content(
        model=model,
        contents=f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {", ".join(results)}

        Provide a comprehensive answer that addresses the query:""",
    )
    ans_text = (resp.text or "").strip()
    return ans_text
