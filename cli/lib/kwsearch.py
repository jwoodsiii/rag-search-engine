import string

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    has_matching_token,
    load_movies,
    tokenize,
)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokenize(query)
        title_tokens = tokenize(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results
