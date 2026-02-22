import string

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    query = query.translate(str.maketrans("", "", string.punctuation))
    for movie in movies:
        if (
            query.lower()
            in movie["title"]
            .translate(str.maketrans("", "", string.punctuation))
            .lower()
        ):
            results.append(movie)
            if len(results) >= limit:
                break
    return results
