# import string

import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    has_matching_token,
    load_movies,
    tokenize,
)


class InvertedIndex:
    def __init__(
        self,
    ) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        docs = tokenize(text)
        for tok in docs:
            self.index[tok].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        ids = self.index.get(term.lower(), set())
        return sorted(list(ids))

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            # doc_id = movie['id']
            # doc_desc = f'{movie["title"]} {movie["description"]}'
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token `merida` = {docs[0]}")


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
