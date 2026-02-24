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
        for tok in set(docs):
            self.index[tok].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        ids = self.index.get(term, set())
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

    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                idx = pickle.load(f)
                self.index = idx
            with open(self.docmap_path, "rb") as f:
                docmap = pickle.load(f)
                self.docmap = docmap
        except FileNotFoundError:
            raise FileNotFoundError(f"Index file not found at {self.index_path}")


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    # docs = idx.get_documents("merida")
    # print(f"First document for token `merida` = {docs[0]}")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file not found")
        sys.exit(1)

    results = []
    processed = set()
    # print(f"idx: {idx.index.get('1')}")
    # print(f"docmap: {idx.docmap.get(3586)}")
    for token in tokenize(query):
        # print(
        #     "LOOKUP:",
        #     repr(token),
        #     "exists_in_index?",
        #     token in idx.index,
        #     "lower_exists?",
        #     token.lower() in idx.index,
        # )
        docs = idx.get_documents(token)
        # print("DOC COUNT:", len(docs), "FIRST:", docs[:5])
        for id in docs:
            if id in processed:
                continue
            if len(processed) >= limit:
                return results
            mov = idx.docmap.get(id)
            if mov is None:
                continue
            results.append(mov)
            processed.add(id)
    return results
