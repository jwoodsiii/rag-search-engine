# import string

import math
import os
import pickle
import sys
from collections import Counter, defaultdict
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
        self.term_frequencies = defaultdict(Counter)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        docs = tokenize(text)
        ctr = Counter(docs)
        for k in ctr:
            self.index[k].add(doc_id)
            self.term_frequencies[doc_id] = ctr
        for tok in set(docs):
            self.index[tok].add(doc_id)
            self.term_frequencies[doc_id][tok] + 1

    def __normalize_term(self, term: str) -> list[str]:
        tok = tokenize(term)
        if len(tok) == 0:
            raise ValueError("Term must contain at least 1 token")
        return tok

    def get_documents(self, term: str) -> list[int]:
        ids = self.index.get(term, set())
        return sorted(list(ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        tok = tokenize(term)
        if len(tok) != 1:
            raise ValueError("Term must be a single token")
        # tok = self.__normalize_term(term)
        return self.term_frequencies[doc_id][tok[0]]

    def get_idf(self, term: str) -> float:
        # calculate idf, first get total doc count
        tok = tokenize(term)
        if len(tok) != 1:
            raise ValueError("Term must be a single token")
        doc_count = len(self.docmap)
        # print(f"docmap len: {len(self.docmap)}")
        # now get the documents that the term appears in
        appearances = self.get_documents(tok[0])
        # print(f"Get documents result: {appearances}")
        # return idf
        return math.log((doc_count + 1) / (len(appearances) + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tok = tokenize(term)
        if len(tok) != 1:
            raise ValueError("Term must be a single token")
        return self.get_tf(doc_id, tok[0]) * self.get_idf(tok[0])

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
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                idx = pickle.load(f)
                self.index = idx
            with open(self.docmap_path, "rb") as f:
                docmap = pickle.load(f)
                self.docmap = docmap
            with open(self.tf_path, "rb") as f:
                tf = pickle.load(f)
                self.term_frequencies = tf
        except FileNotFoundError:
            raise FileNotFoundError(f"Index file not found at {self.index_path}")


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file not found")
        sys.exit(1)
    return idx.get_tfidf(doc_id, term.lower())


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file not found")
        sys.exit(1)
    return idx.get_idf(term.lower())


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file not found")
        sys.exit(1)

    return idx.get_tf(doc_id, term)


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
