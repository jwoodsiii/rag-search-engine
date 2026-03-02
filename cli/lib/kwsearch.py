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

BM25_K1 = 1.5
BM25_B = 0.75


class InvertedIndex:
    def __init__(
        self,
    ) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        docs = tokenize(text)
        for tok in set(docs):
            self.index[tok].add(doc_id)
        self.term_frequencies[doc_id].update(docs)
        self.doc_lengths[doc_id] = len(docs)

    def __normalize_term(self, term: str) -> list[str]:
        tok = tokenize(term)
        if len(tok) == 0:
            raise ValueError("Term must contain at least 1 token")
        return tok

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        total_length = 0
        for length in self.doc_lengths.values():
            total_length += length
        return total_length / len(self.doc_lengths)

    def bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> dict[int, float]:
        tokens = tokenize(query)
        scores = dict()
        for doc_id in self.doc_lengths.keys():
            doc_score = 0
            for tok in tokens:
                doc_score += self.bm25(doc_id, tok)
            scores[doc_id] = doc_score
        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        return dict(list(scores.items())[:limit])

    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term, BM25_K1, BM25_B)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1

        score = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        # print(
        #     f'tf: {tf}\ndoc_length" {doc_length}\navg_doc_len: {avg_doc_length}\nlength_norm: {length_norm}\nscore: {score}'
        # )
        return score

    def get_bm25_idf(self, term: str) -> float:
        tok = self.__normalize_term(term)
        if len(tok) != 1:
            raise ValueError("Term must be a single token")
        doc_count = len(self.docmap)
        term_count = len(self.get_documents(tok[0]))
        return math.log((doc_count - term_count + 0.5) / (term_count + 0.5) + 1)

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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

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
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Index file not found at {self.index_path}")


def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file not found")
        sys.exit(1)
    lmt = 1
    for doc_id, score in idx.bm25_search(query, limit).items():
        if lmt >= limit:
            return
        print(
            f"{lmt}. ({doc_id}) {idx.docmap[doc_id].get('title')} - Score: {score:.2f}"
        )
        lmt += 1


def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file not found")
        sys.exit(1)
    return idx.get_bm25_tf(doc_id, term, k1)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file not found")
        sys.exit(1)
    return idx.get_bm25_idf(term)


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
