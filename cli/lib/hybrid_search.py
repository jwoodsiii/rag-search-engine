import os

from transformers.core_model_loading import Chunk

from .chunked_semantic_search import ChunkedSemanticSearch
from .kwsearch import InvertedIndex


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> dict[int, float]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha, limit: int = 5) -> None:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k, limit: int = 10) -> None:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize(scores: list) -> list:
    if scores == []:
        return scores
    if min(scores) == max(scores):
        return [1.0]
    min_score = min(scores)
    max_score = max(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]
