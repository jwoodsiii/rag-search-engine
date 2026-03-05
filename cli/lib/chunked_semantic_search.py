import json
import os

import numpy as np

from .search_utils import CACHE_DIR, SCORE_PRECISION, load_movies
from .semantic_search import SemanticSearch, cosine_similarity, semantic_chunk

MAX_CHUNK_SIZE = 4
OVERLAP_LIMIT = 1


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        chunks = list()
        metadata = list()
        for i, doc in enumerate(self.documents):
            self.document_map[doc["id"]] = doc
            if not doc["description"] or not doc["description"].strip():
                continue
            tmp_chunks = semantic_chunk(
                doc["description"], MAX_CHUNK_SIZE, OVERLAP_LIMIT
            )
            for j, chunk in enumerate(tmp_chunks):
                chunks.append(chunk)
                metadata.append(
                    {
                        "movie_idx": i,
                        "chunk_idx": j,
                        "total_chunks": len(tmp_chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = metadata
        np.save(self.chunk_path, self.chunk_embeddings)
        with open(self.metadata_path, "w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(chunks)},
                f,
                indent=2,
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.chunk_path) and os.path.exists(self.metadata_path):
            self.chunk_embeddings = np.load(self.chunk_path)
            self.chunk_metadata = json.load(open(self.metadata_path))["chunks"]
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list:
        query_embedding = self.generate_embedding(query)
        chunk_score = list()
        # print(type(self.chunk_metadata))
        # print(type(self.chunk_metadata[0]), self.chunk_metadata[0])
        for i, chunk in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk)
            chunk_score.append(
                {
                    "chunk_idx": i,
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": score,
                }
            )
        movie_map = dict()
        for score in chunk_score:
            if movie_map.get(score["movie_idx"]) is None or score[
                "score"
            ] > movie_map.get(score["movie_idx"]):
                movie_map[score["movie_idx"]] = score["score"]
        sorted_movies = sorted(movie_map.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]
        output = list()
        for mov_idx, score in sorted_movies:
            output.append(
                {
                    "id": self.documents[mov_idx]["id"],
                    "title": self.documents[mov_idx]["title"],
                    "document": self.documents[mov_idx]["description"][:100],
                    "score": round(score, SCORE_PRECISION),
                    "metadata": self.documents[mov_idx].get("metadata", {}),
                }
            )
        return output


def search_chunked(query: str, limit: int = 5):
    movies = load_movies()
    cs = ChunkedSemanticSearch()
    embeddings = cs.load_or_create_chunk_embeddings(movies)
    return cs.search_chunks(query, limit)


def embed_chunks() -> None:
    movies = load_movies()
    cs = ChunkedSemanticSearch()
    embeddings = cs.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")
