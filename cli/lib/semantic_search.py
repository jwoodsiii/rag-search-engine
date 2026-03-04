import os
import re
import string

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents: list[dict]
        self.document_map = dict()
        self.embedding_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.model.encode([query])[0]
        scores = [cosine_similarity(query_embedding, emb) for emb in self.embeddings]
        top_indices = np.argsort(scores)[::-1][:limit]
        return [
            {
                "score": scores[i],
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"],
            }
            for i in top_indices
        ]

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.embedding_path):
            self.embeddings = np.load(self.embedding_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
            else:
                return self.build_embeddings(documents)
        else:
            return self.build_embeddings(documents)

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        str_docs = list()
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            str_docs.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(str_docs, show_progress_bar=True)
        np.save(os.path.join(CACHE_DIR, "movie_embeddings.npy"), self.embeddings)
        return self.embeddings

    def generate_embedding(self, text: str):
        if text == "" or text is string.whitespace:
            raise ValueError("Text cannot be empty or None")
        tlist = list()
        tlist.append(text)
        print(f"DEBUG: tlist = {tlist}")
        return self.model.encode(tlist)[0]


def search(query, limit):
    model = SemanticSearch()
    movies = load_movies()
    model.load_or_create_embeddings(movies)
    output = model.search(query, limit)
    for i in output:
        print(
            f"{i.get('title')} (score: {i.get('score')})\n{i['description'][:100]}..."
        )


def semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    output = list()
    i = 0
    # print(sentences)
    chunk = ""
    while i < len(sentences):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if output and len(chunk_sentences) <= overlap:
            break
        chunk = " ".join(sentences[i : i + max_chunk_size])
        output.append(chunk)
        step = max_chunk_size - overlap
        if step <= 0:
            raise ValueError("overlap must be less than max_chunk_size")
        i += step

    return output


def chunk_text(text: str, chunk_size: int, overlap: int) -> None:
    split = text.split()
    i = 0
    chunks = list()
    print(f"Chunking {len(text)} characters")
    while i < len(split):
        nxt = " ".join(split[i : i + chunk_size])
        if len(split[i : i + chunk_size]) <= overlap:
            break
        chunks.append(nxt)
        if overlap > 0:
            i += chunk_size - overlap
        else:
            i += chunk_size
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_embeddings():
    model = SemanticSearch()
    movies = load_movies()
    model.load_or_create_embeddings(movies)
    # print(f"Embeddings: {model.embeddings}")
    print(f"Number of docs:   {len(model.documents)}")
    print(
        f"Embeddings shape: {model.embeddings.shape[0]} vectors in {model.embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str) -> None:
    model = SemanticSearch()
    emb = model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {emb[:5]}")
    print(f"Shape: {emb.shape}")


def embed_text(text: str):
    model = SemanticSearch()
    emb = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {emb[:3]}")
    print(f"Dimensions: {emb.shape[0]}")


def verify_model() -> None:
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")
