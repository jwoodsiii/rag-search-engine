import os
import string

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = dict()
        self.embedding_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

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


def verify_embeddings():
    model = SemanticSearch()
    movies = load_movies()
    model.load_or_create_embeddings(movies)
    # print(f"Embeddings: {model.embeddings}")
    print(f"Number of docs:   {len(model.documents)}")
    print(
        f"Embeddings shape: {model.embeddings.shape[0]} vectors in {model.embeddings.shape[1]} dimensions"
    )


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
