import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_utils import load_movies
from .semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in self.documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def search_with_image(self, image_path: str):
        img_embedding = self.embed_image(image_path)
        scores = [cosine_similarity(img_embedding, emb) for emb in self.text_embeddings]
        top_indices = np.argsort(scores)[::-1][:5]

        return [
            {
                "id": self.documents[i]["id"],
                "score": scores[i],
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"],
            }
            for i in top_indices
        ]

    def embed_image(self, image_path: str) -> None:
        img = Image.open(image_path)
        return self.model.encode([img])[0]


def verify_image_embedding(image_path: str) -> None:
    movies = load_movies()
    ms = MultimodalSearch(documents=movies)
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str):
    movies = load_movies()
    ms = MultimodalSearch(documents=movies)
    return ms.search_with_image(image_path)
