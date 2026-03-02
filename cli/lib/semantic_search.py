import string

from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str):
        if text == "" or text is string.whitespace:
            raise ValueError("Text cannot be empty or None")
        tlist = list()
        tlist.append(text)
        print(f"DEBUG: tlist = {tlist}")
        return self.model.encode(tlist)[0]


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
