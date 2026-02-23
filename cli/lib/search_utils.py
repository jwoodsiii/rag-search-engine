import json
import os
import string

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        data = f.read().splitlines()
    return data


def tokenize(line: str) -> list[str]:
    text = preprocess_text(line)
    stopwords = load_stopwords()
    return remove_stopwords([tok for tok in text.split() if tok], stopwords)


def remove_stopwords(text: list[str], stopwords: list[str]) -> list[str]:
    for w in text:
        if w in stopwords:
            text.remove(w)
    return text


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# def tokenize_text(text: str) -> list[str]:
#     text = preprocess_text(text)
#     tokens = text.split()
#     valid_tokens = []
#     for token in tokens:
#         if token:
#             valid_tokens.append(token)
#     return valid_tokens
