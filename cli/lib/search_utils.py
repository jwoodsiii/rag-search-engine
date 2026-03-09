import json
import os
import string

from dotenv import load_dotenv
from google import genai
from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3
DEFAULT_ALPHA = 0.5
DEFAULT_K = 60
SEARCH_MULTIPLIER = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
GOLDEN_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


def get_gemini_client() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)

    return client


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        data = f.read().splitlines()
    return data


def load_golden_dataset() -> dict:
    with open(GOLDEN_PATH, "r") as f:
        return json.load(f)


def tokenize(line: str) -> list[str]:
    stemmer = PorterStemmer()
    text = preprocess_text(line)
    stopwords = load_stopwords()
    tmp = remove_stopwords(text.split(), stopwords)
    return [stemmer.stem(tok) for tok in tmp]


def remove_stopwords(tokens: list[str], stopwords: list[str]) -> list[str]:
    return [t for t in tokens if t not in stopwords]


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    stemmer = PorterStemmer()
    for query_token in query_tokens:
        for title_token in title_tokens:
            if stemmer.stem(query_token) in stemmer.stem(title_token):
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def format_search_result(
    doc_id: str, title: str, document: str, score: float = 0.0, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }


# def tokenize_text(text: str) -> list[str]:
#     text = preprocess_text(text)
#     tokens = text.split()
#     valid_tokens = []
#     for token in tokens:
#         if token:
#             valid_tokens.append(token)
#     return valid_tokens
