import argparse
import json

from lib.hybrid_search import (
    rrf_search,
)
from lib.search_utils import DEFAULT_K, GOLDEN_PATH


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open(GOLDEN_PATH, "r") as f:
        data = json.load(f)
        print(f"k={limit}")
        # print(data)
        test_cases = data.get("test_cases", "")
        if test_cases == "":
            raise ValueError("No test cases found in golden file")
        for test in test_cases:
            query = test.get("query", "")
            print(
                f"Query: {query}\nRelevant Documents: {', '.join(test.get('relevant_docs', []))}"
            )
            res = rrf_search(query, DEFAULT_K, limit=limit)
            retrv = []
            for r in res["results"]:
                retrv.append(r.get("title", ""))
            print(f"- Query: {test.get('query', '')}")
            print(
                f"    - Precision@{limit}: {float(len(set(retrv).intersection(set(test.get('relevant_docs', [])))) / len(retrv)):.4f}"
            )
            print(f"    - Retrieved: {', '.join(retrv)}")
            print(f"    - Relevant: {', '.join(test.get('relevant_docs', []))}")


if __name__ == "__main__":
    main()
