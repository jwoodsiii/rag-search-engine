import argparse

from lib.hybrid_search import normalize, weighted_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="Scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Run weighted search with given query"
    )
    weighted_search_parser.add_argument("query", help="Query to search")
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha value for weighted search"
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="Limit search results"
    )

    args = parser.parse_args()

    match args.command:
        case "weighted-search":
            result = weighted_search(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()

        case "normalize":
            print("Normalizing scores...")
            output = normalize(args.scores)
            if output == []:
                print()
            else:
                for i in output:
                    print(f"* {i:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
