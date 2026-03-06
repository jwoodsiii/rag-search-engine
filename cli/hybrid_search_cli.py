import argparse

from lib.hybrid_search import enhance_query, normalize, rrf_search, weighted_search


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

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Run reciprocal rank fusion search with given query"
    )
    rrf_search_parser.add_argument("query", help="Query to search")
    rrf_search_parser.add_argument(
        "-k", type=int, default=60, help="Rank coefficient for RRF"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="Limit search results"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            if args.enhance is None:
                query = args.query
            else:
                query = enhance_query(args.query, args.enhance)
                print(
                    f"Enhanced query ({query['method']}): {query['query']} -> {query['enhanced_query']}"
                )
                query = query["enhanced_query"]
            result = rrf_search(query, args.k, args.limit)
            print(f"RRF Search Results for '{result['query']}' (k={result['k']}):")
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                # print(f"   RRF Score: {res.get('rrf_score', 0):.3f}")
                # print(f"BM25 Rank: {res.get('bm25_rank', 0)}, Semantic Rank: {res.get('semantic_rank', 0)}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...")
                print()
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
