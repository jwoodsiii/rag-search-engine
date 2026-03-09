import argparse

from lib.hybrid_search import (
    evaluate_results,
    normalize,
    rrf_search,
    weighted_search,
)


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
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        default="",
        choices=["individual", "batch", "cross_encoder"],
    )
    rrf_search_parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate results using RRF scores"
    )

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            result = rrf_search(
                args.query, args.k, args.enhance, args.rerank_method, args.limit
            )
            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )
            if result["reranked"]:
                print(
                    f"Re-ranking top {len(result['results'])} results using {result['rerank_method']} method...\n"
                )
            print(
                f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):"
            )
            rrf_results = []
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                rrf_results.append(f"{i}. {res['title']}")
                if "individual_score" in res:
                    print(f"   Re-rank Score: {res.get('individual_score', 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Re-rank Rank: {res.get('batch_rank', 0)}")
                if "crossencoder_score" in res:
                    print(
                        f"    Cross Encoder Score: {res.get('crossencoder_score', 0):.3f}"
                    )
                print(f"   RRF Score: {res.get('score', 0):.3f}")
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
            if args.evaluate:
                evaluate_results(args.query, rrf_results)

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
