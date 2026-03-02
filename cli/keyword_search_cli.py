#!/usr/bin/env python3
import argparse

from lib.kwsearch import (
    BM25_B,
    BM25_K1,
    bm25_idf_command,
    bm25_tf_command,
    bm25search_command,
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)

from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build movies inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser = subparsers.add_parser(
        "tf", help="List term frequency for provided document id"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document to search term for")
    tf_parser.add_argument("term", type=str, help="Term to search frequency for")
    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency for provided term"
    )
    idf_parser.add_argument("term", type=str, help="Term to search for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get tfidf score for provided docid and term"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document to search term for")
    tfidf_parser.add_argument("term", type=str, help="Term to search for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="limit search results",
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            out = search_command(args.query)
            for i, res in enumerate(out, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "build":
            # build inverted index
            print("Building movie index")
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            print(
                f"Pulling term frequency for term: {args.term} from document: {args.doc_id}"
            )
            freq = tf_command(args.doc_id, args.term)
            print(f"Frequency: {freq}")

        case "idf":
            print(f"Calculating inverse document frequency for term {args.term}")
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            print(
                f"Calculating tfidf score for term {args.term} in document {args.doc_id}"
            )
            tfidf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}"
            )
        case "bm25idf":
            print(f"Calculating BM25 IDF score for term {args.term}")
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            print(
                f"Calculating BM25 TF score for term {args.term} in document {args.doc_id}"
            )
            bm25tf = bm25_tf_command(args.doc_id, args.term)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            print(f"Searching movies using full BM25 scoring for query '{args.query}'")
            bm25search_command(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
