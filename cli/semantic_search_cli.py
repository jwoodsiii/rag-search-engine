#!/usr/bin/env python3

import argparse
from enum import verify

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    search,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the model")

    embed_parser = subparsers.add_parser(
        "embed_text", help="Generate embeddings for provided text"
    )
    embed_parser.add_argument("text", help="Text to generate embeddings for")

    verify_embed_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings"
    )

    embedquery_parser = subparsers.add_parser(
        "embedquery", help="Generate embedding for query"
    )
    embedquery_parser.add_argument("query", help="Query to generate embeddings for")

    search_parser = subparsers.add_parser("search", help="Search for documents")
    search_parser.add_argument("query", help="Query to search for")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Verifying the model")
            verify_model()
            print("Finished model verification...")
        case "embed_text":
            print("Generate embeddings for text")
            embed_text(args.text)
            print("Finished generating embeddings...")
        case "verify_embeddings":
            print("Verifying embeddings")
            verify_embeddings()
            print("Finished verifying embeddings...")
        case "embedquery":
            print("Embedding query")
            embed_query_text(args.query)
            print("Finished embedding query...")
        case "search":
            print("Searching for documents")
            search(args.query, args.limit)
            print("Finished searching...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
