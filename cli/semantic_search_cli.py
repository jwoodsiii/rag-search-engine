#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    chunk_text,
    embed_query_text,
    embed_text,
    search,
    semantic_chunk,
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
    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Split text into words on whitespace, grouped by <chunk-size> words",
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size")
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Overlap size for chunks"
    )
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Semantic chunk provided text"
    )
    semantic_chunk_parser.add_argument("text", help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Max chunk size"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Overlap size"
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
        case "chunk":
            print("Chunking text")
            chunk_text(args.text, args.chunk_size, args.overlap)
            print("Finished chunking...")
        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            output = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            for i, chunk in enumerate(output):
                print(f"{i + 1}. {chunk}")
            print("Finished semantic chunking...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
