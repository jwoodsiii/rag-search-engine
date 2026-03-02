#!/usr/bin/env python3

import argparse
from enum import verify

from lib.semantic_search import embed_text, verify_embeddings, verify_model


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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
