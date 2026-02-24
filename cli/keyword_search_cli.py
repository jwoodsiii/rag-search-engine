#!/usr/bin/env python3
import argparse

from lib.kwsearch import build_command, search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build movies inverted index")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
