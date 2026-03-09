import argparse

from lib.augmented_generation import generate_answer
from lib.hybrid_search import rrf_search


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            rrf_results = rrf_search(query)
            results = [res["title"] for res in rrf_results["results"]]
            print("Search Results:")
            for title in results:
                print(f"- {title}")

            print("RAG Response")
            rag_response = generate_answer(query, results)
            print(rag_response)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
