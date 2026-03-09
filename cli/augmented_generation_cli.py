import argparse

from lib.augmented_generation import (
    generate_answer,
    generate_citation,
    generate_qa,
    generate_summary,
    rag_command,
)
from lib.hybrid_search import rrf_search


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize documents from search"
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    summarize_parser.add_argument("--limit", type=int, default=5, help="Query limits")

    citations_parser = subparsers.add_parser(
        "citations", help="Generate citations from search results"
    )
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument("--limit", type=int, default=5, help="Query limits")

    question_parser = subparsers.add_parser(
        "question", help="Generate an answer to a question based on search results"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument("--limit", type=int, default=5, help="Query limits")

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query, generate_answer)

            print("Search Results:")
            for doc in result["search_results"]:
                print(f"- {doc['title']}")

            print("RAG Response")
            print(result["answer"])
        case "summarize":
            result = rag_command(args.query, generate_summary)

            print("Search Results:")
            for doc in result["search_results"]:
                print(f"- {doc['title']}")

            print("LLM Summary")
            print(result["answer"])

        case "citations":
            result = rag_command(args.query, generate_citation)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"- {doc['title']}")

            print("LLM Answer")
            print(result["answer"])
        case "question":
            query = args.question
            result = rag_command(query, generate_qa)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"- {doc['title']}")

            print("LLM Answer")
            print(result["answer"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
