import argparse

from lib.hybrid_search import normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument(
        "scores", nargs="+", default="", type=float, help="Scores to normalize"
    )

    args = parser.parse_args()

    match args.command:
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
