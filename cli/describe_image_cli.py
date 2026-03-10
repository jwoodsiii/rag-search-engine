import argparse
import mimetypes
import types

from google import genai
from lib.search_utils import get_gemini_client


def main():
    client = get_gemini_client()
    model = "gemma-3-27b-it"

    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--query", type=str, help="query to rewrite based on the image")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
            - Synthesize visual and textual information
            - Focus on movie-specific details (actors, scenes, style, etc.)
            - Return only the rewritten query, without any additional commentary"""
    with open(args.image, "rb") as f:
        image_data = f.read()
    parts = [
        system_prompt,
        genai.types.Part.from_bytes(data=image_data, mime_type=mime),
        args.query.strip(),
    ]
    resp = client.models.generate_content(model=model, contents=parts)
    print(f"Rewritten query: {resp.text.strip()}")
    if resp.usage_metadata is not None:
        print(f"Total tokens:    {resp.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
