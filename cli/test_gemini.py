import os

from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

from google import genai

client = genai.Client(api_key=api_key)

resp = client.models.generate_content(
    model="gemma-3-27b-it",
    contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
)

print(resp.text)
print(
    f"Prompt tokens: {resp.usage_metadata.prompt_token_count}\nResponse tokens: {resp.usage_metadata.candidates_token_count}"
)
