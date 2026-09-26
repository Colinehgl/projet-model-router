import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def generate(prompt: str, max_tokens: int = 500) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

if __name__ == "__main__":
    print(generate("Explique-moi ce qu'est un routeur de modèles LLM en deux phrases."))