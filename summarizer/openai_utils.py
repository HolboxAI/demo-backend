import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

def generate_summary(text: str) -> str:
    """
    Generate a summary of the input text using OpenAI ChatCompletion.
    If the text is very long, consider chunking before calling.
    """
    system_prompt = "You are a helpful assistant that summarizes documents."
    user_prompt = f"Please provide a concise summary of the following document content:\n\n{text}"

    response = openai.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
