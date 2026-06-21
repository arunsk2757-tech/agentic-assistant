import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()  # auto-detects GEMINI_API_KEY from env

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one sentence."
)
print(response.text)