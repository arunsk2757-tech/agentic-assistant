import os
import re
import time
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException

load_dotenv()

app = FastAPI()
print("DEBUG - key present:", bool(os.environ.get("GEMINI_API_KEY")))
print("DEBUG - key length:", len(os.environ.get("GEMINI_API_KEY", "")))
client = genai.Client()

SYSTEM_INSTRUCTION = """You are a code generator. Given a description of a web app,
return a SINGLE complete HTML file with inline <style> and <script> tags.
Rules:
- Output ONLY raw HTML code, starting with <!DOCTYPE html>
- Do NOT wrap the output in markdown code fences
- Do NOT include any explanation or text before/after the code
- The app must be fully functional using only vanilla HTML/CSS/JavaScript
- NEVER use eval() or the Function() constructor for calculations or logic — 
  implement arithmetic and logic explicitly using operators and conditionals
- Make it visually clean and usable
"""

class GenerateRequest(BaseModel):
    prompt: str

def clean_code(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:html)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()

@app.post("/generate")
def generate(req: GenerateRequest):
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=req.prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            )
            code = clean_code(response.text)
            return {"code": code}
        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)  # wait 1s, then 2s, then 4s before retrying

    raise HTTPException(status_code=503, detail="Gemini is busy right now — please try again in a moment.")

app.mount("/", StaticFiles(directory="static", html=True), name="static")