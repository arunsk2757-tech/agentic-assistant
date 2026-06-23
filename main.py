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
from typing import Dict
from fastapi.responses import Response

load_dotenv()

app = FastAPI()
#print("DEBUG - key present:", bool(os.environ.get("GEMINI_API_KEY")))
#print("DEBUG - key length:", len(os.environ.get("GEMINI_API_KEY", "")))
client = genai.Client()

# In-memory storage for the current project's files (simple dict for now — real
# session handling and persistence comes later)
session_files: Dict[str, str] = {}

MAX_STEPS = 10

# ---- Tool implementations ----
def write_file(filename: str, content: str) -> str:
    session_files[filename] = content
    return f"Wrote {len(content)} characters to {filename}"

def read_file(filename: str) -> str:
    if filename not in session_files:
        return f"Error: {filename} does not exist"
    return session_files[filename]

def list_files() -> str:
    if not session_files:
        return "No files yet"
    return ", ".join(session_files.keys())

TOOL_FUNCTIONS = {
    "write_file": write_file,
    "read_file": read_file,
    "list_files": list_files,
}

# ---- Tool schemas Gemini will see ----
write_file_decl = types.FunctionDeclaration(
    name="write_file",
    description="Create or overwrite a file with the given content. Use this for every file in the project.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Name of the file, e.g. index.html, style.css, about.html"},
            "content": {"type": "string", "description": "The full content to write into the file"},
        },
        "required": ["filename", "content"],
    },
)

read_file_decl = types.FunctionDeclaration(
    name="read_file",
    description="Read the current content of an existing file in the project",
    parameters_json_schema={
        "type": "object",
        "properties": {"filename": {"type": "string", "description": "Name of the file to read"}},
        "required": ["filename"],
    },
)

list_files_decl = types.FunctionDeclaration(
    name="list_files",
    description="List all filenames that currently exist in the project",
    parameters_json_schema={"type": "object", "properties": {}},
)

build_tool = types.Tool(function_declarations=[write_file_decl, read_file_decl, list_files_decl])

BUILD_SYSTEM_INSTRUCTION = """You are a web app building agent. Given a project description,
build a complete multi-file website using the write_file tool.
Rules:
- Always create an index.html as the main entry file
- Link CSS/JS files using relative paths (e.g. <link rel="stylesheet" href="style.css">)
- Use the write_file tool for every file you create — never return code as plain text
- You may call list_files or read_file to check your previous work before finishing
- Never use eval() or the Function() constructor
- When the project is fully complete and working, respond with a short plain-text summary and STOP calling tools
- Keep the design clean and usable
"""

class BuildRequest(BaseModel):
    prompt: str

@app.post("/build")
def build(req: BuildRequest):
    session_files.clear()  # fresh project each time, for now

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=req.prompt)])]
    steps_log = []

    for step in range(MAX_STEPS):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=BUILD_SYSTEM_INSTRUCTION,
                tools=[build_tool],
            ),
        )

        if not response.function_calls:
            steps_log.append("Done: " + (response.text or "Build complete."))
            break

        contents.append(response.candidates[0].content)

        response_parts = []
        for fc in response.function_calls:
            func = TOOL_FUNCTIONS.get(fc.name)
            result = func(**fc.args) if func else f"Unknown tool: {fc.name}"
            steps_log.append(f"{fc.name}({fc.args}) -> {result}")
            response_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))

        contents.append(types.Content(role="tool", parts=response_parts))
    else:
        steps_log.append("Reached max steps without finishing.")

    return {"files": list(session_files.keys()), "log": steps_log}

@app.get("/preview/{filename:path}")
def preview(filename: str):
    if filename not in session_files:
        return Response("Not found", status_code=404)
    content = session_files[filename]
    media_type = "text/plain"
    if filename.endswith(".html"):
        media_type = "text/html"
    elif filename.endswith(".css"):
        media_type = "text/css"
    elif filename.endswith(".js"):
        media_type = "application/javascript"
    return Response(content, media_type=media_type)



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