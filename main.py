import time
import os
import re
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Dict
from openai import OpenAI

load_dotenv()

app = FastAPI()

client = genai.Client()

groq_client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

# In-memory storage for the current project's files (simple dict for now — real
# session handling and persistence comes later)
session_files: Dict[str, str] = {}

MAX_STEPS = 10
MAX_FIX_ATTEMPTS = 3

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

groq_tools = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content. Use this for every file in the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file, e.g. index.html, style.css"},
                    "content": {"type": "string", "description": "The full content to write into the file"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the current content of an existing file in the project",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "Name of the file to read"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all filenames that currently exist in the project",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

BUILD_SYSTEM_INSTRUCTION = """You are a web app building agent. Given a project description,
build a complete multi-file website using the write_file tool.
Rules:
- Always create an index.html as the main entry file
- Link CSS/JS files using relative paths (e.g. <link rel="stylesheet" href="style.css">)
- Use the write_file tool for every file you create — never return code as plain text
- You may call list_files or read_file to check your previous work before finishing
- Never use eval() or the Function() constructor
- Avoid complex regular expressions (patterns with many backslashes); use simple methods like .includes() or .indexOf() for basic text checks instead
- When the project is fully complete and working, respond with a short plain-text summary and STOP calling tools
- Keep the design clean and usable
"""

EDIT_SYSTEM_INSTRUCTION = """You are editing an existing project. Files already exist.
Rules:
- First call list_files to see what exists
- Then call read_file on any file you need to change before editing it
- Make only the specific change requested using write_file
- Do not recreate or rewrite files you are not changing
- Never use eval() or the Function() constructor
- Avoid complex regular expressions; use simple methods like .includes() or .indexOf() instead
- When the edit is complete, respond with a short plain-text summary and STOP calling tools
"""

class BuildRequest(BaseModel):
    prompt: str

class EditRequest(BaseModel):
    prompt: str

def run_agent_loop(messages, max_steps):
    """Runs the tool-calling loop until the model stops calling tools or max_steps is hit."""
    log = []
    for step in range(max_steps):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=groq_tools,
            )
        except Exception as e:
            log.append(f"Model formatting hiccup, retrying this step: {e}")
            continue

        msg = response.choices[0].message

        if not msg.tool_calls:
            log.append("Done: " + (msg.content or "Build complete."))
            return log

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            func = TOOL_FUNCTIONS.get(tc.function.name)
            args = json.loads(tc.function.arguments)
            result = func(**args) if func else f"Unknown tool: {tc.function.name}"
            log.append(f"{tc.function.name}({args}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    log.append("Reached max steps without finishing.")
    return log

def test_project_in_sandbox():
    """Writes all .js files into a fresh E2B sandbox and checks their syntax with Node."""
    from e2b_code_interpreter import Sandbox

    js_files = {name: content for name, content in session_files.items() if name.endswith(".js")}
    if not js_files:
        return []

    errors = []
    with Sandbox.create() as sandbox:
        for filename, content in js_files.items():
            path = f"/home/user/{filename}"
            sandbox.files.write(path, content)
            try:
                sandbox.commands.run(f"node --check {path}")
            except Exception as e:
                errors.append({"filename": filename, "error": str(e)})
    return errors

def run_build_or_edit(messages):
    """Runs the agent loop, then tests and self-corrects, shared by /build and /edit."""
    steps_log = run_agent_loop(messages, MAX_STEPS)

    test_results = test_project_in_sandbox()
    fix_attempts = 0
    while test_results and fix_attempts < MAX_FIX_ATTEMPTS:
        error_report = "\n".join(f"- {r['filename']}: {r['error']}" for r in test_results)
        messages.append({
            "role": "user",
            "content": f"Testing found real errors in the generated code:\n{error_report}\nPlease fix these files using write_file."
        })
        steps_log.append(f"--- Fix attempt {fix_attempts + 1} ---")
        steps_log += run_agent_loop(messages, MAX_STEPS)
        test_results = test_project_in_sandbox()
        fix_attempts += 1

    if test_results:
        steps_log.append(f"Remaining issues after {fix_attempts} fix attempt(s): {test_results}")
    elif fix_attempts > 0:
        steps_log.append("All issues resolved after self-correction.")
    else:
        steps_log.append("All JavaScript files passed syntax check on the first try.")

    return steps_log

@app.post("/build")
def build(req: BuildRequest):
    session_files.clear()

    messages = [
        {"role": "system", "content": BUILD_SYSTEM_INSTRUCTION},
        {"role": "user", "content": req.prompt},
    ]

    steps_log = run_build_or_edit(messages)

    print("BUILD LOG:", steps_log)
    return {"files": list(session_files.keys()), "log": steps_log}

@app.post("/edit")
def edit(req: EditRequest):
    if not session_files:
        raise HTTPException(status_code=400, detail="No project exists yet. Build one first.")

    messages = [
        {"role": "system", "content": EDIT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": req.prompt},
    ]

    steps_log = run_build_or_edit(messages)

    print("EDIT LOG:", steps_log)
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
            time.sleep(2 ** attempt)

    raise HTTPException(status_code=503, detail="Gemini is busy right now — please try again in a moment.")

app.mount("/", StaticFiles(directory="static", html=True), name="static")