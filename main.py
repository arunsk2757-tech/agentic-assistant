import time
import os
import re
import json
import requests
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

session_files: Dict[str, str] = {}

MAX_STEPS = 10
MAX_FIX_ATTEMPTS = 3

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

# ---- Phase 5: Job Search Assistant (Adzuna + JSearch combined) ----

ROLE_KEYWORDS = {
    "backend": "Django FastAPI Python developer",
    "fullstack": "React Python full stack developer",
    "ai": "AI LLM machine learning engineer Python",
}

ADZUNA_LOCATION = {
    "kochi": "Kochi",
    "kerala": "Kerala",
    "kerala_remote": "Kerala",
    "india": "",
}

JSEARCH_LOCATION_TEXT = {
    "kochi": "Kochi, India",
    "kerala": "Kerala, India",
    "kerala_remote": "Kerala, India",
    "india": "India",
}

EMPLOYMENT_TYPES = {
    "fulltime": "FULLTIME",
    "fulltime_intern": "FULLTIME,INTERN",
    "fulltime_contract": "FULLTIME,CONTRACTOR",
}

class JobSearchRequest(BaseModel):
    location: str
    role_type: str
    job_type: str

def search_adzuna_jobs(keywords, location):
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
    params = {
        "app_id": os.environ["ADZUNA_APP_ID"],
        "app_key": os.environ["ADZUNA_APP_KEY"],
        "what": keywords,
        "where": location,
        "results_per_page": 8,
        "content-type": "application/json",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    raw_jobs = response.json().get("results", [])

    normalized = []
    for job in raw_jobs:
        normalized.append({
            "id": "adzuna_" + str(job.get("id")),
            "title": job.get("title", ""),
            "company": (job.get("company") or {}).get("display_name", ""),
            "location": (job.get("location") or {}).get("display_name", ""),
            "salary_min": job.get("salary_min"),
            "salary_max": job.get("salary_max"),
            "url": job.get("redirect_url", ""),
            "source": "Adzuna",
        })
    return normalized

def search_jsearch_jobs(keywords, location_text, employment_types):
    url = "https://jsearch.p.rapidapi.com/search-v2"
    headers = {
        "X-RapidAPI-Key": os.environ["RAPIDAPI_KEY"],
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }
    params = {
        "query": f"{keywords} in {location_text}",
        "num_pages": "1",
        "country": "in",
        "date_posted": "all",
    }
    response = requests.get(url, headers=headers, params=params, timeout=20)
    response.raise_for_status()
    raw_jobs = response.json().get("data", {}).get("jobs", [])

    normalized = []
    for job in raw_jobs:
        normalized.append({
            "id": "jsearch_" + str(job.get("job_id")),
            "title": job.get("job_title", ""),
            "company": job.get("employer_name", ""),
            "location": job.get("job_city") or job.get("job_country", ""),
            "salary_min": job.get("job_min_salary"),
            "salary_max": job.get("job_max_salary"),
            "url": job.get("job_apply_link", ""),
            "source": "JSearch",
        })
    return normalized

@app.post("/jobs")
def jobs(req: JobSearchRequest):
    adzuna_location = ADZUNA_LOCATION.get(req.location, "")
    jsearch_location_text = JSEARCH_LOCATION_TEXT.get(req.location, "India")
    employment_types = EMPLOYMENT_TYPES.get(req.job_type, "FULLTIME")

    if req.role_type == "all":
        keyword_sets = list(ROLE_KEYWORDS.values())
    else:
        keyword_sets = [ROLE_KEYWORDS.get(req.role_type, "Python developer")]

    all_results = []
    for kw in keyword_sets:
        try:
            all_results.extend(search_adzuna_jobs(kw, adzuna_location))
        except Exception as e:
            print("Adzuna search failed for", kw, "->", e)

        try:
            all_results.extend(search_jsearch_jobs(kw, jsearch_location_text, employment_types))
        except Exception as e:
            print("JSearch search failed for", kw, "->", e)

    if not all_results:
        return {"summary": "No job listings found for these criteria. Try different options.", "jobs": []}

    seen_ids = set()
    unique_results = []
    for job in all_results:
        if job["id"] not in seen_ids:
            seen_ids.add(job["id"])
            unique_results.append(job)

    job_summaries = unique_results[:25]

    summary_prompt = f"""Here are real job listings from multiple sources:
{json.dumps(job_summaries, indent=2)}

Write a short, clear plain-text summary of the best matching jobs. For each, include title, company, location, salary if available, and the apply link. Group similar roles together. Skip irrelevant or duplicate listings."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": summary_prompt}],
        )
        summary = response.choices[0].message.content
    except Exception as e:
        summary = f"Found {len(unique_results)} jobs, but summarization failed: {e}"

    return {"summary": summary, "jobs": job_summaries}

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