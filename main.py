import time
import os
import re
import json
import io
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

app = FastAPI()

client = genai.Client()

groq_client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

# ---- Persistence (SQLite, with multi-project support) ----

DB_PATH = "agent_bench.db"

current_project = "default"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            project TEXT,
            filename TEXT,
            content TEXT,
            PRIMARY KEY (project, filename)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note TEXT,
            due_date TEXT
        )
    """)
    return conn

def clear_files():
    conn = get_db()
    conn.execute("DELETE FROM files WHERE project = ?", (current_project,))
    conn.commit()
    conn.close()

def get_all_files() -> dict:
    conn = get_db()
    rows = conn.execute("SELECT filename, content FROM files WHERE project = ?", (current_project,)).fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

def get_file_content(filename: str):
    conn = get_db()
    row = conn.execute(
        "SELECT content FROM files WHERE project = ? AND filename = ?",
        (current_project, filename)
    ).fetchone()
    conn.close()
    return row[0] if row else None

# ---- Tool implementations (scoped to current_project automatically) ----

def write_file(filename: str, content: str) -> str:
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO files (project, filename, content) VALUES (?, ?, ?)",
        (current_project, filename, content)
    )
    conn.commit()
    conn.close()
    return f"Wrote {len(content)} characters to {filename}"

def read_file(filename: str) -> str:
    content = get_file_content(filename)
    if content is None:
        return f"Error: {filename} does not exist"
    return content

def list_files() -> str:
    files = get_all_files()
    if not files:
        return "No files yet"
    return ", ".join(files.keys())

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
    project: Optional[str] = None

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
            args = json.loads(tc.function.arguments) or {}
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

    files = get_all_files()
    js_files = {name: content for name, content in files.items() if name.endswith(".js")}
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

MAX_STEPS = 10
MAX_FIX_ATTEMPTS = 3

@app.post("/build")
def build(req: BuildRequest):
    global current_project
    if req.project:
        current_project = req.project.strip()

    clear_files()

    messages = [
        {"role": "system", "content": BUILD_SYSTEM_INSTRUCTION},
        {"role": "user", "content": req.prompt},
    ]

    steps_log = run_build_or_edit(messages)

    print("BUILD LOG:", steps_log)
    return {"files": list(get_all_files().keys()), "project": current_project, "log": steps_log}

@app.post("/edit")
def edit(req: EditRequest):
    if not get_all_files():
        raise HTTPException(status_code=400, detail="No project exists yet. Build one first.")

    messages = [
        {"role": "system", "content": EDIT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": req.prompt},
    ]

    steps_log = run_build_or_edit(messages)

    print("EDIT LOG:", steps_log)
    return {"files": list(get_all_files().keys()), "log": steps_log}

@app.get("/preview/{filename:path}")
def preview(filename: str):
    content = get_file_content(filename)
    if content is None:
        return Response("Not found", status_code=404)
    media_type = "text/plain"
    if filename.endswith(".html"):
        media_type = "text/html"
    elif filename.endswith(".css"):
        media_type = "text/css"
    elif filename.endswith(".js"):
        media_type = "application/javascript"
    return Response(content, media_type=media_type)

@app.get("/files")
def list_existing_files():
    return {"files": list(get_all_files().keys()), "project": current_project}

class FileEditRequest(BaseModel):
    filename: str
    content: str

@app.get("/file/{filename}")
def get_single_file(filename: str):
    content = get_file_content(filename)
    if content is None:
        raise HTTPException(status_code=404, detail=f"{filename} not found in current project.")
    return {"filename": filename, "content": content}

@app.post("/file")
def save_single_file(req: FileEditRequest):
    write_file(req.filename, req.content)
    return {"filename": req.filename, "saved": True}

# ---- Project management ----

class ProjectSwitchRequest(BaseModel):
    project: str

@app.get("/projects")
def list_projects():
    conn = get_db()
    rows = conn.execute("SELECT DISTINCT project FROM files ORDER BY project").fetchall()
    conn.close()
    projects = {row[0] for row in rows}
    projects.add(current_project)
    return {"projects": sorted(projects), "current": current_project}

@app.post("/projects/switch")
def switch_project(req: ProjectSwitchRequest):
    global current_project
    name = req.project.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name cannot be empty.")
    current_project = name
    return {"current": current_project, "files": list(get_all_files().keys())}

# ---- Phase 5/9: Job Search Assistant (Adzuna + JSearch, custom location/role, experience filter) ----

ROLE_KEYWORDS = {
    "backend": "Django FastAPI Python developer",
    "fullstack": "React Python full stack developer",
    "ai": "AI LLM machine learning engineer Python",
}

ADZUNA_LOCATION = {
    "kochi": "Kochi",
    "trivandrum": "Thiruvananthapuram",
    "kozhikode": "Kozhikode",
    "kerala": "Kerala",
    "kerala_remote": "Kerala",
    "bangalore": "Bangalore",
    "hyderabad": "Hyderabad",
    "chennai": "Chennai",
    "pune": "Pune",
    "india": "",
}

JSEARCH_LOCATION_TEXT = {
    "kochi": "Kochi, India",
    "trivandrum": "Thiruvananthapuram, India",
    "kozhikode": "Kozhikode, India",
    "kerala": "Kerala, India",
    "kerala_remote": "Kerala, India",
    "bangalore": "Bangalore, India",
    "hyderabad": "Hyderabad, India",
    "chennai": "Chennai, India",
    "pune": "Pune, India",
    "india": "India",
}

EMPLOYMENT_TYPES = {
    "fulltime": "FULLTIME",
    "fulltime_intern": "FULLTIME,INTERN",
    "fulltime_contract": "FULLTIME,CONTRACTOR",
}

EXPERIENCE_KEYWORDS = {
    "any": "",
    "fresher": "fresher entry level",
    "junior": "1 to 3 years",
    "mid": "3 to 5 years",
    "senior": "5+ years senior",
}

class JobSearchRequest(BaseModel):
    location: str
    custom_location: Optional[str] = None
    role_type: str
    custom_role: Optional[str] = None
    job_type: str
    experience: Optional[str] = "any"

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
            "description": job.get("description", ""),
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
            "description": job.get("job_description", ""),
            "source": "JSearch",
        })
    return normalized

@app.post("/jobs")
def jobs(req: JobSearchRequest):
    if req.location == "custom" and req.custom_location:
        adzuna_location = req.custom_location.strip()
        jsearch_location_text = req.custom_location.strip() + ", India"
    else:
        adzuna_location = ADZUNA_LOCATION.get(req.location, "")
        jsearch_location_text = JSEARCH_LOCATION_TEXT.get(req.location, "India")

    employment_types = EMPLOYMENT_TYPES.get(req.job_type, "FULLTIME")
    experience_suffix = EXPERIENCE_KEYWORDS.get(req.experience or "any", "")

    if req.role_type == "custom" and req.custom_role:
        keyword_sets = [req.custom_role.strip()]
    elif req.role_type == "all":
        keyword_sets = list(ROLE_KEYWORDS.values())
    else:
        keyword_sets = [ROLE_KEYWORDS.get(req.role_type, "Python developer")]

    if experience_suffix:
        keyword_sets = [f"{kw} {experience_suffix}" for kw in keyword_sets]

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

    return {"jobs": job_summaries}

# ---- Phase 6: Cover Letter Generator ----

class CoverLetterRequest(BaseModel):
    job_title: str
    company: str
    job_description: str
    background: str

@app.post("/cover-letter")
def cover_letter(req: CoverLetterRequest):
    prompt = f"""Write a professional, tailored cover letter for this job application.

Job Title: {req.job_title}
Company: {req.company}
Job Description: {req.job_description}

Candidate Background:
{req.background}

Rules:
- Keep it concise, 3-4 paragraphs
- Reference specific requirements from the job description where genuinely relevant
- Professional but natural tone, not overly formal
- Do not invent skills, experience, or claims not present in the candidate background
- The letter MUST start with a greeting line on its own, exactly like this:
Dear Hiring Manager,
- The closing must be formatted as exactly two separate lines, like this:
Best regards,
[Candidate's name]
- Never put the closing phrase and the name on the same line or in the same sentence
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        letter = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cover letter generation failed: {e}")

    return {"letter": letter}

@app.post("/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        raw_text = ""
        for page in pdf_reader.pages:
            raw_text += page.extract_text() or ""
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from this PDF. It might be a scanned image rather than real text.")

    summary_prompt = f"""Here is raw text extracted from a resume:

{raw_text[:6000]}

Write a concise, professional background summary (3-5 sentences) describing this candidate's
role/title, years of experience, key technical skills, and education.
Write in third person, factual only — do not invent any information not present in the text above.
This summary will be used as input for generating cover letters, so keep it general enough to apply to multiple job applications."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": summary_prompt}],
        )
        background_summary = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Resume summarization failed: {e}")

    return {"background": background_summary}

# ---- Phase 7: Reminders / Task Tracking ----

class ReminderRequest(BaseModel):
    note: str
    days_from_now: int

def get_reminders_db() -> list:
    conn = get_db()
    rows = conn.execute("SELECT note, due_date FROM reminders ORDER BY due_date ASC").fetchall()
    conn.close()
    return [{"note": row[0], "due_date": row[1]} for row in rows]

@app.post("/reminders")
def add_reminder(req: ReminderRequest):
    due_date = (datetime.now() + timedelta(days=req.days_from_now)).strftime("%Y-%m-%d")
    conn = get_db()
    conn.execute("INSERT INTO reminders (note, due_date) VALUES (?, ?)", (req.note, due_date))
    conn.commit()
    conn.close()
    return {"reminders": get_reminders_db()}

@app.get("/reminders")
def get_reminders():
    today = datetime.now().strftime("%Y-%m-%d")
    return {
        "reminders": get_reminders_db(),
        "today": today,
    }

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