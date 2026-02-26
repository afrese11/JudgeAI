# server.py
import os
import tempfile
from typing import List, Optional

import httpx
from pydantic import BaseModel
from supabase import create_client

from fastapi import FastAPI, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware

from judge_core import run_prediction_with_uploaded_pdfs


PROMPT_TEMPLATE = """
You are serving as an appellate judge reviewing an appellate case which includes {num_docs} document(s).

CRITICAL INSTRUCTIONS:
- You must PREDICT the outcome based ONLY on the legal arguments, facts, and law presented in these documents.
- DO NOT use any external knowledge about this case, parties, or docket number.
- DO NOT simply extract or report any actual court decision that appears in the documents.
- Treat this as if YOU are the appellate court making the decision for the first time.
- Base your prediction solely on: the strength of legal arguments presented, applicable law and precedent discussed in the briefs, the quality of each party's legal reasoning, and the facts of the case.

Your task is to produce TWO separate PREDICTIVE outputs:

===CASE SUMMARY===
[Your predicted case summary here]

===CASE DECISION===
[Your predicted case decision here]
""".strip()


def _authorized(passcode: str | None) -> bool:
    expected = os.getenv("JUDGEAI_SHARED_PASSCODE", "")
    return bool(expected) and (passcode == expected)


def parse_gpt_response(response_text: str):
    parts = response_text.split("===CASE SUMMARY===")
    if len(parts) < 2:
        return None, None

    remainder = parts[1]
    decision_parts = remainder.split("===CASE DECISION===")
    if len(decision_parts) < 2:
        return None, None

    case_summary = decision_parts[0].strip()
    case_decision = decision_parts[1].strip()
    return case_summary, case_decision


def _get_cors_origins() -> List[str]:
    raw = os.getenv("FRONTEND_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    defaults = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    return origins or defaults


# ---- Supabase (server-side) ----
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "judgeai-pdfs")

supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


# -------------------------
# Existing endpoint (small uploads)
# -------------------------
@app.post("/api/judge")
async def judge_case(
    files: List[UploadFile] = File(...),
    redact: bool = Form(False),
    x_judgeai_passcode: str | None = Header(default=None),
):
    if not _authorized(x_judgeai_passcode):
        return {"error": "Unauthorized"}

    tmp_paths: List[str] = []

    try:
        if not files:
            return {"error": "No files uploaded."}

        for uf in files:
            name = uf.filename or "uploaded.pdf"
            ext = os.path.splitext(name)[1].lower()
            if ext != ".pdf":
                return {"error": f"Only PDFs supported right now. Got: {name}"}

            fd, path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)

            with open(path, "wb") as out:
                out.write(await uf.read())

            tmp_paths.append(path)

        prompt = PROMPT_TEMPLATE.format(num_docs=len(tmp_paths))
        model = os.getenv("OPENAI_MODEL", "gpt-5")

        raw = run_prediction_with_uploaded_pdfs(tmp_paths, prompt, model=model)
        case_summary, case_decision = parse_gpt_response(raw)

        return {
            "raw": raw,
            "case_summary": case_summary,
            "case_decision": case_decision,
            "num_documents": len(tmp_paths),
            "redact": redact,
            "model": model,
        }

    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}

    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass


# -------------------------
# NEW: Signed upload init
# -------------------------
class UploadInitFile(BaseModel):
    filename: str
    content_type: Optional[str] = "application/pdf"


class UploadInitRequest(BaseModel):
    files: List[UploadInitFile]


@app.post("/api/uploads/init")
def init_uploads(
    req: UploadInitRequest,
    x_judgeai_passcode: str | None = Header(default=None),
):
    if not _authorized(x_judgeai_passcode):
        return {"error": "Unauthorized"}

    if not supabase:
        return {"error": "Supabase is not configured on the server."}

    uploads: List[dict] = []

    for f in req.files:
        safe_name = (f.filename or "uploaded.pdf").replace("/", "_")
        object_path = f"uploads/{os.urandom(12).hex()}-{safe_name}"

        # Signed upload URL + token
        # supabase-py returns an object with .data OR sometimes a dict
        res = supabase.storage.from_(SUPABASE_BUCKET).create_signed_upload_url(object_path)
        data = getattr(res, "data", None) if not isinstance(res, dict) else res.get("data")

        if not data or "signedUrl" not in data or "token" not in data or "path" not in data:
            return {"error": f"Failed to create signed upload URL for {f.filename}"}

        uploads.append(
            {
                "path": data["path"],
                "signed_url": data["signedUrl"],
                "token": data["token"],
            }
        )

    return {"uploads": uploads, "bucket": SUPABASE_BUCKET}


# -------------------------
# NEW: Judge from storage paths
# -------------------------
class JudgeFromStorageRequest(BaseModel):
    paths: List[str]
    redact: bool = False
    # optional: delete after processing
    cleanup: bool = True


@app.post("/api/judge-from-storage")
async def judge_from_storage(
    req: JudgeFromStorageRequest,
    x_judgeai_passcode: str | None = Header(default=None),
):
    if not _authorized(x_judgeai_passcode):
        return {"error": "Unauthorized"}

    if not supabase:
        return {"error": "Supabase is not configured on the server."}

    if not req.paths:
        return {"error": "No storage paths provided."}

    tmp_paths: List[str] = []
    client = httpx.AsyncClient(timeout=120)

    try:
        # Download each PDF using signed download URLs
        for path in req.paths:
            signed = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(path, 60 * 15)
            data = getattr(signed, "data", None) if not isinstance(signed, dict) else signed.get("data")

            # supabase might return "signedURL" or "signedUrl" depending on lib version
            url = None
            if data:
                url = data.get("signedURL") or data.get("signedUrl") or data.get("signed_url")

            if not url:
                return {"error": f"Failed to create signed download URL for {path}"}

            r = await client.get(url)
            r.raise_for_status()

            fd, local_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            with open(local_path, "wb") as out:
                out.write(r.content)

            tmp_paths.append(local_path)

        prompt = PROMPT_TEMPLATE.format(num_docs=len(tmp_paths))
        model = os.getenv("OPENAI_MODEL", "gpt-5")

        raw = run_prediction_with_uploaded_pdfs(tmp_paths, prompt, model=model)
        case_summary, case_decision = parse_gpt_response(raw)

        # Optional cleanup: delete stored objects after processing
        if req.cleanup:
            try:
                supabase.storage.from_(SUPABASE_BUCKET).remove(req.paths)
            except Exception:
                pass

        return {
            "raw": raw,
            "case_summary": case_summary,
            "case_decision": case_decision,
            "num_documents": len(tmp_paths),
            "redact": req.redact,
            "model": model,
        }

    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}

    finally:
        await client.aclose()
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass