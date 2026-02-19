# server.py
import os
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
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
    """
    Comma-separated list of allowed origins, e.g.
    FRONTEND_ORIGINS="http://localhost:5173,https://yourapp.vercel.app"
    """
    raw = os.getenv("FRONTEND_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]

    # sensible local defaults
    defaults = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    return origins or defaults


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


@app.post("/api/judge")
async def judge_case(
    files: List[UploadFile] = File(...),
    redact: bool = Form(False),
):
    tmp_paths: List[str] = []

    try:
        if not files:
            return {"error": "No files uploaded."}

        # Save uploads to temp files
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
        # helpful error surface for debugging deploys
        return {"error": f"{type(e).__name__}: {str(e)}"}

    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
