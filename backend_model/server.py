# server.py
import os
import sys
import tempfile
from dataclasses import asdict
from typing import List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from judge_core import run_prediction_with_uploaded_pdfs

# Ensure project root is on path so rag_context can be imported (e.g. when run from backend_model)
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from rag_context.top_k_retrieval import retrieve_similar_cases_from_pdf_uploads


# -----------------------------
# Your existing prompt + parsing
# (inlined so it's copy/pasteable)
# -----------------------------

PROMPT_TEMPLATE = """
You are serving as an appellate judge reviewing an appellate case which includes {num_docs} document(s).

CRITICAL INSTRUCTIONS:
- You must PREDICT the outcome based ONLY on the legal arguments, facts, and law presented in these documents.
- DO NOT use any external knowledge about this case, parties, or docket number.
- DO NOT simply extract or report any actual court decision that appears in the documents.
- Treat this as if YOU are the appellate court making the decision for the first time.
- Base your prediction solely on: the strength of legal arguments presented, applicable law and precedent discussed in the briefs, the quality of each party's legal reasoning, and the facts of the case.

The documents may include:
- Appellant's brief (arguing the lower court was wrong)
- Appellee's brief (defending the lower court decision)
- Reply briefs
- Addendums with relevant statutes, exhibits, or the lower court's decision

Your task is to produce TWO separate PREDICTIVE outputs:

**OUTPUT 1: CASE SUMMARY DOCUMENT (PREDICTIVE)**
Predict and provide:
1. A brief 3-5 sentence summary of the key legal issues on appeal.
2. Statement of the lower court's decision (what ruling is being appealed from).
3. Your PREDICTED recommendation for the length of oral argument, based on the legal complexity of the issues on appeal.
4. An explanation of the case complexity and your reasoning for the oral argument time recommendation.

**OUTPUT 2: CASE DECISION DOCUMENT (PREDICTIVE)**
Predict and provide:
1. A written judicial opinion that decides all of the issues raised on the appeal, based on your analysis of the arguments presented.
2. Your PREDICTED determination of whether the case should be AFFIRMED, REVERSED, VACATED, or another disposition.
3. Legal reasoning supporting your predicted decision, citing the arguments and law from the briefs.

Please structure your response exactly as follows:
""".strip()

PROMPT_APPEND = """
===CASE SUMMARY===
[Your predicted case summary here]

===CASE DECISION===
[Your predicted case decision here]
""".strip()


def parse_gpt_response(response_text: str):
    """
    Returns (case_summary, case_decision) or (None, None) if markers missing.
    """
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


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI()

# Dev CORS so your Vite frontend can call the Python backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/retrieve-similar")
async def retrieve_similar(
    files: List[UploadFile] = File(...),
    k: int = Form(3),
):
    """
    Accept drag-and-drop PDF case briefs; return top-k similar cases from the RAG index.
    """
    if not files:
        return {"error": "No files uploaded."}

    uploads: List[Tuple[str, bytes]] = []
    for uf in files:
        name = uf.filename or "brief.pdf"
        ext = os.path.splitext(name)[1].lower()
        if ext != ".pdf":
            return {"error": f"Only PDFs supported. Got: {name}"}
        pdf_bytes = await uf.read()
        uploads.append((name, pdf_bytes))

    try:
        results = retrieve_similar_cases_from_pdf_uploads(
            uploads,
            k=k,
        )
        return {
            "num_briefs": len(uploads),
            "k": k,
            "cases": [asdict(card) for card in results],
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/judge")
async def judge_case(
    files: List[UploadFile] = File(...),
    redact: bool = Form(False),  # placeholder for later
):
    # Save uploads to temp files
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

        raw = run_prediction_with_uploaded_pdfs(tmp_paths, prompt + PROMPT_APPEND)

        case_summary, case_decision = parse_gpt_response(raw)

        return {
            "raw": raw,
            "case_summary": case_summary,
            "case_decision": case_decision,
            "num_documents": len(tmp_paths),
            "redact": redact,
        }

    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except:
                pass
