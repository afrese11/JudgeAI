# server.py
import os
import sys
import tempfile
import logging
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

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
logger = logging.getLogger("backend_server")


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

You are also provided with topically similar prior cases retrieved from an internal database.
Use them only as persuasive context for legal framing and reasoning patterns.
Do NOT treat them as binding authority unless the uploaded briefs themselves support that use.
If there is any conflict between uploaded documents and retrieved context, prioritize uploaded documents.

Your task is to produce TWO separate PREDICTIVE outputs:

**OUTPUT 1: CASE SUMMARY DOCUMENT (PREDICTIVE)**
Predict and provide:
1. A brief 3-5 sentence summary of the key legal issues on appeal.
2. Statement of the lower court's decision (what ruling is being appealed from).
3. Your PREDICTED recommendation for the length of oral argument, based on the legal complexity of the issues on appeal.
4. An explanation of the case complexity and your reasoning for the oral argument time recommendation.
5. A short "Retrieved Case Signals (Top-{retrieval_k})" subsection that:
   - names 2-3 most relevant retrieved cases by cas
   
   e_id;
   - includes each listed case's case_type label (e.g., criminal, civil, immigration);
   - states what issue/doctrine/procedural similarity each one contributes;
   - briefly explains how (if at all) those signals affected your prediction.

**OUTPUT 2: CASE DECISION DOCUMENT (PREDICTIVE)**
Predict and provide:
1. A written judicial opinion that decides all of the issues raised on the appeal, based on your analysis of the arguments presented.
2. Your PREDICTED determination of whether the case should be AFFIRMED, REVERSED, VACATED, or another disposition.
3. Legal reasoning supporting your predicted decision, citing the arguments and law from the briefs.
4. A short "Use of Retrieved Cases (Top-{retrieval_k})" subsection identifying:
   - which retrieved cases were most persuasive;
   - each cited case's case_type;
   - why they were persuasive (fact pattern, posture, doctrine);
   - any retrieved cases you discounted and why.

Please structure your response exactly as follows:
""".strip()

PROMPT_APPEND = """
===CASE SUMMARY===
[Your predicted case summary here]

===CASE DECISION===
[Your predicted case decision here]
[Include a brief "Use of Retrieved Cases (Top-{retrieval_k})" subsection inside this section.]

""".strip()

RETRIEVAL_CONTEXT_TEMPLATE = """
You are also provided with topically similar prior cases retrieved from an internal database.
Use them only as persuasive context for legal framing and reasoning patterns.
Do NOT treat them as binding authority unless the uploaded briefs themselves support that use.
If there is any conflict between uploaded documents and retrieved context, prioritize uploaded documents.

===RETRIEVED SIMILAR CASE CONTEXT===
{retrieved_context}
""".strip()


def build_retrieval_context(results) -> str:
    if not results:
        return "No similar cases were retrieved."

    chunks: List[str] = []
    for idx, card in enumerate(results, start=1):
        chunks.append(
            (
                f"[Similar Case {idx}] "
                f"case_id={card.case_id}; "
                f"score={card.score:.4f}; "
                f"case_type={card.case_type or 'unknown'}\n"
                f"{card.case_card_text.strip()}"
            )
        )
    return "\n\n".join(chunks)


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
        logger.warning("[retrieve-similar] No files uploaded.")
        return {"error": "No files uploaded."}

    logger.info(
        "[retrieve-similar] Request received: file_count=%d k=%d",
        len(files),
        k,
    )
    uploads: List[Tuple[str, bytes]] = []
    for uf in files:
        name = uf.filename or "brief.pdf"
        ext = os.path.splitext(name)[1].lower()
        if ext != ".pdf":
            logger.warning(
                "[retrieve-similar] Rejected non-pdf upload: filename=%s ext=%s",
                name,
                ext,
            )
            return {"error": f"Only PDFs supported. Got: {name}"}
        pdf_bytes = await uf.read()
        uploads.append((name, pdf_bytes))
        logger.info(
            "[retrieve-similar] Parsed upload: filename=%s bytes=%d",
            name,
            len(pdf_bytes),
        )

    try:
        logger.info(
            "[retrieve-similar] Starting retrieval pipeline for %d upload(s).",
            len(uploads),
        )
        results = retrieve_similar_cases_from_pdf_uploads(
            uploads,
            k=k,
        )
        logger.info(
            "[retrieve-similar] Retrieval finished: returned_cases=%d case_ids=%s",
            len(results),
            [r.case_id for r in results],
        )
        return {
            "num_briefs": len(uploads),
            "k": k,
            "cases": [asdict(card) for card in results],
        }
    except Exception as e:
        logger.exception("[retrieve-similar] Retrieval failed: %s", e)
        return {"error": str(e)}


@app.post("/api/judge")
async def judge_case(
    files: List[UploadFile] = File(...),
    redact: bool = Form(False),  # placeholder for later
    retrieval_k: int = Form(3),
):
    # Save uploads to temp files
    tmp_paths: List[str] = []
    uploads: List[Tuple[str, bytes]] = []
    try:
        if not files:
            logger.warning("[judge] No files uploaded.")
            return {"error": "No files uploaded."}
        if retrieval_k < 1:
            logger.warning("[judge] Invalid retrieval_k=%d", retrieval_k)
            return {"error": "retrieval_k must be >= 1"}

        logger.info(
            "[judge] Request received: file_count=%d retrieval_k=%d redact=%s",
            len(files),
            retrieval_k,
            redact,
        )
        for uf in files:
            name = uf.filename or "uploaded.pdf"
            ext = os.path.splitext(name)[1].lower()
            if ext != ".pdf":
                logger.warning(
                    "[judge] Rejected non-pdf upload: filename=%s ext=%s",
                    name,
                    ext,
                )
                return {"error": f"Only PDFs supported right now. Got: {name}"}
            pdf_bytes = await uf.read()
            uploads.append((name, pdf_bytes))
            logger.info(
                "[judge] Parsed upload: filename=%s bytes=%d",
                name,
                len(pdf_bytes),
            )

            fd, path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            with open(path, "wb") as out:
                out.write(pdf_bytes)
            tmp_paths.append(path)
            logger.info("[judge] Wrote temp file: %s", path)

        retrieval_error = None
        try:
            logger.info(
                "[judge] Starting top-k retrieval from uploaded briefs (k=%d).",
                retrieval_k,
            )
            similar_cases = retrieve_similar_cases_from_pdf_uploads(
                uploads,
                k=retrieval_k,
            )
            logger.info(
                "[judge] Retrieval finished: returned_cases=%d case_ids=%s",
                len(similar_cases),
                [r.case_id for r in similar_cases],
            )
        except Exception as e:
            similar_cases = []
            retrieval_error = str(e)
            logger.exception("[judge] Retrieval failed, continuing without retrieval context: %s", e)
        retrieval_context = build_retrieval_context(similar_cases)
        logger.info(
            "[judge] Retrieval context status: retrieval_error=%s context_preview=%s",
            retrieval_error,
            retrieval_context[:180].replace("\n", " "),
        )

        prompt = PROMPT_TEMPLATE.format(num_docs=len(tmp_paths), retrieval_k=retrieval_k)
        prompt = (
            f"{prompt}\n\n"
            f"{RETRIEVAL_CONTEXT_TEMPLATE.format(retrieved_context=retrieval_context)}\n\n"
            f"{PROMPT_APPEND.format(retrieval_k=retrieval_k)}"
        )
        logger.info("[judge] Prompt constructed: chars=%d", len(prompt))

        raw = run_prediction_with_uploaded_pdfs(tmp_paths, prompt)
        logger.info("[judge] Model response received: chars=%d", len(raw or ""))

        case_summary, case_decision = parse_gpt_response(raw)
        logger.info(
            "[judge] Parsed model response: has_case_summary=%s has_case_decision=%s",
            bool(case_summary),
            bool(case_decision),
        )

        return {
            "raw": raw,
            "case_summary": case_summary,
            "case_decision": case_decision,
            "num_documents": len(tmp_paths),
            "retrieval_k": retrieval_k,
            "similar_cases": [asdict(card) for card in similar_cases],
            "retrieval_error": retrieval_error,
            "redact": redact,
        }

    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
                logger.info("[judge] Deleted temp file: %s", p)
            except:
                logger.warning("[judge] Failed to delete temp file: %s", p)
