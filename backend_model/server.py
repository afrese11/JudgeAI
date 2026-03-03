# server.py
import os
import sys
import tempfile
import logging
import importlib
from typing import List, Optional, Tuple

import httpx
from pydantic import BaseModel

from fastapi import FastAPI, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware

from judge_core import (
    get_top_k_retrieval_for_uploaded_pdfs,
    run_prediction_with_uploaded_pdfs,
)

# Ensure project root is on path so rag_context can be imported (e.g. when run from backend_model)
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
logger = logging.getLogger("backend_server")


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

create_client = None
try:
    supabase_module = importlib.import_module("supabase")
    create_client = getattr(supabase_module, "create_client", None)
except Exception:
    create_client = None

supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and create_client:
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
    logger.info(
        "[judge] Request received: files=%d redact=%s",
        len(files) if files else 0,
        redact,
    )
    '''
    if not _authorized(x_judgeai_passcode):
        logger.warning("[judge] Unauthorized request")
        return {"error": "Unauthorized"}
    '''

    uploads: List[Tuple[str, bytes]] = []
    tmp_paths: List[str] = []
    try:
        if not files:
            logger.warning("[judge] No files uploaded.")
            return {"error": "No files uploaded."}

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

        retrieval_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))
        retrieval_error = None
        top_k_retrieval = {"retrieved_cases": []}
        logger.info(
            "[judge] Starting retrieval: k=%d uploads=%d",
            retrieval_k,
            len(uploads),
        )

        try:
            top_k_retrieval = get_top_k_retrieval_for_uploaded_pdfs(
                uploads=uploads,
                k=retrieval_k,
            )
            logger.info(
                "[judge] Retrieval complete: retrieved_cases=%d",
                len(top_k_retrieval.get("retrieved_cases", [])),
            )
        except Exception as e:
            retrieval_error = f"{type(e).__name__}: {str(e)}"
            logger.exception("[judge] Retrieval failed")

        similar_cases = top_k_retrieval.get("retrieved_cases", [])

        prompt = PROMPT_TEMPLATE.format(
            num_docs=len(tmp_paths),
            retrieval_k=retrieval_k,
        )
        model = os.getenv("OPENAI_MODEL", "gpt-5")
        logger.info(
            "[judge] Prepared model request: model=%s num_docs=%d retrieval_cases=%d",
            model,
            len(tmp_paths),
            len(similar_cases),
        )

        logger.info("[judge] Calling run_prediction_with_uploaded_pdfs")
        raw = run_prediction_with_uploaded_pdfs(
            tmp_paths,
            prompt,
            top_k_retrieval=top_k_retrieval,
            model=model,
        )
        logger.info("[judge] Model call completed: raw_chars=%d", len(raw or ""))

        logger.info("[judge] Parsing model response sections")
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
            "similar_cases": similar_cases,
            "retrieval_error": retrieval_error,
            "redact": redact,
            "model": model,
        }

    except Exception as e:
        logger.exception("[judge] Endpoint failed with exception")
        return {"error": f"{type(e).__name__}: {str(e)}"}

    finally:
        logger.info("[judge] Cleaning up temp files: count=%d", len(tmp_paths))
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                logger.warning("[judge] Failed to delete temp file: %s", p)
                pass
        logger.info("[judge] Request complete")


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
    logger.info(
        "[judge-from-storage] Request received: paths=%d redact=%s cleanup=%s",
        len(req.paths) if req.paths else 0,
        req.redact,
        req.cleanup,
    )
    if not _authorized(x_judgeai_passcode):
        logger.warning("[judge-from-storage] Unauthorized request")
        return {"error": "Unauthorized"}

    if not supabase:
        logger.error("[judge-from-storage] Supabase not configured")
        return {"error": "Supabase is not configured on the server."}

    if not req.paths:
        logger.warning("[judge-from-storage] No storage paths provided")
        return {"error": "No storage paths provided."}

    tmp_paths: List[str] = []
    client = httpx.AsyncClient(timeout=120)

    try:
        # Download each PDF using signed download URLs
        for path in req.paths:
            logger.info("[judge-from-storage] Creating signed URL for: %s", path)
            signed = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(path, 60 * 15)
            data = getattr(signed, "data", None) if not isinstance(signed, dict) else signed.get("data")

            # supabase might return "signedURL" or "signedUrl" depending on lib version
            url = None
            if data:
                url = data.get("signedURL") or data.get("signedUrl") or data.get("signed_url")

            if not url:
                logger.error("[judge-from-storage] Could not create signed URL for: %s", path)
                return {"error": f"Failed to create signed download URL for {path}"}

            logger.info("[judge-from-storage] Downloading file from signed URL")
            r = await client.get(url)
            r.raise_for_status()
            logger.info(
                "[judge-from-storage] Download complete: path=%s bytes=%d",
                path,
                len(r.content),
            )

            fd, local_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            with open(local_path, "wb") as out:
                out.write(r.content)

            tmp_paths.append(local_path)
            logger.info("[judge-from-storage] Wrote temp file: %s", local_path)

        prompt = PROMPT_TEMPLATE.format(num_docs=len(tmp_paths))
        model = os.getenv("OPENAI_MODEL", "gpt-5")
        logger.info(
            "[judge-from-storage] Calling model: model=%s num_docs=%d",
            model,
            len(tmp_paths),
        )

        raw = run_prediction_with_uploaded_pdfs(tmp_paths, prompt, model=model)
        logger.info("[judge-from-storage] Model call completed: raw_chars=%d", len(raw or ""))
        case_summary, case_decision = parse_gpt_response(raw)
        logger.info(
            "[judge-from-storage] Parsed response: has_case_summary=%s has_case_decision=%s",
            bool(case_summary),
            bool(case_decision),
        )

        # Optional cleanup: delete stored objects after processing
        if req.cleanup:
            logger.info(
                "[judge-from-storage] Cleaning up storage objects: count=%d",
                len(req.paths),
            )
            try:
                supabase.storage.from_(SUPABASE_BUCKET).remove(req.paths)
            except Exception:
                logger.warning("[judge-from-storage] Failed to clean up storage objects")
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
        logger.exception("[judge-from-storage] Endpoint failed with exception")
        return {"error": f"{type(e).__name__}: {str(e)}"}

    finally:
        logger.info("[judge-from-storage] Cleaning up temp files: count=%d", len(tmp_paths))
        await client.aclose()
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                logger.warning("[judge-from-storage] Failed to delete temp file: %s", p)
                pass
        logger.info("[judge-from-storage] Request complete")
