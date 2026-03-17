# judge_core.py
import os
import sys
import time
import logging
import threading
from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterator, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load backend_model/.env and project root .env so DATABASE_URL, OPENAI_API_KEY / FINN_API_KEY are available
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)
_root_env = Path(__file__).resolve().parent.parent / ".env"
if _root_env.exists():
    load_dotenv(dotenv_path=_root_env)

# Ensure project root is on path so rag_context can be imported when judge_core is used from server or elsewhere
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from rag_context.top_k_retrieval import (
    retrieve_similar_cases_from_pdf_uploads,
    _split_case_card_main_and_addendum,
)

# Use OPENAI_API_KEY or FINN_API_KEY (rag_context uses FINN_API_KEY in config)
_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("FINN_API_KEY")
if not _api_key:
    raise RuntimeError(
        "Set OPENAI_API_KEY or FINN_API_KEY in .env (e.g. backend_model/.env or project root .env)"
    )
client = OpenAI(api_key=_api_key)
logger = logging.getLogger("judge_core")


def _condense_document_text(
    text: str,
    *,
    per_doc_limit_chars: int,
) -> str:
    """
    Keep a representative slice of long document text.
    We preserve both the opening and closing portions because legal briefs often
    front-load procedural posture and conclude with disposition-focused arguments.
    """
    text = (text or "").strip()
    if per_doc_limit_chars <= 0 or len(text) <= per_doc_limit_chars:
        return text

    if per_doc_limit_chars < 4000:
        return text[:per_doc_limit_chars]

    head = int(per_doc_limit_chars * 0.7)
    tail = per_doc_limit_chars - head
    return (
        text[:head]
        + "\n\n[... document middle omitted for latency control ...]\n\n"
        + text[-tail:]
    )


def _delete_uploaded_openai_files_background(uploaded_file_ids: List[str]) -> None:
    """Best-effort asynchronous cleanup of uploaded OpenAI files."""
    if not uploaded_file_ids:
        return

    def _worker(file_ids: List[str]) -> None:
        with timed_section(
            "judge_core.delete_uploaded_openai_files_async",
            uploaded_file_count=len(file_ids),
        ):
            for fid in file_ids:
                try:
                    client.files.delete(fid)
                except Exception:
                    logger.warning("[cleanup] Failed to delete uploaded OpenAI file: %s", fid)

    threading.Thread(
        target=_worker,
        args=(list(uploaded_file_ids),),
        daemon=True,
    ).start()


def _format_log_fields(fields: dict) -> str:
    if not fields:
        return ""
    return " " + " ".join(f"{k}={v}" for k, v in fields.items())


@contextmanager
def timed_section(section: str, **fields) -> Iterator[None]:
    start = time.perf_counter()
    logger.info("[timing] ENTER section=%s%s", section, _format_log_fields(fields))
    try:
        yield
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.exception(
            "[timing] FAIL section=%s elapsed_ms=%.1f%s",
            section,
            elapsed_ms,
            _format_log_fields(fields),
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "[timing] EXIT section=%s elapsed_ms=%.1f%s",
            section,
            elapsed_ms,
            _format_log_fields(fields),
        )


def parse_top_k_retrieval_dict_to_prompt_context(top_k_retrieval: Dict[str, Any] | None) -> str:
    """
    Convert retrieval payload from get_top_k_retrieval_for_uploaded_pdfs into prompt-ready text.

    Expected input shape:
      {
        "retrieved_cases": [
          {
            "case_id": str,
            "score": float,
            "case_type": str | None,
            "issue_tags": list[str] | None,
            "statute_tags": list[str] | None,
            "doctrine_tags": list[str] | None,
            "procedural_posture": str | None,
            "summary": str,
            ...
          },
          ...
        ]
      }
    """
    if not top_k_retrieval:
        return "No top-k related cases were provided."

    retrieved_cases = top_k_retrieval.get("retrieved_cases")
    if not isinstance(retrieved_cases, list) or not retrieved_cases:
        return "No top-k related cases were retrieved."

    blocks: List[str] = []
    for idx, case in enumerate(retrieved_cases, start=1):
        if not isinstance(case, dict):
            continue

        case_id = case.get("case_id", "unknown")
        score = case.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "unknown"
        case_type = case.get("case_type") or "unknown"
        posture = case.get("procedural_posture") or "unknown"

        issue_tags = case.get("issue_tags") or []
        statute_tags = case.get("statute_tags") or []
        doctrine_tags = case.get("doctrine_tags") or []

        issues_text = ", ".join(issue_tags) if issue_tags else "none"
        statutes_text = ", ".join(statute_tags) if statute_tags else "none"
        doctrines_text = ", ".join(doctrine_tags) if doctrine_tags else "none"

        summary = (case.get("summary") or "").strip()
        if not summary:
            summary = (case.get("case_card_text") or "").strip()
        if not summary:
            summary = "No summary available."

        blocks.append(
            (
                f"[Retrieved Case {idx}]\n"
                f"- case_id: {case_id}\n"
                f"- similarity_score: {score_str}\n"
                f"- case_type: {case_type}\n"
                f"- procedural_posture: {posture}\n"
                f"- issue_tags: {issues_text}\n"
                f"- statute_tags: {statutes_text}\n"
                f"- doctrine_tags: {doctrines_text}\n"
                f"- summary: {summary}"
            )
        )

    if not blocks:
        return "No usable top-k related case entries were found."
    return "\n\n".join(blocks)


def parse_query_signals_to_prompt_context(top_k_retrieval: Dict[str, Any] | None) -> str:
    """
    Convert query_signals from get_top_k_retrieval_for_uploaded_pdfs into
    prompt-ready text for outcome reasoning.
    """
    if not top_k_retrieval:
        return "No query signals were provided."

    query_signals = top_k_retrieval.get("query_signals")
    if not isinstance(query_signals, dict) or not query_signals:
        return "No query signals were returned by retrieval."

    case_type = query_signals.get("case_type") or "unknown"
    posture = query_signals.get("procedural_posture") or "unknown"
    posture_bucket = query_signals.get("posture_bucket") or "unknown"

    standards = query_signals.get("standards_of_review") or []
    statutes = query_signals.get("statute_tags") or []
    doctrines = query_signals.get("doctrine_tags") or []
    issues = query_signals.get("issue_tags") or []

    standards_text = ", ".join(standards) if standards else "none"
    statutes_text = ", ".join(statutes) if statutes else "none"
    doctrines_text = ", ".join(doctrines) if doctrines else "none"
    issues_text = ", ".join(issues) if issues else "none"

    return (
        "[Query Signals]\n"
        f"- case_type: {case_type}\n"
        f"- procedural_posture: {posture}\n"
        f"- posture_bucket: {posture_bucket}\n"
        f"- standards_of_review: {standards_text}\n"
        f"- statute_tags: {statutes_text}\n"
        f"- doctrine_tags: {doctrines_text}\n"
        f"- issue_tags: {issues_text}"
    )


def build_prediction_prompt_with_retrieval_context(
    prompt_text: str,
    top_k_retrieval: Dict[str, Any] | None,
) -> str:
    """Inject retrieved-case context into the user prompt in a structured way."""
    retrieved_context = parse_top_k_retrieval_dict_to_prompt_context(top_k_retrieval)
    query_signals_context = parse_query_signals_to_prompt_context(top_k_retrieval)
    return (
        f"{prompt_text.strip()}\n\n"
        "===QUERY_SIGNALS_ANALYZED===\n"
        "These structured signals were extracted from the uploaded case materials by the retrieval pipeline.\n"
        "Treat them as case-specific anchors for your reasoning about likely outcome.\n\n"
        f"{query_signals_context}\n\n"
        "===TOP_K_RETRIEVED_CASES===\n"
        "The following cases are top-k semantically related prior cases retrieved from an internal corpus.\n"
        "Use them as persuasive comparative context only; do not treat them as binding authority unless\n"
        "supported by law in the uploaded briefs.\n\n"
        f"{retrieved_context}\n\n"
        "===INSTRUCTIONS_FOR_USING_QUERY_SIGNALS_AND_TOP_K_CASES===\n"
        "In your final prediction, explicitly incorporate both query signals and retrieved cases by:\n"
        "1) Citing the query signals (case type, posture, standards, doctrine/statute/issue tags) as part of your outcome analysis.\n"
        "2) Comparing retrieved-case analogies against those query signals and the uploaded briefs.\n"
        "3) Explaining whether each key signal supports or cuts against the predicted disposition.\n"
        "4) Including a short subsection in ===CASE DECISION=== titled 'Use of Query Signals and Retrieved Cases'.\n"
        "5) Prioritizing the uploaded documents whenever there is tension with either query signals or retrieved-case context.\n"
    )


def get_top_k_retrieval_for_uploaded_pdfs(
    uploads: List[Tuple[str, bytes]],
    k: int = 3,
    db_url: str | None = None,
    candidate_chunks: int | None = None,
) -> Dict[str, Any]:
    """
    Helper for run_prediction_with_uploaded_pdfs and other consumers.
    Takes uploaded PDFs (as passed from server.py's judge_case: list of (filename, pdf_bytes)),
    retrieves the top_k similar cases via rag_context.top_k_retrieval, and returns a nested
    JSON structure with extracted features/fields and summary for each retrieved case.

    Args:
        uploads: List of (label, pdf_bytes) e.g. from judge_case [(filename, await uf.read()), ...].
        k: Number of similar cases to return (top_k).
        db_url: Optional database URL; if None, uses DATABASE_URL from .env.
        candidate_chunks: Optional chunk limit for retrieval; if None, uses module default.

    Returns:
        Nested dict suitable for JSON serialization, with structure:
        {
          "query_signals": {
            "case_type": str | null,
            "procedural_posture": str | null,
            "posture_bucket": str | null,
            "standards_of_review": list[str],
            "statute_tags": list[str],
            "doctrine_tags": list[str],
            "issue_tags": list[str]
          },
          "retrieved_cases": [
            {
              "case_id": str,
              "score": float,
              "case_type": str | null,
              "issue_tags": list[str] | null,
              "statute_tags": list[str] | null,
              "doctrine_tags": list[str] | null,
              "procedural_posture": str | null,
              "score_breakdown": { "embed", "doctrine", "statute", "posture", "raw_embed_agg" } | null,
              "summary": str,   # main narrative part of case card (before addendum if present)
              "case_card_text": str   # full case card text
            },
            ...
          ]
        }
    """
    kwargs: Dict[str, Any] = {"k": k}
    kwargs["db_url"] = db_url if db_url is not None else os.getenv("DATABASE_URL")
    if candidate_chunks is not None:
        kwargs["candidate_chunks"] = candidate_chunks

    with timed_section(
        "judge_core.retrieve_similar_cases_from_pdf_uploads",
        k=k,
        upload_count=len(uploads),
    ):
        cards, query_signals, uploaded_brief_texts = retrieve_similar_cases_from_pdf_uploads(
            uploads,
            return_query_signals=True,
            return_extracted_texts=True,
            **kwargs,
        )

    retrieved_cases: List[Dict[str, Any]] = []
    with timed_section(
        "judge_core.serialize_retrieved_cards",
        retrieved_count=len(cards),
    ):
        for card in cards:
            full_text = (card.case_card_text or "").strip()
            main_part, _ = _split_case_card_main_and_addendum(full_text)
            summary = main_part if main_part else full_text

            entry: Dict[str, Any] = {
                "case_id": card.case_id,
                "score": card.score,
                "case_type": card.case_type,
                "issue_tags": card.issue_tags,
                "statute_tags": card.statute_tags,
                "doctrine_tags": card.doctrine_tags,
                "procedural_posture": card.procedural_posture,
                "score_breakdown": asdict(card.score_breakdown) if card.score_breakdown else None,
                "summary": summary,
                "case_card_text": full_text,
            }
            retrieved_cases.append(entry)

    return {
        "query_signals": asdict(query_signals),
        "retrieved_cases": retrieved_cases,
        "uploaded_brief_texts": uploaded_brief_texts,
    }


def run_prediction_with_uploaded_pdfs(
    pdf_paths: List[str],
    prompt_text: str,
    top_k_retrieval: Dict[str, Any] | None = None,
    extracted_brief_texts: Optional[List[Dict[str, str]]] = None,
    model: str | None = None,
) -> str:
    """
    Call Chat Completions with either extracted document text (preferred)
    or uploaded PDF files as fallback; return output text.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-5")
    with timed_section(
        "judge_core.build_prediction_prompt_with_retrieval_context",
        model=model,
        pdf_count=len(pdf_paths),
    ):
        final_prompt_text = build_prediction_prompt_with_retrieval_context(
            prompt_text=prompt_text,
            top_k_retrieval=top_k_retrieval,
        )

    uploaded_file_ids: List[str] = []
    file_handles = []

    try:
        if extracted_brief_texts:
            per_doc_limit_chars = int(os.getenv("CASE_OUTCOME_DOC_CHAR_LIMIT", "18000"))
            total_docs_limit_chars = int(os.getenv("CASE_OUTCOME_TOTAL_DOC_CHAR_LIMIT", "60000"))
            doc_sections: List[str] = []
            total_chars = 0
            for idx, item in enumerate(extracted_brief_texts, start=1):
                label = (item or {}).get("label") or f"document_{idx}"
                text = ((item or {}).get("text") or "").strip()
                if not text:
                    continue
                condensed = _condense_document_text(
                    text,
                    per_doc_limit_chars=per_doc_limit_chars,
                )
                if total_docs_limit_chars > 0 and total_chars >= total_docs_limit_chars:
                    break
                if total_docs_limit_chars > 0 and (total_chars + len(condensed)) > total_docs_limit_chars:
                    remaining = total_docs_limit_chars - total_chars
                    if remaining <= 0:
                        break
                    condensed = _condense_document_text(
                        condensed,
                        per_doc_limit_chars=remaining,
                    )
                if not condensed:
                    continue
                doc_sections.append(
                    f"{'='*60}\nDOCUMENT: {label}\n{'='*60}\n\n{condensed}"
                )
                total_chars += len(condensed)
            combined_docs = "\n\n".join(doc_sections) if doc_sections else "No extracted document text available."
            user_text = f"{final_prompt_text}\n\n===UPLOADED_DOCUMENT_TEXT===\n{combined_docs}"
            logger.info(
                "[case_outcome] Prepared condensed extracted docs: sections=%d chars=%d per_doc_limit=%d total_limit=%d",
                len(doc_sections),
                len(combined_docs),
                per_doc_limit_chars,
                total_docs_limit_chars,
            )
            with timed_section(
                "judge_core.openai_chat_completions_create_text",
                model=model,
                extracted_doc_count=len(doc_sections),
            ):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_text}],
                )
            return response.choices[0].message.content or ""

        # Fallback path: upload PDFs and call Chat Completions with file inputs.
        for path in pdf_paths:
            with timed_section("judge_core.open_pdf_for_upload", path=path):
                fh = open(path, "rb")
                file_handles.append(fh)
            with timed_section("judge_core.openai_file_upload", path=path, model=model):
                uploaded = client.files.create(file=fh, purpose="user_data")
                uploaded_file_ids.append(uploaded.id)

        content: List[Dict[str, Any]] = [{"type": "text", "text": final_prompt_text}]
        for fid in uploaded_file_ids:
            content.append({"type": "file", "file": {"file_id": fid}})

        with timed_section(
            "judge_core.openai_chat_completions_create_files",
            model=model,
            uploaded_file_count=len(uploaded_file_ids),
        ):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
            )
        return response.choices[0].message.content or ""

    finally:
        # Close local file handles
        with timed_section("judge_core.close_local_file_handles", file_handle_count=len(file_handles)):
            for fh in file_handles:
                try:
                    fh.close()
                except Exception:
                    pass

        # Best-effort cleanup in background so it does not block request completion.
        _delete_uploaded_openai_files_background(uploaded_file_ids)
