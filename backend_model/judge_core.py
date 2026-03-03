# judge_core.py
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def build_prediction_prompt_with_retrieval_context(
    prompt_text: str,
    top_k_retrieval: Dict[str, Any] | None,
) -> str:
    """Inject retrieved-case context into the user prompt in a structured way."""
    retrieved_context = parse_top_k_retrieval_dict_to_prompt_context(top_k_retrieval)
    return (
        f"{prompt_text.strip()}\n\n"
        "===TOP_K_RETRIEVED_CASES===\n"
        "The following cases are top-k semantically related prior cases retrieved from an internal corpus.\n"
        "Use them as persuasive comparative context only; do not treat them as binding authority unless\n"
        "supported by law in the uploaded briefs.\n\n"
        f"{retrieved_context}\n\n"
        "===INSTRUCTIONS_FOR_USING_TOP_K_CASES===\n"
        "In your final prediction, explicitly incorporate these retrieved cases by:\n"
        "1) Comparing fact patterns, issues, and procedural posture to the current appeal.\n"
        "2) Explaining whether those analogies support or cut against each potential disposition.\n"
        "3) Prioritizing the uploaded documents whenever there is tension with retrieved-case context.\n"
        "4) Grounding the final outcome in the uploaded record, while using top-k cases to improve legal reasoning.\n"
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

    cards = retrieve_similar_cases_from_pdf_uploads(uploads, **kwargs)

    retrieved_cases: List[Dict[str, Any]] = []
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

    return {"retrieved_cases": retrieved_cases}


def run_prediction_with_uploaded_pdfs(
    pdf_paths: List[str],
    prompt_text: str,
    top_k_retrieval: Dict[str, Any] | None = None,
    model: str | None = None,
) -> str:
    """
    Upload PDFs, call the Responses API with file inputs, return output text.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-5")
    final_prompt_text = build_prediction_prompt_with_retrieval_context(
        prompt_text=prompt_text,
        top_k_retrieval=top_k_retrieval,
    )

    uploaded_file_ids: List[str] = []
    file_handles = []

    try:
        # Upload PDFs (recommended purpose for model inputs is "user_data")
        for path in pdf_paths:
            fh = open(path, "rb")
            file_handles.append(fh)

            uploaded = client.files.create(
                file=fh,
                purpose="user_data",
            )
            uploaded_file_ids.append(uploaded.id)

        # Build Responses API input with multiple files + your prompt text.
        # (For PDF inputs, Responses API is the recommended path.) :contentReference[oaicite:1]{index=1}
        content = []
        for fid in uploaded_file_ids:
            content.append({"type": "input_file", "file_id": fid})
        content.append({"type": "input_text", "text": final_prompt_text})

        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        # The docs show using response.output_text for Responses API output. :contentReference[oaicite:2]{index=2}
        return response.output_text or ""

    finally:
        # Close local file handles
        for fh in file_handles:
            try:
                fh.close()
            except Exception:
                pass

        # Best-effort cleanup: delete uploaded OpenAI files
        for fid in uploaded_file_ids:
            try:
                client.files.delete(fid)
            except Exception:
                pass
