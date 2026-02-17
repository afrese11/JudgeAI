"""
New Case Retrieval Pipeline
Supports arbitrary number of uploaded briefs.
Performs retrieval only (no database writes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import os
import re
import psycopg
from psycopg.rows import dict_row

from config import DATABASE_URL, API_KEY


# -------------------------
# Configuration
# -------------------------

DEFAULT_K = 3
DEFAULT_CANDIDATE_CHUNKS = 80
DEFAULT_MAX_CHUNKS_PER_CASE = 4


# -------------------------
# Data Models
# -------------------------

@dataclass
class BriefInput:
    """
    Represents one uploaded brief.
    """
    label: str
    text: str


@dataclass
class RetrievedCaseCard:
    case_id: str
    score: float
    case_card_text: str
    issue_tags: Optional[List[str]] = None
    statute_tags: Optional[List[str]] = None
    doctrine_tags: Optional[List[str]] = None


# -------------------------
# Text Processing
# -------------------------

def _clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_query_fingerprint(
    briefs: List[BriefInput],
    max_chars_per_brief: int = 12000,
    max_total_chars: int = 24000,
) -> str:
    """
    Build a structured retrieval fingerprint from N uploaded briefs.
    """

    parts: List[str] = [
        "NEW CASE (QUERY) — RETRIEVAL FINGERPRINT",
        "",
        "EMBEDDING INSTRUCTIONS:",
        "- Focus on legal issues, procedural posture, standards of review.",
        "- Emphasize statutes, doctrines, constitutional claims.",
        "- Ignore boilerplate and formatting sections.",
        "",
    ]

    remaining = max_total_chars

    for brief in briefs:
        if remaining <= 0:
            break

        label = (brief.label or "brief").strip()
        text = _clean_text(brief.text)
        text = text[:max_chars_per_brief]
        text = text[:remaining]

        remaining -= len(text)

        parts.append(f"[{label.upper()} — selected text]")
        parts.append(text)
        parts.append("")

    return "\n".join(parts).strip()


# -------------------------
# Embedding (OpenAI Example)
# -------------------------

def embed_text_openai(text: str) -> List[float]:
    """
    Embeds fingerprint text using OpenAI.
    Model must match dimension used in chunks.embedding.
    """

    from openai import OpenAI

    client = OpenAI(api_key=API_KEY)

    response = client.embeddings.create(
        model=os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
        input=text,
    )

    return response.data[0].embedding


# -------------------------
# Retrieval Logic
# -------------------------

def retrieve_top_k_case_cards(
    conn: psycopg.Connection,
    query_embedding: List[float],
    k: int,
    candidate_chunks: int,
    max_chunks_per_case: int = DEFAULT_MAX_CHUNKS_PER_CASE,
) -> List[RetrievedCaseCard]:

    if k <= 0:
        return []

    emb_literal = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

    # Step 1: Retrieve nearest chunks
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
              case_id,
              chunk_id,
              (1.0 - (embedding <=> %s::vector)) AS sim
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (emb_literal, emb_literal, candidate_chunks),
        )
        chunk_rows = cur.fetchall()

    # Step 2: Aggregate similarities per case
    per_case: Dict[str, List[float]] = {}

    for row in chunk_rows:
        cid = row["case_id"]
        sim = float(row["sim"])
        per_case.setdefault(cid, []).append(sim)

    case_scores: List[Tuple[str, float]] = []

    for cid, sims in per_case.items():
        sims_sorted = sorted(sims, reverse=True)[:max_chunks_per_case]
        score = sum(sims_sorted)
        case_scores.append((cid, score))

    case_scores.sort(key=lambda x: x[1], reverse=True)

    top_case_ids = [cid for cid, _ in case_scores[:k]]
    score_lookup = {cid: score for cid, score in case_scores[:k]}

    if not top_case_ids:
        return []

    # Step 3: Fetch case_cards
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
              case_id,
              case_card_text,
              issue_tags,
              statute_tags,
              doctrine_tags
            FROM case_cards
            WHERE case_id = ANY(%s);
            """,
            (top_case_ids,),
        )
        cards = cur.fetchall()

    card_map = {c["case_id"]: c for c in cards}

    results: List[RetrievedCaseCard] = []

    for cid in top_case_ids:
        card = card_map.get(cid)
        if not card:
            continue

        results.append(
            RetrievedCaseCard(
                case_id=cid,
                score=score_lookup[cid],
                case_card_text=card["case_card_text"],
                issue_tags=card.get("issue_tags"),
                statute_tags=card.get("statute_tags"),
                doctrine_tags=card.get("doctrine_tags"),
            )
        )

    return results


# -------------------------
# End-to-End API
# -------------------------

def retrieve_similar_cases_for_new_case(
    *,
    db_url: str,
    briefs: List[BriefInput],
    k: int = DEFAULT_K,
    candidate_chunks: int = DEFAULT_CANDIDATE_CHUNKS,
) -> List[RetrievedCaseCard]:
    """
    Full pipeline:
      - Build fingerprint
      - Embed
      - Retrieve top-k cases
    """

    if not briefs:
        return []

    fingerprint = build_query_fingerprint(briefs)
    embedding = embed_text_openai(fingerprint)

    with psycopg.connect(db_url) as conn:
        return retrieve_top_k_case_cards(
            conn=conn,
            query_embedding=embedding,
            k=k,
            candidate_chunks=candidate_chunks,
        )
# -------------------------
# Test dataset iteration
# -------------------------

DEFAULT_TEST_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "AIth Circuit Test Dataset",
    "test",
)


def _iter_test_cases(test_dir: str):
    """
    Yield (case_dir, doc_list) for each case in test_dir.
    doc_list is [(label, pdf_path), ...] for every PDF in the case directory,
    so multiple briefs (e.g. appellant, appellee, decision) are included.
    """
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test dataset directory not found: {test_dir}")
    for name in sorted(os.listdir(test_dir)):
        case_dir = os.path.join(test_dir, name)
        if not os.path.isdir(case_dir) or name.startswith("."):
            continue
        doc_list: List[Tuple[str, str]] = []
        for f in sorted(os.listdir(case_dir)):
            if not f.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(case_dir, f)
            if not os.path.isfile(pdf_path):
                continue
            # Label from filename: strip case prefix and ".pdf" (e.g. "23-2869 appellant.pdf" -> "appellant")
            label = f[:-4].strip()
            if label.lower().startswith(name.lower()):
                label = label[len(name) :].strip() or "document"
            doc_list.append((label or "document", pdf_path))
        if not doc_list:
            continue
        yield case_dir, doc_list


def main() -> None:
    try:
        from rag_context.case_ingestion import extract_text_from_pdf
    except ImportError:
        from case_ingestion import extract_text_from_pdf

    test_dir = os.path.abspath(DEFAULT_TEST_DATASET)

    k = DEFAULT_K
    for case_dir, doc_list in _iter_test_cases(test_dir):
        case_id = os.path.basename(case_dir)
        briefs = [
            BriefInput(label=label, text=extract_text_from_pdf(pdf_path))
            for label, pdf_path in doc_list
        ]
        results = retrieve_similar_cases_for_new_case(
            db_url=DATABASE_URL,
            briefs=briefs,
            k=k,
            candidate_chunks=DEFAULT_CANDIDATE_CHUNKS,
        )
        print(f"\n[{case_id}] ({len(briefs)} doc(s)) top-{k} retrieved:")
        for r in results:
            print(f"  - {r.case_id} (score={r.score:.4f})")


if __name__ == "__main__":
    main()
