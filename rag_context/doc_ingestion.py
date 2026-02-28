#!/usr/bin/env python3
"""
Ingest an 8th Circuit decision PDF into:
  1) documents (raw-ish cleaned text)
  2) chunks (retrieval units + embeddings)

Schema: Eighth_Circuit_RAG_MVP_Schema.md
- cases: case-level metadata (must exist; insert case row first)
- documents: doc_id, case_id, doc_type, text_clean, page_count, token_count, created_at
- chunks: chunk_id, case_id, doc_id, section_type, heading_path, chunk_text, token_count, embedding VECTOR(1536), created_at

Assumptions:
- Postgres + pgvector is running
- You already inserted cases rows before ingesting documents.
"""

import os
import re
import json
import uuid
from dataclasses import dataclass
from typing import List, Tuple

import psycopg
from psycopg.rows import dict_row

# PDF text extraction (pip install pypdf)
from pypdf import PdfReader

# Token counting is optional; this fallback is "good enough" for guardrails.
# If you want exact-ish tokens, you can add tiktoken later.
def rough_token_count(s: str) -> int:
    # ~4 chars/token heuristic (very rough)
    return max(1, len(s) // 4)

# OpenAI client uses API_KEY from config.py (loads OPENAI_API_KEY via .env).
from openai import OpenAI

from rag_context.config import DATABASE_URL, API_KEY

# Schema: chunks.embedding is VECTOR(1536) for text-embedding-3-small
EMBEDDING_DIMENSION = 1536


def infer_section_type(heading: str) -> str:
    """
    Map heading to schema section_type: facts, argument, standard_of_review, statute.
    Used for chunks.section_type per MVP schema.
    """
    h = heading.upper()
    if "FACT" in h or "BACKGROUND" in h:
        return "facts"
    if "STANDARD OF REVIEW" in h:
        return "standard_of_review"
    if "STATUTE" in h or "STATUTORY" in h or re.search(r"\b\d+\s*U\.?S\.?C\.?", heading, re.I):
        return "statute"
    if "ARGUMENT" in h or "DISCUSSION" in h or "ANALYSIS" in h or "HOLDING" in h:
        return "argument"
    return "argument"  # default for opinion body


@dataclass
class IngestConfig:
    db_url: str
    embedding_model: str
    chunk_target_tokens: int
    chunk_max_tokens: int
    heading_min_len: int


def extract_pdf_text(pdf_path: str) -> Tuple[str, int]:
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    text = "\n".join(pages)
    return text, len(reader.pages)


def clean_legal_text(text: str) -> str:
    # Minimal cleaning (don’t overdo; decisions have meaningful formatting)
    text = text.replace("\r", "\n")
    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    # strip obvious page number lines like "Page 3" / "3" alone
    text = re.sub(r"\n\s*(Page\s+\d+|\d+)\s*\n", "\n", text, flags=re.IGNORECASE)
    return text.strip()


def looks_like_heading(line: str, heading_min_len: int) -> bool:
    s = line.strip()
    if len(s) < heading_min_len:
        return False
    # common signals: all caps-ish, roman numerals, numbered headings
    if re.match(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)\.?\s+", s):
        return True
    if re.match(r"^\d+(\.\d+)*\s+", s):
        return True
    if s.isupper() and len(s) <= 120:
        return True
    # headings often end with ":" (less common in opinions, but possible)
    if s.endswith(":") and len(s) <= 140:
        return True
    return False


def split_into_sections(text: str, heading_min_len: int) -> List[Tuple[str, str]]:
    """
    Returns list of (heading_path, section_text).
    Very lightweight: uses line-based heading detection.
    """
    lines = text.split("\n")
    sections: List[Tuple[str, List[str]]] = []
    cur_heading = "ROOT"
    cur_buf: List[str] = []

    def flush():
        nonlocal cur_buf, cur_heading
        section_text = "\n".join(cur_buf).strip()
        if section_text:
            sections.append((cur_heading, section_text))
        cur_buf = []

    for line in lines:
        if looks_like_heading(line, heading_min_len):
            flush()
            cur_heading = line.strip()
        else:
            cur_buf.append(line)
    flush()

    # If we failed to find headings, return whole doc as one section
    if len(sections) == 0:
        return [("ROOT", text)]
    return sections


def chunk_section(
    heading: str,
    section_text: str,
    target_tokens: int,
    max_tokens: int,
) -> List[Tuple[str, str]]:
    """
    Chunks a section by paragraphs until target/max token thresholds.
    """
    paras = [p.strip() for p in section_text.split("\n\n") if p.strip()]
    chunks: List[Tuple[str, str]] = []

    buf: List[str] = []
    buf_tokens = 0

    for p in paras:
        p_tokens = rough_token_count(p)

        # If single paragraph is huge, hard-split by sentences as fallback
        if p_tokens > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            sbuf = []
            stoks = 0
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                t = rough_token_count(sent)
                if stoks + t > max_tokens and sbuf:
                    chunks.append((heading, " ".join(sbuf).strip()))
                    sbuf, stoks = [], 0
                sbuf.append(sent)
                stoks += t
            if sbuf:
                chunks.append((heading, " ".join(sbuf).strip()))
            continue

        # Normal paragraph packing
        if buf_tokens + p_tokens > max_tokens and buf:
            chunks.append((heading, "\n\n".join(buf).strip()))
            buf, buf_tokens = [], 0

        buf.append(p)
        buf_tokens += p_tokens

        if buf_tokens >= target_tokens:
            chunks.append((heading, "\n\n".join(buf).strip()))
            buf, buf_tokens = [], 0

    if buf:
        chunks.append((heading, "\n\n".join(buf).strip()))

    return chunks


def embed_texts(client: OpenAI, model: str, texts: List[str]) -> List[List[float]]:
    # OpenAI embeddings endpoint supports batching
    resp = client.embeddings.create(model=model, input=texts)
    # Ensure original order
    return [d.embedding for d in resp.data]


def upsert_document(
    conn: psycopg.Connection,
    case_id: str,
    doc_type: str,
    text_clean: str,
    page_count: int,
    token_count: int,
) -> str:
    """
    Inserts a new document row for (case_id, doc_type) OR updates existing.
    Returns doc_id. Table: doc_id, case_id, doc_type, text_clean, page_count, token_count, created_at.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT doc_id
            FROM documents
            WHERE case_id = %s AND doc_type = %s
            """,
            (case_id, doc_type),
        )
        row = cur.fetchone()
        if row:
            doc_id = row["doc_id"]
            cur.execute(
                """
                UPDATE documents
                SET text_clean = %s,
                    page_count = %s,
                    token_count = %s
                WHERE doc_id = %s
                """,
                (text_clean, page_count, token_count, doc_id),
            )
            return doc_id

        doc_id = str(uuid.uuid4())
        cur.execute(
            """
            INSERT INTO documents (doc_id, case_id, doc_type, text_clean, page_count, token_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (doc_id, case_id, doc_type, text_clean, page_count, token_count),
        )
        return doc_id


def insert_chunks(
    conn: psycopg.Connection,
    case_id: str,
    doc_id: str,
    chunks: List[Tuple[str, str]],
    embeddings: List[List[float]],
) -> int:
    """
    Inserts chunks; deletes prior chunks for same doc_id to keep idempotent behavior.
    """
    assert len(chunks) == len(embeddings)

    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))

        for (heading, chunk_text), emb in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            token_count = rough_token_count(chunk_text)
            section_type = infer_section_type(heading)
            heading_path = heading

            if len(emb) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"embedding length {len(emb)} != schema dimension {EMBEDDING_DIMENSION}; "
                    "use text-embedding-3-small or adjust VECTOR(n) in chunks"
                )

            cur.execute(
                """
                INSERT INTO chunks (chunk_id, case_id, doc_id, section_type, heading_path, chunk_text, token_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (chunk_id, case_id, doc_id, section_type, heading_path, chunk_text, token_count, emb),
            )

    return len(chunks)


def ensure_case_exists(conn: psycopg.Connection, case_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM cases WHERE case_id = %s", (case_id,))
        if cur.fetchone() is None:
            raise SystemExit(
                f"case_id={case_id} not found in cases table. "
                "Insert the case row first (you said you did)."
            )


# Default train dataset path relative to this script (rag_context/)
DEFAULT_TRAIN_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sept. 2024 decisions"
)


def _iter_train_decision_pdfs(train_dir: str):
    """
    Yield (case_id, pdf_path) for each case in train_dir.

    Supports both layouts:
    1) Nested: <train_dir>/<case_id>/<case_id> decision.pdf
    2) Flat:   <train_dir>/<case_id> decision.pdf
    """
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train dataset directory not found: {train_dir}")
    for name in sorted(os.listdir(train_dir)):
        if name.startswith("."):
            continue
        path = os.path.join(train_dir, name)

        # Nested layout: <train_dir>/<case_id>/<case_id> decision.pdf
        if os.path.isdir(path):
            case_id = name
            pdf_name = f"{case_id} decision.pdf"
            pdf_path = os.path.join(path, pdf_name)
            if not os.path.isfile(pdf_path):
                continue
            yield case_id, pdf_path
            continue

        # Flat layout: <train_dir>/<case_id> decision.pdf
        if os.path.isfile(path) and name.lower().endswith(" decision.pdf"):
            case_id = name[: -len(" decision.pdf")].strip()
            if not case_id:
                continue
            yield case_id, path


def main():
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL must be set (e.g. in config.py / .env)")
    if not API_KEY:
        raise SystemExit("OPENAI_API_KEY must be set in .env (used for embeddings)")

    cfg = IngestConfig(
        db_url=DATABASE_URL,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_target_tokens=int(os.getenv("CHUNK_TARGET_TOKENS", "900")),
        chunk_max_tokens=int(os.getenv("CHUNK_MAX_TOKENS", "1200")),
        heading_min_len=int(os.getenv("HEADING_MIN_LEN", "8")),
    )

    client = OpenAI(api_key=API_KEY)  # API_KEY from config.py
    train_dir = os.path.abspath(DEFAULT_TRAIN_DATASET)
    doc_type = os.getenv("DOC_TYPE", "decision")

    total = 0
    errors: List[Tuple[str, str]] = []
    for case_id, pdf_path in _iter_train_decision_pdfs(train_dir):
        try:
            raw_text, page_count = extract_pdf_text(pdf_path)
            text_clean = clean_legal_text(raw_text)
            doc_token_count = rough_token_count(text_clean)

            with psycopg.connect(cfg.db_url) as conn:
                conn.execute("SET statement_timeout = '60s'")
                ensure_case_exists(conn, case_id)

                doc_id = upsert_document(
                    conn=conn,
                    case_id=case_id,
                    doc_type=doc_type,
                    text_clean=text_clean,
                    page_count=page_count,
                    token_count=doc_token_count,
                )

                sections = split_into_sections(text_clean, cfg.heading_min_len)
                all_chunks: List[Tuple[str, str]] = []
                for heading, section_text in sections:
                    all_chunks.extend(
                        chunk_section(
                            heading=heading,
                            section_text=section_text,
                            target_tokens=cfg.chunk_target_tokens,
                            max_tokens=cfg.chunk_max_tokens,
                        )
                    )

                texts = [c[1] for c in all_chunks]
                embeddings: List[List[float]] = []
                BATCH = 64
                for i in range(0, len(texts), BATCH):
                    embeddings.extend(embed_texts(client, cfg.embedding_model, texts[i : i + BATCH]))

                n = insert_chunks(conn, case_id, doc_id, all_chunks, embeddings)

            total += 1
            print(json.dumps({
                "case_id": case_id,
                "doc_id": doc_id,
                "doc_type": doc_type,
                "pages": page_count,
                "doc_token_count": doc_token_count,
                "chunks_inserted": n,
                "embedding_model": cfg.embedding_model,
            }, indent=2))
        except Exception as e:
            errors.append((case_id, str(e)))
            print(f"❌ {case_id}: {e}")

    print(f"\nDone. Ingested documents for {total} cases.")
    if errors:
        print(f"Failed ({len(errors)}): {[c for c, _ in errors]}")


if __name__ == "__main__":
    main()
