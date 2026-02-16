import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from openai import OpenAI

from config import API_KEY, DATABASE_URL

load_dotenv()

# Keep this conservative; you can increase once you see typical chunk sizes.
MAX_EVIDENCE_CHARS = 120_000  # rough cap; the model will still have its own context limits

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=API_KEY)
REGEN = True
MAX_CASES = 100

CASE_CARD_SCHEMA: Dict[str, Any] = {
    "name": "case_card",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "case_card_text": {
                "type": "string",
                "description": "Compact bullet summary suitable for prompt injection."
            },
            "issue_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short, normalized issue labels (snake_case or short phrases)."
            },
            "statute_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key statutes/regulations, normalized (e.g., '18 USC 922(g)')."
            },
            "doctrine_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key doctrines/standards (e.g., 'qualified immunity', 'plain error')."
            },
        },
        "required": ["case_card_text", "issue_tags", "statute_tags", "doctrine_tags"],
    },
    "strict": True,
}


SECTION_PRIORITY = {
    "facts": 1,
    "standard_of_review": 2,
    "statute": 3,
    "argument": 4,
    # everything else later
}


def pick_cases_to_process(conn) -> List[str]:
    """
    Get case_ids that need case_cards.
    - If REGEN: all cases
    - Else: cases without a row in case_cards
    """
    with conn.cursor() as cur:
        if REGEN:
            cur.execute("SELECT case_id FROM cases ORDER BY case_id LIMIT %s", (MAX_CASES,))
        else:
            cur.execute(
                """
                SELECT c.case_id
                FROM cases c
                LEFT JOIN case_cards cc ON cc.case_id = c.case_id
                WHERE cc.case_id IS NULL
                ORDER BY c.case_id
                LIMIT %s
                """,
                (MAX_CASES,),
            )
        rows = cur.fetchall()
    return [r[0] for r in rows]


def load_case_metadata(conn, case_id: str) -> Dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM cases WHERE case_id=%s", (case_id,))
        row = cur.fetchone()
    if not row:
        raise ValueError(f"case_id not found in cases: {case_id}")
    return dict(row)


def load_case_evidence_packet(conn, case_id: str) -> str:
    """
    Build a compact 'evidence packet' from chunks for this case, prioritizing sections.
    We do NOT need all chunks. We want enough signal to produce a good case_card.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
                ch.section_type,
                ch.heading_path,
                ch.chunk_text,
                ch.token_count,
                d.doc_type
            FROM chunks ch
            JOIN documents d ON d.doc_id = ch.doc_id
            WHERE ch.case_id = %s
            """,
            (case_id,),
        )
        chunks = cur.fetchall()

    # Sort by section priority, then by token_count desc to prefer meatier sections
    def sort_key(r):
        return (SECTION_PRIORITY.get((r["section_type"] or "").lower(), 999),
                -(r["token_count"] or 0))

    chunks_sorted = sorted(chunks, key=sort_key)

    out_parts: List[str] = []
    total_chars = 0

    for r in chunks_sorted:
        section = (r["section_type"] or "unknown").strip()
        heading = (r["heading_path"] or "").strip()
        doc_type = (r["doc_type"] or "unknown").strip()
        text = (r["chunk_text"] or "").strip()

        if not text:
            continue

        block = f"[doc_type={doc_type}] [section_type={section}]"
        if heading:
            block += f" [heading={heading}]"
        block += "\n" + text + "\n"

        if total_chars + len(block) > MAX_EVIDENCE_CHARS:
            break

        out_parts.append(block)
        total_chars += len(block)

    if not out_parts:
        return ""

    return "\n---\n".join(out_parts)


def build_prompt(case_meta: Dict[str, Any], evidence_packet: str) -> str:
    # Keep this fairly structured so outputs are consistent across cases.
    return f"""
You are generating a compact "case card" for retrieval-augmented prediction of Eighth Circuit outcomes/complexity.

Write a bulletized summary that is:
- short (aim ~150-300 words)
- factual and neutral (no speculation)
- organized and scannable for later prompt injection

Include (when available):
- parties (if inferable), court/circuit context
- procedural posture
- key facts (only the essential ones)
- issues presented
- rules/standards (e.g., standard of review, doctrines)
- holding/disposition IF it is present in the evidence (otherwise omit)
- anything that makes this case distinctive for similarity matching

Then produce:
- issue_tags: 3-8 concise normalized labels
- statute_tags: 0-8 key statutes/regulations (exact-ish citations when possible)
- doctrine_tags: 0-8 doctrines/standards/tests

CASE METADATA (may be partial):
{json.dumps(case_meta, default=str, indent=2)}

EVIDENCE PACKET (selected chunks):
{evidence_packet}
""".strip()


def call_openai_case_card(prompt: str) -> Dict[str, Any]:
    """
    Uses Chat Completions API + JSON Schema structured outputs.
    """
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": CASE_CARD_SCHEMA,
        },
    )

    output_text = resp.choices[0].message.content
    if not output_text:
        raise RuntimeError("OpenAI returned empty content.")

    try:
        data = json.loads(output_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON. Raw:\n{output_text[:2000]}") from e

    return data


def upsert_case_card(conn, case_id: str, card: Dict[str, Any], model_name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO case_cards (
                case_id,
                case_card_text,
                issue_tags,
                statute_tags,
                doctrine_tags,
                generated_by_model,
                generated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (case_id) DO UPDATE SET
                case_card_text = EXCLUDED.case_card_text,
                issue_tags = EXCLUDED.issue_tags,
                statute_tags = EXCLUDED.statute_tags,
                doctrine_tags = EXCLUDED.doctrine_tags,
                generated_by_model = EXCLUDED.generated_by_model,
                generated_at = NOW()
            """,
            (
                case_id,
                card["case_card_text"],
                card["issue_tags"],
                card["statute_tags"],
                card["doctrine_tags"],
                model_name,
            ),
        )


def main():
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("SET statement_timeout = '5min'")
        case_ids = pick_cases_to_process(conn)

        print(f"Found {len(case_ids)} case(s) to process (REGEN={REGEN}).")

        for idx, case_id in enumerate(case_ids, 1):
            print(f"\n[{idx}/{len(case_ids)}] case_id={case_id}")

            case_meta = load_case_metadata(conn, case_id)
            evidence = load_case_evidence_packet(conn, case_id)

            if not evidence:
                print("  - No chunks found; skipping.")
                continue

            prompt = build_prompt(case_meta, evidence)

            # Basic retry loop for transient failures
            for attempt in range(3):
                try:
                    card = call_openai_case_card(prompt)
                    upsert_case_card(conn, case_id, card, OPENAI_MODEL)
                    conn.commit()
                    print("  - case_card upserted.")
                    break
                except Exception as e:
                    conn.rollback()
                    print(f"  - ERROR attempt {attempt+1}/3: {e}")
                    if attempt == 2:
                        raise
                    time.sleep(2.0 * (attempt + 1))


if __name__ == "__main__":
    main()
