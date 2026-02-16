from config import API_KEY
from config import DATABASE_URL

import json
import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional

import psycopg2
from psycopg2.extras import execute_values
from pypdf import PdfReader

from openai import OpenAI  # pip install openai
# extract all basic data from the case for the cases table (decision, oral argument, etc)
case_table_prompt = """
You are an information extraction engine for an appellate-case ingestion pipeline.

Your job: read the provided case decision brief, plus a separate config.txt snippet, and return a SINGLE JSON object that matches the schema below exactly. Do not include any additional keys. Do not include markdown. Do not include commentary.

TARGET OUTPUT (strict JSON; no trailing commas):
{
  "case_id": string,
  "docket_number": string | null,
  "case_type": string | null,
  "procedural_posture": string | null,
  "decision_date": string | null,                 // format: YYYY-MM-DD
  "disposition_label": string | null,             // e.g., "affirmed", "reversed", "vacated", "remanded", "dismissed", "mixed", "unknown"
  "complexity_label": string | null,              // e.g., "low", "medium", "high" (or "unknown" if unclear)
  "oral_argument_minutes": number | null,         // integer minutes
  "argued_flag": boolean | null,                  // true if argued, false if submitted on briefs / not argued
  "ingest_version": string
}

HARD RULES
1) Output MUST be valid JSON (no code fences, no markdown).
2) If a field is not clearly supported by the text, use null (or "unknown" only for the label fields as allowed above).
3) Do NOT guess docket numbers, dates, or outcomes. Only extract if explicitly present or reliably inferable from an unambiguous statement.
4) decision_date must be an ISO date YYYY-MM-DD if present; otherwise null.
5) oral_argument_minutes: extract from the config (or decision text) whenever a duration is mentioned (e.g. "15 minutes", "20 min"); else null.
6) argued_flag: true if oral argument occurred; false if explicitly indicates "submitted" / "no oral argument"; else null.
7) Consistency: whenever you set argued_flag to true and the config or decision text mentions an oral argument duration (in minutes or other units), you MUST also set oral_argument_minutes to that value (convert to integer minutes). Do not leave oral_argument_minutes null when a duration is present and argued_flag is true.
8) ingest_version must be copied exactly from the provided INGEST_VERSION value.

FIELD EXTRACTION GUIDANCE
- case_id:
  Use the docket number only. Same value as docket_number (e.g., "23-2869"). Do not include court name or any other prefix.
- docket_number:
  Look for patterns like "No. 23-____", "Case No.", "Docket No.", or caption headers.
- case_type:
  Choose a short category when possible (e.g., "criminal", "civil", "immigration", "bankruptcy", "administrative", "habeas", "prisoner civil rights", "employment", "contract", "torts"). If multiple, choose the primary one suggested by the brief/caption and issues.
- procedural_posture:
  A concise phrase describing what is on appeal (e.g., "appeal from denial of motion to suppress", "appeal from summary judgment", "sentencing appeal", "petition for review of agency decision"). Keep it short.
- decision_date:
  Only if explicitly stated in the documents (cover page, opinion excerpt, docket header, etc.). Otherwise null.
- disposition_label:
  Only if explicitly stated (e.g., "AFFIRMED", "REVERSED", "VACATED AND REMANDED"). If there’s no decision/outcome text in what you received, set null (or "unknown" if your pipeline requires a non-null label—but prefer null).
- complexity_label:
  Use "low/medium/high" based on indicators in the briefs:
    * low: narrow issue, few legal questions, short record, straightforward standard of review
    * medium: multiple issues or moderate record/complexity
    * high: many issues, complex statutory/regulatory scheme, extensive record, multiple parties, complicated procedural history
  If you cannot assess, return "unknown".

CONFIG FILE PARSING
You will be given the raw contents of a config file (if provided). Config files vary in format. Examples:
- "docket #: 23-2869" or "docket_number=23-2869"
- "oral argument: 15 minutes" or "oral_argument_minutes=15" or "oral_argument_duration=15"
Parse the config to extract: docket_number (string), oral_argument_minutes (integer; convert to minutes if given in other units), and argued_flag (true if oral argument occurred or a duration is present, false if "submitted" or "no oral argument"). Use null only for fields not found.
Important: If the config (or decision text) states an oral argument duration in any form (e.g. "15 minutes", "20 min"), you MUST set both argued_flag=true and oral_argument_minutes to that integer. Never set argued_flag=true while leaving oral_argument_minutes null when a duration is stated.

Now produce the JSON object only.
"""

@dataclass
class IngestConfig:
    docket_number: Optional[str] = None
    oral_argument_minutes: Optional[int] = None
    argued_flag: Optional[bool] = None


def read_config_file(path: Optional[str]) -> str:
    """Read raw config file content for OpenAI to parse. Returns empty string if no file."""
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# Config format: "docket #: 23-2687" and "oral argument: 20 minutes"
CONFIG_DOCKET_RE = re.compile(r"docket\s*#?\s*:\s*([\d\-]+)", re.IGNORECASE)
CONFIG_ORAL_MINUTES_RE = re.compile(r"oral\s+argument\s*:\s*(\d+)\s*(?:minutes?|min)?", re.IGNORECASE)


def parse_config_fallback(raw_config_text: str) -> Dict[str, Any]:
    """
    Parse the standard config format. Returns a dict with keys that were found
    (docket_number, oral_argument_minutes, argued_flag) so we can fill in model
    output when the model misses them.
    """
    out: Dict[str, Any] = {}
    if not raw_config_text:
        return out
    m = CONFIG_DOCKET_RE.search(raw_config_text)
    if m:
        out["docket_number"] = m.group(1).strip()
    m = CONFIG_ORAL_MINUTES_RE.search(raw_config_text)
    if m:
        out["oral_argument_minutes"] = int(m.group(1))
        out["argued_flag"] = True
    return out


def parse_config_txt(path: Optional[str]) -> IngestConfig:
    if not path:
        return IngestConfig()
    cfg = IngestConfig()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().lower()
            v = v.strip()

            if k == "docket_number":
                cfg.docket_number = v or None
            elif k == "oral_argument_minutes":
                try:
                    cfg.oral_argument_minutes = int(v)
                except ValueError:
                    cfg.oral_argument_minutes = None
            elif k == "argued_flag":
                if v.lower() in ("true", "1", "yes", "y"):
                    cfg.argued_flag = True
                elif v.lower() in ("false", "0", "no", "n"):
                    cfg.argued_flag = False
                else:
                    cfg.argued_flag = None
    return cfg
  
def extract_text_from_pdf(pdf_path: str, max_chars: int = 1_500_000) -> str:
    reader = PdfReader(pdf_path)
    parts: list[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
        if sum(len(p) for p in parts) >= max_chars:
            break
    text = "\n\n".join(parts)
    # light cleanup
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
  
# ---------------------------------------------------------------------
# OpenAI call (Structured Outputs via JSON Schema)
# ---------------------------------------------------------------------


def case_json_schema() -> Dict[str, Any]:
    """
    JSON Schema that matches the response you described.
    Structured Outputs will force valid JSON that conforms to this schema. :contentReference[oaicite:6]{index=6}
    """
    return {
        "name": "case_table_row",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "case_id": {"type": "string"},
                "docket_number": {"type": ["string", "null"]},
                "case_type": {"type": ["string", "null"]},
                "procedural_posture": {"type": ["string", "null"]},
                "decision_date": {
                    "type": ["string", "null"],
                    "description": "YYYY-MM-DD",
                    "pattern": r"^\d{4}-\d{2}-\d{2}$",
                },
                "disposition_label": {"type": ["string", "null"]},
                "complexity_label": {"type": ["string", "null"]},
                "oral_argument_minutes": {"type": ["integer", "null"]},
                "argued_flag": {"type": ["boolean", "null"]},
                "ingest_version": {"type": "string"},
            },
            "required": [
                "case_id",
                "docket_number",
                "case_type",
                "procedural_posture",
                "decision_date",
                "disposition_label",
                "complexity_label",
                "oral_argument_minutes",
                "argued_flag",
                "ingest_version",
            ],
        },
    }


def call_openai_for_case_json(
    *,
    decision_text: str,
    ingest_version: str,
    raw_config_text: str = "",
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    client = OpenAI(api_key=os.environ["FINN_API_KEY"])

    # Let OpenAI parse config (formats vary: "docket #: X", "oral argument: N minutes", etc.)
    if raw_config_text:
        config_section = f"CONFIG_FILE (raw content):\n{raw_config_text}\n\n"
    else:
        config_section = "No config file provided.\n\n"

    input_text = (
        f"{case_table_prompt}\n\n"
        f"{config_section}"
        "DECISION_DOCUMENT_TEXT:\n"
        f"{decision_text}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": input_text}],
        response_format={
            "type": "json_schema",
            "json_schema": case_json_schema(),
        },
    )

    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("OpenAI returned empty content; cannot parse JSON.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model output was not valid JSON: {e}\nRAW:\n{raw}") from e

    # Ensure ingest_version is exactly what we passed (defensive)
    data["ingest_version"] = ingest_version

    # Fallback: parse standard config format so oral argument is always recorded
    if raw_config_text:
        fallback = parse_config_fallback(raw_config_text)
        if fallback.get("docket_number") is not None and data.get("docket_number") is None:
            data["docket_number"] = fallback["docket_number"]
        if fallback.get("oral_argument_minutes") is not None:
            data["oral_argument_minutes"] = fallback["oral_argument_minutes"]
        if fallback.get("argued_flag") is not None:
            data["argued_flag"] = fallback["argued_flag"]

    return data


# ---------------------------------------------------------------------
# DB upsert into cases
# ---------------------------------------------------------------------


UPSERT_CASE_SQL = """
INSERT INTO cases (
  case_id,
  docket_number,
  case_type,
  procedural_posture,
  decision_date,
  disposition_label,
  complexity_label,
  oral_argument_minutes,
  argued_flag,
  ingest_version
) VALUES (
  %(case_id)s,
  %(docket_number)s,
  %(case_type)s,
  %(procedural_posture)s,
  %(decision_date)s,
  %(disposition_label)s,
  %(complexity_label)s,
  %(oral_argument_minutes)s,
  %(argued_flag)s,
  %(ingest_version)s
)
ON CONFLICT (case_id) DO UPDATE SET
  docket_number = EXCLUDED.docket_number,
  case_type = EXCLUDED.case_type,
  procedural_posture = EXCLUDED.procedural_posture,
  decision_date = EXCLUDED.decision_date,
  disposition_label = EXCLUDED.disposition_label,
  complexity_label = EXCLUDED.complexity_label,
  oral_argument_minutes = EXCLUDED.oral_argument_minutes,
  argued_flag = EXCLUDED.argued_flag,
  ingest_version = EXCLUDED.ingest_version
;
"""


def upsert_case_row(db_url: str, row: Dict[str, Any]) -> None:
    # Convert decision_date (YYYY-MM-DD) string -> date or None
    dd = row.get("decision_date")
    if isinstance(dd, str) and dd:
        try:
            y, m, d = dd.split("-")
            row["decision_date"] = date(int(y), int(m), int(d))
        except Exception:
            # If the model gave an invalid date despite schema (shouldn't happen),
            # store NULL rather than crashing ingestion.
            row["decision_date"] = None
    else:
        row["decision_date"] = None

    conn = psycopg2.connect(db_url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(UPSERT_CASE_SQL, row)
    finally:
        conn.close()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


# Default train dataset path relative to this script (rag_context/)
DEFAULT_TRAIN_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "AIth Circuit Test Dataset",
    "train",
)


def _iter_train_cases(train_dir: str):
    """Yield (case_dir, pdf_path, config_path) for each case in train_dir."""
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train dataset directory not found: {train_dir}")
    for name in sorted(os.listdir(train_dir)):
        case_dir = os.path.join(train_dir, name)
        if not os.path.isdir(case_dir) or name.startswith("."):
            continue
        # PDF: "{name} decision.pdf"
        pdf_name = f"{name} decision.pdf"
        pdf_path = os.path.join(case_dir, pdf_name)
        if not os.path.isfile(pdf_path):
            continue
        config_path = os.path.join(case_dir, "config.txt")
        if not os.path.isfile(config_path):
            config_path = None
        yield case_dir, pdf_path, config_path


def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL env var is required (e.g., postgresql://...).")

    train_dir = os.path.abspath(DEFAULT_TRAIN_DATASET)
    ingest_version = os.environ.get("INGEST_VERSION", "v1")
    model = os.environ.get("INGEST_MODEL", "gpt-4o")

    total = 0
    errors = []
    for case_dir, pdf_path, config_path in _iter_train_cases(train_dir):
        case_id = os.path.basename(case_dir)
        try:
            raw_config = read_config_file(config_path)
            decision_text = extract_text_from_pdf(pdf_path)
            case_row = call_openai_for_case_json(
                decision_text=decision_text,
                ingest_version=ingest_version,
                raw_config_text=raw_config,
                model=model,
            )
            # Use folder name as authoritative case_id so DB always matches train directory
            case_row["case_id"] = case_id
            upsert_case_row(db_url, case_row)
            total += 1
            print(f"✅ [{total}] {case_id}")
        except Exception as e:
            errors.append((case_id, str(e)))
            print(f"❌ {case_id}: {e}")
    print(f"\nDone. Ingested {total} cases.")
    if errors:
        print(f"Failed ({len(errors)}): {[c for c, _ in errors]}")


if __name__ == "__main__":
    main()