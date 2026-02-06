
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# extract all basic data from the case for the cases table (decision, oral argument, etc)
case_table_prompt = """
You are an information extraction engine for an appellate-case ingestion pipeline.

Your job: read the provided case-brief texts (from PDFs), the decisison brief, plus a separate config.txt snippet, and return a SINGLE JSON object that matches the schema below exactly. Do not include any additional keys. Do not include markdown. Do not include commentary.

TARGET OUTPUT (strict JSON; no trailing commas):
{
  "case_id": string,
  "docket_number": string | null,
  "judges": [string],
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
5) oral_argument_minutes: extract from the oral-argument .txt if present; else null.
6) argued_flag: true if oral argument occurred; false if explicitly indicates "submitted" / "no oral argument"; else null.
7) ingest_version must be copied exactly from the provided INGEST_VERSION value.

FIELD EXTRACTION GUIDANCE
- case_id:
  Dervice a case id from the docket number and the court name. The court name will be "8th Cir."
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

Alongside the briefs, you will be provided a config.txt file that contains the following information:
- docket_number
- oral_argument_duration [if oral_argument_duration is present, set argued_flag to true]

Now produce the JSON object only.
"""