"""
New Case Retrieval Pipeline (v2)
Supports arbitrary number of uploaded briefs.
Performs retrieval only (no database writes).

v2 changes:
- Case-type gating (civil vs criminal hard filter at final selection)
- Tag/posture-aware reranking (doctrine > statute > posture weighting)
- Hub-dominance mitigation (chunk cap + exponential decay)
- Improved query fingerprint (boilerplate stripping, signal front-loading)
- Structured JSON-lines retrieval logging for auditability
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple, Any, Set
import os

import psycopg
from pypdf import PdfReader
from psycopg.rows import dict_row

from rag_context.config import DATABASE_URL, API_KEY

logger = logging.getLogger("rag_retrieval")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DEFAULT_K = 3
DEFAULT_CANDIDATE_N = 50
DEFAULT_CANDIDATE_CHUNKS = 80          # legacy; overridden when RetrievalConfig used
DEFAULT_MAX_CHUNKS_PER_CASE = 4        # legacy
DEFAULT_CHUNK_CAP = 3
DEFAULT_DECAY = 0.85
DEFAULT_WEIGHTS = {"embed": 0.65, "doctrine": 0.20, "statute": 0.10, "posture": 0.05}


@dataclass
class RetrievalConfig:
    """All tunable retrieval parameters in one place."""
    k: int = DEFAULT_K
    candidate_n: int = DEFAULT_CANDIDATE_N
    chunk_cap: int = DEFAULT_CHUNK_CAP
    decay: float = DEFAULT_DECAY
    allow_cross_type: bool = False
    w_embed: float = DEFAULT_WEIGHTS["embed"]
    w_doctrine: float = DEFAULT_WEIGHTS["doctrine"]
    w_statute: float = DEFAULT_WEIGHTS["statute"]
    w_posture: float = DEFAULT_WEIGHTS["posture"]
    chunks_per_candidate: int = 5      # heuristic multiplier for fetch count


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class BriefInput:
    """Represents one uploaded brief."""
    label: str
    text: str


@dataclass
class QuerySignals:
    """Extracted signals about the query case for gating and reranking."""
    case_type: Optional[str] = None
    procedural_posture: Optional[str] = None
    posture_bucket: Optional[str] = None
    statute_tags: List[str] = field(default_factory=list)
    doctrine_tags: List[str] = field(default_factory=list)
    issue_tags: List[str] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    """Per-component scores for auditability."""
    embed: float
    doctrine: float
    statute: float
    posture: float
    raw_embed_agg: float


@dataclass
class RetrievedCaseCard:
    case_id: str
    score: float
    case_card_text: str
    case_type: Optional[str] = None
    issue_tags: Optional[List[str]] = None
    statute_tags: Optional[List[str]] = None
    doctrine_tags: Optional[List[str]] = None
    procedural_posture: Optional[str] = None
    score_breakdown: Optional[ScoreBreakdown] = None


def _preview(text: str, max_chars: int = 180) -> str:
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."


# ─────────────────────────────────────────────
# Text Processing
# ─────────────────────────────────────────────

def _clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_boilerplate(text: str) -> str:
    """Remove appellate brief boilerplate that dilutes embedding signal."""
    text = re.sub(
        r"(?i)\n\s*CERTIFICATE\s+OF\s+(?:COMPLIANCE|SERVICE|FILING)\b[\s\S]{0,2000}$",
        "\n", text,
    )
    text = re.sub(
        r"(?i)\n\s*Respectfully\s+submitted[\s\S]{0,1000}$",
        "\n", text,
    )
    text = re.sub(
        r"(?i)(?:TABLE\s+OF\s+(?:CONTENTS|AUTHORITIES))\s*\n"
        r"(?:\s*(?:[ivxlc]+|\d+|[A-Z])\.\s+.*\n){0,80}",
        "\n", text,
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ─────────────────────────────────────────────
# Posture Bucketization
# ─────────────────────────────────────────────

_POSTURE_BUCKETS: Dict[str, List[str]] = {
    "12b6":                   ["12(b)(6)", "motion to dismiss", "failure to state a claim", "12(b)(1)"],
    "summary_judgment":       ["summary judgment", "rule 56", "cross-motions"],
    "trial":                  ["jury trial", "bench trial", "trial verdict", "post-trial",
                               "judgment as a matter of law", "jmol", "directed verdict"],
    "sentencing":             ["sentencing", "sentence", "guidelines", "resentencing"],
    "plea":                   ["guilty plea", "plea agreement", "plea bargain"],
    "preliminary_injunction": ["preliminary injunction", "temporary restraining order", "tro"],
    "class_certification":    ["class certification", "class action"],
    "habeas":                 ["habeas corpus", "2254", "2255", "habeas"],
    "default_judgment":       ["default judgment"],
    "interlocutory":          ["interlocutory appeal", "mandamus", "writ"],
    "post_conviction":        ["post-conviction", "rule 60", "rule 59", "reconsideration"],
    "administrative":         ["administrative review", "agency decision",
                               "immigration judge", "bia", "alj"],
    "qualified_immunity":     ["qualified immunity", "immunity appeal"],
    "suppression":            ["suppress", "motion to suppress", "fourth amendment"],
}


def _bucketize_posture(posture: Optional[str]) -> Optional[str]:
    """Map free-text procedural posture to a canonical bucket."""
    if not posture:
        return None
    low = posture.lower()
    best, best_count = None, 0
    for bucket, keywords in _POSTURE_BUCKETS.items():
        count = sum(1 for kw in keywords if kw in low)
        if count > best_count:
            best_count = count
            best = bucket
    return best


# ─────────────────────────────────────────────
# Case Type Normalization
# ─────────────────────────────────────────────

_CIVIL_SIGNALS = [
    "civil", "§ 1983", "section 1983", "42 u.s.c", "prisoner civil rights",
    "deliberate indifference", "excessive force", "equal protection",
    "title vii", "title ix", "ada", "fmla", "employment discrimination",
    "tort", "contract dispute", "negligence", "bivens", "ftca",
    "federal tort claims", "class action", "prison conditions",
]
_CRIMINAL_SIGNALS = [
    "criminal", "sentencing", "guilty plea", "plea agreement",
    "restitution", "guidelines range", "indictment", "conviction",
    "probation", "supervised release", "18 u.s.c", "21 u.s.c",
    "rico", "conspiracy to distribute", "drug trafficking",
    "felon in possession", "922(g)", "armed career criminal",
    "mandatory minimum", "counts of conviction",
]
_IMMIGRATION_SIGNALS = [
    "immigration", "removal", "deportation", "asylum", "bia",
    "immigration judge", "ina", "8 u.s.c",
    "withholding of removal", "convention against torture",
]


def _normalize_case_type(raw: Optional[str]) -> Optional[str]:
    """Canonicalize case_type strings to a small set of categories."""
    if not raw:
        return None
    r = raw.lower().strip()
    if any(s in r for s in ("criminal", "sentencing", "restitution", "plea", "conviction")):
        return "criminal"
    if any(s in r for s in ("civil", "1983", "tort", "contract", "employment", "prisoner")):
        return "civil"
    if any(s in r for s in ("immigration", "removal", "deportation", "asylum")):
        return "immigration"
    if "bankruptcy" in r:
        return "bankruptcy"
    if any(s in r for s in ("administrative", "agency")):
        return "administrative"
    return r


def _infer_case_type_from_text(text: str) -> Optional[str]:
    """Infer civil/criminal/immigration from brief text by keyword frequency."""
    t = text if isinstance(text, str) else ""
    t = t.lower()
    scores = {
        "civil":       sum(t.count(s) for s in _CIVIL_SIGNALS),
        "criminal":    sum(t.count(s) for s in _CRIMINAL_SIGNALS),
        "immigration": sum(t.count(s) for s in _IMMIGRATION_SIGNALS),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 3 else None


# ─────────────────────────────────────────────
# Signal Extraction
# ─────────────────────────────────────────────

_STATUTE_RE = re.compile(
    r"(?:"
    r"\d{1,2}\s+U\.?S\.?C\.?\s*§+\s*\d+(?:\([a-zA-Z0-9]+\))*"
    r"|(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth"
    r"|Eleventh|Twelfth|Thirteenth|Fourteenth)\s+Amendment"
    r"|Fed\.?\s*R\.?\s*(?:Civ|Crim|App|Evid)\.?\s*P\.?\s*\d+"
    r"|Rule\s+\d+(?:\([a-z]\)(?:\(\d+\))?)?"
    r"|PLRA"
    r")",
    re.IGNORECASE,
)

_DOCTRINE_KEYWORDS = [
    "qualified immunity", "deliberate indifference", "excessive force",
    "due process", "equal protection", "plra exhaustion",
    "plain error", "harmless error", "abuse of discretion",
    "strickland", "batson", "brady", "daubert", "chevron",
    "habeas corpus", "ineffective assistance", "cruel and unusual",
    "probable cause", "reasonable suspicion", "terry stop",
    "sovereign immunity", "mootness", "standing", "ripeness",
    "res judicata", "collateral estoppel", "claim preclusion",
    "issue preclusion", "monell", "iqbal", "twombly",
    "deliberate indifference to serious medical needs",
    "totality of the circumstances", "objective reasonableness",
]

_ISSUE_KEYWORDS = [
    "prison conditions", "medical care", "use of force",
    "first amendment", "free speech", "religious exercise",
    "search and seizure", "miranda", "confrontation clause",
    "speedy trial", "double jeopardy", "self-incrimination",
    "class certification", "arbitration", "preemption",
    "statutory interpretation", "commerce clause",
    "employment discrimination", "retaliation", "hostile work environment",
    "title vii", "ada", "fmla", "section 1981",
    "wrongful death", "personal injury", "malpractice",
    "immigration relief", "asylum", "cancellation of removal",
]


def extract_query_signals(
    brief_texts: List[str],
    llm_metadata: Optional[Dict[str, Any]] = None,
) -> QuerySignals:
    """
    Extract structured signals from query briefs for gating/reranking.
    Prefers LLM-extracted metadata when available; falls back to regex.
    """
    combined = " ".join(brief_texts)[:200_000]
    combined_lower = combined.lower()

    case_type = None
    if llm_metadata and llm_metadata.get("case_type"):
        case_type = _normalize_case_type(llm_metadata["case_type"])
    if not case_type:
        case_type = _infer_case_type_from_text(combined_lower)

    posture = (llm_metadata or {}).get("procedural_posture")

    statute_tags = sorted({m.strip() for m in _STATUTE_RE.findall(combined)} - {""})
    doctrine_tags = sorted({kw for kw in _DOCTRINE_KEYWORDS if kw in combined_lower})
    issue_tags = sorted({kw for kw in _ISSUE_KEYWORDS if kw in combined_lower})

    return QuerySignals(
        case_type=case_type,
        procedural_posture=posture,
        posture_bucket=_bucketize_posture(posture),
        statute_tags=statute_tags,
        doctrine_tags=doctrine_tags,
        issue_tags=issue_tags,
    )


# ─────────────────────────────────────────────
# Fingerprint Builder (v2)
# ─────────────────────────────────────────────

def build_query_fingerprint(
    briefs: List[BriefInput],
    query_signals: Optional[QuerySignals] = None,
    max_chars_per_brief: int = 12000,
    max_total_chars: int = 24000,
) -> str:
    """
    Build a structured retrieval fingerprint from N uploaded briefs.
    Front-loads legal signals and strips boilerplate for better embedding quality.
    """
    parts: List[str] = ["LEGAL CASE RETRIEVAL FINGERPRINT", ""]

    if query_signals:
        signal_lines: List[str] = []
        if query_signals.case_type:
            signal_lines.append(f"CASE TYPE: {query_signals.case_type}")
        if query_signals.procedural_posture:
            signal_lines.append(f"PROCEDURAL POSTURE: {query_signals.procedural_posture}")
        if query_signals.statute_tags:
            signal_lines.append(f"KEY STATUTES: {'; '.join(query_signals.statute_tags[:15])}")
        if query_signals.doctrine_tags:
            signal_lines.append(f"KEY DOCTRINES: {'; '.join(query_signals.doctrine_tags[:10])}")
        if query_signals.issue_tags:
            signal_lines.append(f"LEGAL ISSUES: {'; '.join(query_signals.issue_tags[:10])}")
        if signal_lines:
            parts.extend(signal_lines)
            parts.append("")

    remaining = max_total_chars - sum(len(p) + 1 for p in parts)

    for brief in briefs:
        if remaining <= 0:
            break
        text = _strip_boilerplate(_clean_text(brief.text))
        text = text[:max_chars_per_brief]
        text = text[:remaining]
        remaining -= len(text)
        label = (brief.label or "brief").strip()
        parts.append(f"[{label.upper()}]")
        parts.append(text)
        parts.append("")

    return "\n".join(parts).strip()


# ─────────────────────────────────────────────
# Embedding (OpenAI)
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Scoring Utilities
# ─────────────────────────────────────────────

def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two tag sets."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _aggregate_with_decay(
    sims: List[float], chunk_cap: int, decay: float,
) -> float:
    """
    Aggregate chunk similarities with diminishing returns.
    Only the top chunk_cap chunks count; successive chunks are decayed.
    score = sum(sim_i * decay^rank_i)
    """
    ranked = sorted(sims, reverse=True)[:chunk_cap]
    return sum(s * (decay ** i) for i, s in enumerate(ranked))


# ─────────────────────────────────────────────
# Retrieval Logic (v2)
# ─────────────────────────────────────────────

def retrieve_top_k_case_cards(
    conn: psycopg.Connection,
    query_embedding: List[float],
    k: int,
    candidate_chunks: int = DEFAULT_CANDIDATE_CHUNKS,
    max_chunks_per_case: int = DEFAULT_MAX_CHUNKS_PER_CASE,
    *,
    config: Optional[RetrievalConfig] = None,
    query_signals: Optional[QuerySignals] = None,
) -> List[RetrievedCaseCard]:
    """
    Retrieve top-k similar case cards.

    When config/query_signals are provided, uses the full v2 pipeline:
    decay aggregation, tag-aware reranking, and case-type gating.
    Degrades gracefully when signals are missing (pure embed ranking).
    """
    cfg = config or RetrievalConfig()
    effective_k = cfg.k if config else k
    chunk_cap = cfg.chunk_cap if config else max_chunks_per_case
    fetch_chunks = (
        max(candidate_chunks, cfg.candidate_n * cfg.chunks_per_candidate)
        if config else candidate_chunks
    )

    if effective_k <= 0:
        logger.warning(
            "[retrieve_top_k_case_cards] effective_k<=0; returning no results. k=%s cfg.k=%s",
            k,
            cfg.k,
        )
        return []

    logger.info(
        "[retrieve_top_k_case_cards] start k=%d effective_k=%d candidate_chunks=%d fetch_chunks=%d "
        "candidate_n=%d chunk_cap=%d decay=%.3f allow_cross_type=%s",
        k,
        effective_k,
        candidate_chunks,
        fetch_chunks,
        cfg.candidate_n,
        chunk_cap,
        cfg.decay,
        cfg.allow_cross_type,
    )
    if query_signals:
        logger.info(
            "[retrieve_top_k_case_cards] query_signals case_type=%s posture=%s posture_bucket=%s "
            "doctrines=%d statutes=%d issues=%d",
            query_signals.case_type,
            query_signals.procedural_posture,
            query_signals.posture_bucket,
            len(query_signals.doctrine_tags),
            len(query_signals.statute_tags),
            len(query_signals.issue_tags),
        )

    emb_literal = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

    # ── Step 1: Retrieve nearest chunks via pgvector ──
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT case_id, chunk_id,
                   (1.0 - (embedding <=> %s::vector)) AS sim
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (emb_literal, emb_literal, fetch_chunks),
        )
        chunk_rows = cur.fetchall()
    logger.info(
        "[retrieve_top_k_case_cards] step1 nearest chunk rows=%d",
        len(chunk_rows),
    )

    # ── Step 2: Aggregate chunk scores per case with decay ──
    per_case: Dict[str, List[float]] = {}
    for row in chunk_rows:
        per_case.setdefault(row["case_id"], []).append(float(row["sim"]))

    raw_embed_agg: Dict[str, float] = {
        cid: _aggregate_with_decay(sims, chunk_cap, cfg.decay)
        for cid, sims in per_case.items()
    }

    max_embed = max(raw_embed_agg.values()) if raw_embed_agg else 1.0
    norm_embed: Dict[str, float] = {
        cid: (v / max_embed if max_embed > 0 else 0.0)
        for cid, v in raw_embed_agg.items()
    }

    candidate_ids = sorted(
        norm_embed, key=norm_embed.get, reverse=True,
    )[:cfg.candidate_n]
    logger.info(
        "[retrieve_top_k_case_cards] step2 unique_cases=%d candidate_ids=%d top_candidate_sample=%s",
        len(per_case),
        len(candidate_ids),
        candidate_ids[:5],
    )

    if not candidate_ids:
        logger.warning(
            "[retrieve_top_k_case_cards] no candidate_ids after embedding aggregation. "
            "Possible causes: empty chunks table, embedding mismatch, or DB query returned no rows."
        )
        return []

    # ── Step 3: Fetch metadata + tags for all candidates ──
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT c.case_id, c.case_type, c.procedural_posture,
                   cc.case_card_text, cc.issue_tags,
                   cc.statute_tags, cc.doctrine_tags
            FROM cases c
            LEFT JOIN case_cards cc ON cc.case_id = c.case_id
            WHERE c.case_id = ANY(%s);
            """,
            (candidate_ids,),
        )
        meta_rows = cur.fetchall()
    logger.info(
        "[retrieve_top_k_case_cards] step3 fetched metadata rows=%d for candidate_ids=%d",
        len(meta_rows),
        len(candidate_ids),
    )

    meta_map: Dict[str, Dict] = {r["case_id"]: r for r in meta_rows}

    # ── Step 4: Compute rerank scores ──
    q_doctrines: Set[str] = {
        t.lower() for t in (query_signals.doctrine_tags if query_signals else [])
    }
    q_statutes: Set[str] = {
        t.lower() for t in (query_signals.statute_tags if query_signals else [])
    }
    q_posture_bucket = query_signals.posture_bucket if query_signals else None
    q_case_type = (
        _normalize_case_type(query_signals.case_type) if query_signals else None
    )

    final_scores: Dict[str, float] = {}
    breakdowns: Dict[str, ScoreBreakdown] = {}
    candidate_types: Dict[str, Optional[str]] = {}

    for cid in candidate_ids:
        embed_score = norm_embed.get(cid, 0.0)
        meta = meta_map.get(cid, {})

        c_doctrines = {
            (t or "").lower() for t in (meta.get("doctrine_tags") or [])
        } - {""}
        c_statutes = {
            (t or "").lower() for t in (meta.get("statute_tags") or [])
        } - {""}
        c_posture_bucket = _bucketize_posture(meta.get("procedural_posture"))
        c_type = _normalize_case_type(meta.get("case_type"))
        candidate_types[cid] = c_type

        doctrine_score = _jaccard(q_doctrines, c_doctrines)
        statute_score = _jaccard(q_statutes, c_statutes)
        posture_score = (
            1.0
            if (q_posture_bucket and c_posture_bucket
                and q_posture_bucket == c_posture_bucket)
            else 0.0
        )

        combined = (
            cfg.w_embed * embed_score
            + cfg.w_doctrine * doctrine_score
            + cfg.w_statute * statute_score
            + cfg.w_posture * posture_score
        )
        final_scores[cid] = combined
        breakdowns[cid] = ScoreBreakdown(
            embed=embed_score,
            doctrine=doctrine_score,
            statute=statute_score,
            posture=posture_score,
            raw_embed_agg=raw_embed_agg.get(cid, 0.0),
        )
    logger.info(
        "[retrieve_top_k_case_cards] step4 computed final scores for %d candidates",
        len(final_scores),
    )

    # ── Step 5: Case-type gating ──
    pre_gate_ranked = sorted(
        candidate_ids, key=lambda c: final_scores.get(c, 0), reverse=True,
    )

    gated_out: List[str] = []
    if not cfg.allow_cross_type and q_case_type:
        post_gate: List[str] = []
        for cid in pre_gate_ranked:
            ct = candidate_types.get(cid)
            if ct is None or ct == q_case_type:
                post_gate.append(cid)
            else:
                gated_out.append(cid)
        ranked = post_gate
    else:
        ranked = pre_gate_ranked
    logger.info(
        "[retrieve_top_k_case_cards] step5 gating q_case_type=%s pre_gate=%d gated_out=%d post_gate=%d",
        q_case_type,
        len(pre_gate_ranked),
        len(gated_out),
        len(ranked),
    )

    top_ids = ranked[:effective_k]
    if not top_ids:
        logger.warning(
            "[retrieve_top_k_case_cards] no top_ids after ranking/gating. "
            "If pre_gate>0 and post_gate=0, case-type gating likely filtered all candidates."
        )
    else:
        logger.info(
            "[retrieve_top_k_case_cards] step6 top_ids=%s",
            top_ids,
        )

    # ── Step 6: Build result objects ──
    results: List[RetrievedCaseCard] = []
    for cid in top_ids:
        meta = meta_map.get(cid, {})
        results.append(
            RetrievedCaseCard(
                case_id=cid,
                score=final_scores.get(cid, 0.0),
                case_card_text=meta.get("case_card_text") or "",
                case_type=meta.get("case_type"),
                issue_tags=meta.get("issue_tags"),
                statute_tags=meta.get("statute_tags"),
                doctrine_tags=meta.get("doctrine_tags"),
                procedural_posture=meta.get("procedural_posture"),
                score_breakdown=breakdowns.get(cid),
            )
        )

    # ── Step 7: Emit structured retrieval log ──
    retrieval_log = _build_retrieval_log(
        cfg=cfg,
        query_signals=query_signals,
        pre_gate_ranked=pre_gate_ranked,
        gated_out=gated_out,
        top_ids=top_ids,
        final_scores=final_scores,
        breakdowns=breakdowns,
        candidate_types=candidate_types,
    )
    logger.info(json.dumps(retrieval_log))
    logger.info(
        "[retrieve_top_k_case_cards] complete results_count=%d",
        len(results),
    )

    return results


# ─────────────────────────────────────────────
# Structured Retrieval Log
# ─────────────────────────────────────────────

def _build_retrieval_log(
    *,
    cfg: RetrievalConfig,
    query_signals: Optional[QuerySignals],
    pre_gate_ranked: List[str],
    gated_out: List[str],
    top_ids: List[str],
    final_scores: Dict[str, float],
    breakdowns: Dict[str, ScoreBreakdown],
    candidate_types: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    def _entry(cid: str) -> Dict[str, Any]:
        bd = breakdowns.get(cid)
        return {
            "case_id": cid,
            "final_score": round(final_scores.get(cid, 0), 6),
            "case_type": candidate_types.get(cid),
            "breakdown": {
                "embed": round(bd.embed, 6),
                "doctrine": round(bd.doctrine, 6),
                "statute": round(bd.statute, 6),
                "posture": round(bd.posture, 6),
                "raw_embed_agg": round(bd.raw_embed_agg, 6),
            } if bd else None,
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "k": cfg.k,
            "candidate_n": cfg.candidate_n,
            "chunk_cap": cfg.chunk_cap,
            "decay": cfg.decay,
            "allow_cross_type": cfg.allow_cross_type,
            "weights": {
                "embed": cfg.w_embed,
                "doctrine": cfg.w_doctrine,
                "statute": cfg.w_statute,
                "posture": cfg.w_posture,
            },
        },
        "query_signals": asdict(query_signals) if query_signals else None,
        "total_candidates": len(pre_gate_ranked),
        "candidates_before_gate": [_entry(c) for c in pre_gate_ranked[:20]],
        "gated_out_count": len(gated_out),
        "gated_out_sample": [_entry(c) for c in gated_out[:5]],
        "final_top_k": [_entry(c) for c in top_ids],
    }


# ─────────────────────────────────────────────
# Helper Queries (unchanged)
# ─────────────────────────────────────────────

def fetch_case_metadata(
    conn: psycopg.Connection, case_id: str
) -> Optional[Dict[str, Optional[str]]]:
    """
    Fetch case_type, procedural_posture, docket_number from cases table.
    Returns None if case not in DB.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT case_type, procedural_posture, docket_number
            FROM cases WHERE case_id = %s;
            """,
            (case_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def fetch_case_card_text(conn: psycopg.Connection, case_id: str) -> Optional[str]:
    """
    Fetch case_card_text for a case if it exists. Returns None if not in DB.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT case_card_text FROM case_cards WHERE case_id = %s;",
            (case_id,),
        )
        row = cur.fetchone()
    return row["case_card_text"] if row else None


def _split_case_card_main_and_addendum(text: str) -> Tuple[str, Optional[str]]:
    """
    If case_card_text has an addendum section, split so we can show main content
    (procedural posture, key facts, etc.) first and addendum last.
    """
    text = (text or "").strip()
    if not text:
        return "", None
    addendum_match = re.search(
        r"\n\s*(\*\*)?\s*Addendum\s*(\*\*)?\s*:?\s*\n",
        text,
        re.IGNORECASE,
    )
    if addendum_match:
        main_part = text[: addendum_match.start()].strip()
        addendum_part = text[addendum_match.end() :].strip()
        return main_part, addendum_part if addendum_part else None
    return text, None


def summarize_analyzed_case_from_briefs(brief_text: str) -> Optional[Dict[str, Any]]:
    """
    When the analyzed (query) case is not in the database, call OpenAI to extract
    case type, summary, procedural posture, standards of review from the brief text.
    Returns a dict with keys: case_type, procedural_posture, summary, standards_of_review.
    Returns None on failure or empty input.
    """
    brief_text = (brief_text or "").strip()
    if not brief_text or len(brief_text) > 50_000:
        return None

    from openai import OpenAI

    client = OpenAI(api_key=API_KEY)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    prompt = """From the following litigation brief(s) text, extract and return a JSON object with exactly these keys (use empty string if unclear):
- case_type: e.g. civil, criminal, administrative, appeal; party names or court if evident
- procedural_posture: e.g. appeal from district court, motion to dismiss, summary judgment
- summary: 2-4 sentence summary of the dispute, key facts, and main issues
- standards_of_review: brief list of applicable standards (e.g. de novo, abuse of discretion, clear error)

Return only valid JSON, no markdown or explanation."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt + "\n\n---\n\n" + brief_text[:45_000]},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return None
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)
        if not isinstance(data, dict):
            return None
        raw_sor = data.get("standards_of_review")
        if isinstance(raw_sor, list):
            standards_of_review = "\n".join(str(x).strip() for x in raw_sor if x)
        else:
            standards_of_review = (raw_sor or "").strip() or None
        return {
            "case_type": (data.get("case_type") or "").strip() or None,
            "procedural_posture": (data.get("procedural_posture") or "").strip() or None,
            "summary": (data.get("summary") or "").strip() or None,
            "standards_of_review": standards_of_review,
        }
    except (json.JSONDecodeError, KeyError, Exception):
        return None


# ─────────────────────────────────────────────
# End-to-End API
# ─────────────────────────────────────────────

def retrieve_similar_cases_for_new_case(
    *,
    db_url: str,
    briefs: List[BriefInput],
    k: int = DEFAULT_K,
    candidate_chunks: int = DEFAULT_CANDIDATE_CHUNKS,
    config: Optional[RetrievalConfig] = None,
    llm_metadata: Optional[Dict[str, Any]] = None,
) -> List[RetrievedCaseCard]:
    """
    Full pipeline:
      - Extract query signals (regex, optionally LLM-augmented)
      - Build fingerprint with signals front-loaded
      - Embed
      - Retrieve top-k cases with reranking and gating
    """
    if not briefs:
        logger.warning("[retrieve_similar_cases_for_new_case] No briefs provided; returning [].")
        return []

    cfg = config or RetrievalConfig(k=k)
    logger.info(
        "[retrieve_similar_cases_for_new_case] start briefs=%d requested_k=%d effective_k=%d "
        "candidate_chunks=%d db_url_set=%s",
        len(briefs),
        k,
        cfg.k,
        candidate_chunks,
        bool(db_url),
    )
    signals = extract_query_signals(
        [b.text for b in briefs], llm_metadata=llm_metadata,
    )
    logger.info(
        "[retrieve_similar_cases_for_new_case] extracted signals case_type=%s posture=%s "
        "posture_bucket=%s doctrines=%d statutes=%d issues=%d",
        signals.case_type,
        signals.procedural_posture,
        signals.posture_bucket,
        len(signals.doctrine_tags),
        len(signals.statute_tags),
        len(signals.issue_tags),
    )
    fingerprint = build_query_fingerprint(briefs, query_signals=signals)
    logger.info(
        "[retrieve_similar_cases_for_new_case] fingerprint chars=%d preview=%s",
        len(fingerprint),
        _preview(fingerprint),
    )
    embedding = embed_text_openai(fingerprint)
    logger.info(
        "[retrieve_similar_cases_for_new_case] embedding size=%d",
        len(embedding),
    )

    with psycopg.connect(db_url) as conn:
        results = retrieve_top_k_case_cards(
            conn=conn,
            query_embedding=embedding,
            k=k,
            candidate_chunks=candidate_chunks,
            config=cfg,
            query_signals=signals,
        )
    logger.info(
        "[retrieve_similar_cases_for_new_case] end results_count=%d result_case_ids=%s",
        len(results),
        [r.case_id for r in results],
    )
    return results


# ─────────────────────────────────────────────
# Frontend: PDF uploads (drag-and-drop)
# ─────────────────────────────────────────────


def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_chars: int = 1_500_000) -> str:
    """
    Extract text from a PDF given as in-memory bytes (e.g. from an uploaded file).
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
        if sum(len(p) for p in parts) >= max_chars:
            break
    text = "\n\n".join(parts)
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def retrieve_similar_cases_from_pdf_uploads(
    uploads: List[Tuple[str, bytes]],
    *,
    db_url: Optional[str] = None,
    k: int = DEFAULT_K,
    candidate_chunks: int = DEFAULT_CANDIDATE_CHUNKS,
    config: Optional[RetrievalConfig] = None,
) -> List[RetrievedCaseCard]:
    """
    Version of retrieve_similar_cases_for_new_case for frontend PDF uploads
    (e.g. drag-and-drop). Accepts a list of (label, pdf_bytes) pairs; each
    label is typically the filename (e.g. "appellant_brief.pdf").

    Args:
        uploads: List of (label, pdf_bytes). Label is used for logging/fingerprint.
        db_url: Database URL. If None, uses config.DATABASE_URL.
        k: Number of similar cases to return.
        candidate_chunks: Number of chunks to consider before aggregating by case.
        config: Optional RetrievalConfig for tuning parameters.

    Returns:
        List of RetrievedCaseCard for the top-k similar cases.
    """
    if not uploads:
        logger.warning("[retrieve_similar_cases_from_pdf_uploads] No uploads provided; returning [].")
        return []

    url = db_url or DATABASE_URL
    briefs: List[BriefInput] = []
    logger.info(
        "[retrieve_similar_cases_from_pdf_uploads] start uploads=%d requested_k=%d candidate_chunks=%d",
        len(uploads),
        k,
        candidate_chunks,
    )
    for label, pdf_bytes in uploads:
        text = extract_text_from_pdf_bytes(pdf_bytes)
        name = (label or "brief").strip()
        if name.lower().endswith(".pdf"):
            name = name[:-4].strip() or "brief"
        briefs.append(BriefInput(label=name, text=text))
        logger.info(
            "[retrieve_similar_cases_from_pdf_uploads] extracted text label=%s pdf_bytes=%d text_chars=%d text_preview=%s",
            name,
            len(pdf_bytes or b""),
            len(text),
            _preview(text),
        )

    results = retrieve_similar_cases_for_new_case(
        db_url=url,
        briefs=briefs,
        k=k,
        candidate_chunks=candidate_chunks,
        config=config,
    )
    logger.info(
        "[retrieve_similar_cases_from_pdf_uploads] end results_count=%d result_case_ids=%s",
        len(results),
        [r.case_id for r in results],
    )
    return results


# ─────────────────────────────────────────────
# Test dataset iteration
# ─────────────────────────────────────────────

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
            label = f[:-4].strip()
            if label.lower().startswith(name.lower()):
                label = label[len(name) :].strip() or "document"
            doc_list.append((label or "document", pdf_path))
        if not doc_list:
            continue
        yield case_dir, doc_list


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler("retrieval_log.jsonl", mode="a")],
    )

    try:
        from rag_context.case_ingestion import extract_text_from_pdf
    except ImportError:
        from case_ingestion import extract_text_from_pdf

    test_dir = os.path.abspath(DEFAULT_TEST_DATASET)
    cfg = RetrievalConfig(k=DEFAULT_K)
    main_summary_chars = 1200
    addendum_excerpt_chars = 200
    related_card_preview_chars = 500

    case_tally: Dict[str, int] = {}

    for case_dir, doc_list in _iter_test_cases(test_dir):
        case_id = os.path.basename(case_dir)
        briefs = [
            BriefInput(label=label, text=extract_text_from_pdf(pdf_path))
            for label, pdf_path in doc_list
        ]

        with psycopg.connect(DATABASE_URL) as conn:
            case_meta = fetch_case_metadata(conn, case_id)
            analyzed_summary = fetch_case_card_text(conn, case_id)

            # Build LLM metadata dict from DB or OpenAI
            llm_meta: Optional[Dict[str, Any]] = None
            llm_summary: Optional[Dict[str, Any]] = None
            if case_meta:
                llm_meta = {
                    "case_type": case_meta.get("case_type"),
                    "procedural_posture": case_meta.get("procedural_posture"),
                }
            elif not (analyzed_summary and analyzed_summary.strip()):
                combined_brief = "\n\n".join(b.text[:20_000] for b in briefs)
                llm_summary = summarize_analyzed_case_from_briefs(combined_brief)
                llm_meta = llm_summary

            # Extract query signals
            signals = extract_query_signals(
                [b.text for b in briefs], llm_metadata=llm_meta,
            )

            # Override with DB tags when available (more accurate than regex)
            if analyzed_summary:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        "SELECT issue_tags, statute_tags, doctrine_tags "
                        "FROM case_cards WHERE case_id = %s",
                        (case_id,),
                    )
                    tag_row = cur.fetchone()
                if tag_row:
                    if tag_row.get("doctrine_tags"):
                        signals.doctrine_tags = [t.lower() for t in tag_row["doctrine_tags"]]
                    if tag_row.get("statute_tags"):
                        signals.statute_tags = [t.lower() for t in tag_row["statute_tags"]]
                    if tag_row.get("issue_tags"):
                        signals.issue_tags = [t.lower() for t in tag_row["issue_tags"]]

            fingerprint = build_query_fingerprint(briefs, query_signals=signals)
            embedding = embed_text_openai(fingerprint)

            results = retrieve_top_k_case_cards(
                conn=conn,
                query_embedding=embedding,
                k=cfg.k,
                config=cfg,
                query_signals=signals,
            )

        for r in results:
            case_tally[r.case_id] = case_tally.get(r.case_id, 0) + 1

        # ──── Analyzed case (query) ────
        print("\n" + "=" * 60)
        print("  ANALYZED CASE")
        print("=" * 60)
        print(f"  Case ID:   {case_id}")
        if case_meta:
            case_type = case_meta.get("case_type") or "(not set)"
            proc_posture = case_meta.get("procedural_posture") or "(not set)"
            docket = case_meta.get("docket_number") or ""
        else:
            case_type = (llm_summary and llm_summary.get("case_type")) or "(not in DB)"
            proc_posture = (llm_summary and llm_summary.get("procedural_posture")) or "(not in DB)"
            docket = ""
        print(f"  Case type: {case_type}")
        print(f"  Procedural posture: {proc_posture}")
        if docket:
            print(f"  Docket:    {docket}")
        print(f"  Briefs:    {len(briefs)} ({', '.join(b.label for b in briefs)})")

        # Query signals summary
        print(f"  Query signals: type={signals.case_type}, posture_bucket={signals.posture_bucket}")
        if signals.doctrine_tags:
            print(f"    doctrines: {signals.doctrine_tags[:5]}")
        if signals.statute_tags:
            print(f"    statutes:  {signals.statute_tags[:5]}")
        print()

        if analyzed_summary and analyzed_summary.strip():
            main_part, addendum_part = _split_case_card_main_and_addendum(
                analyzed_summary
            )
            to_show = main_part
            if len(to_show) > main_summary_chars:
                to_show = to_show[:main_summary_chars] + "\n..."
            print("  Summary (case_card — procedural posture, key facts, issues):")
            for line in to_show.split("\n"):
                print(f"    {line}")
            if addendum_part:
                excerpt = addendum_part
                if len(excerpt) > addendum_excerpt_chars:
                    excerpt = excerpt[:addendum_excerpt_chars] + "..."
                print()
                print("  Addendum (excerpt):")
                for line in excerpt.split("\n"):
                    print(f"    {line}")
        else:
            if llm_summary and (llm_summary.get("summary") or llm_summary.get("standards_of_review")):
                print("  Summary (from briefs, OpenAI-extracted):")
                if llm_summary.get("summary"):
                    for line in (llm_summary["summary"] or "").split("\n"):
                        print(f"    {line}")
                if llm_summary.get("standards_of_review"):
                    print()
                    print("  Standards of review:")
                    for line in (llm_summary["standards_of_review"] or "").split("\n"):
                        print(f"    {line}")
            else:
                summary = fingerprint
                if len(summary) > main_summary_chars:
                    summary = summary[:main_summary_chars] + "\n..."
                print("  Summary (from briefs, no case_card in DB):")
                for line in summary.split("\n"):
                    print(f"    {line}")
        print()

        # ──── Related cases (retrieved) ────
        print("-" * 60)
        print(f"  RELATED CASES (top-{cfg.k})")
        print("-" * 60)
        for i, r in enumerate(results, 1):
            case_type_str = f"  [{r.case_type}]" if r.case_type else ""
            print(f"\n  [{i}] {r.case_id}  (score: {r.score:.4f}){case_type_str}")

            if r.score_breakdown:
                bd = r.score_breakdown
                print(f"      breakdown: embed={bd.embed:.4f}  doctrine={bd.doctrine:.4f}"
                      f"  statute={bd.statute:.4f}  posture={bd.posture:.4f}"
                      f"  raw_agg={bd.raw_embed_agg:.4f}")

            card = (r.case_card_text or "").strip()
            if len(card) > related_card_preview_chars:
                card = card[:related_card_preview_chars] + "..."
            if card:
                for line in card.split("\n"):
                    print(f"      {line}")
            tags = []
            if r.issue_tags:
                tags.append(f"issues={r.issue_tags}")
            if r.statute_tags:
                tags.append(f"statutes={r.statute_tags}")
            if r.doctrine_tags:
                tags.append(f"doctrines={r.doctrine_tags}")
            if tags:
                print(f"      Tags: {', '.join(tags)}")
        print()

    # ──── Top-k retrieval tally ────
    print("=" * 60)
    print("  TOP-K RETRIEVAL TALLY (times each case was pulled as a similar case)")
    print("=" * 60)
    if not case_tally:
        print("  (no retrievals)")
    else:
        for case_id, count in sorted(case_tally.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {case_id}: {count}")
    print()


if __name__ == "__main__":
    main()
