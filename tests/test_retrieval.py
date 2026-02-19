"""
Tests for the v2 retrieval pipeline.
Covers: scoring utilities, signal extraction, reranking, case-type gating,
hub-dominance mitigation, and end-to-end civil §1983 filtering.
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_context"))

from top_k_retrieval import (
    RetrievalConfig,
    QuerySignals,
    ScoreBreakdown,
    RetrievedCaseCard,
    _jaccard,
    _aggregate_with_decay,
    _bucketize_posture,
    _normalize_case_type,
    _infer_case_type_from_text,
    _strip_boilerplate,
    extract_query_signals,
    build_query_fingerprint,
    retrieve_top_k_case_cards,
    BriefInput,
)


# ─────────────────────────────────────────────
# Helpers for mocking psycopg connections
# ─────────────────────────────────────────────

def _make_mock_conn(chunk_rows: List[Dict], meta_rows: List[Dict]):
    """
    Build a mock psycopg connection that returns chunk_rows on the first
    cursor block and meta_rows on the second.
    """
    mock_conn = MagicMock()
    responses = [chunk_rows, meta_rows]
    call_idx = {"i": 0}

    def _cursor(**kwargs):
        cur = MagicMock()
        idx = call_idx["i"]
        call_idx["i"] += 1
        cur.fetchall.return_value = responses[idx] if idx < len(responses) else []
        cur.__enter__ = lambda s: s
        cur.__exit__ = lambda s, *a: None
        return cur

    mock_conn.cursor = _cursor
    return mock_conn


# ─────────────────────────────────────────────
# Unit tests: Jaccard
# ─────────────────────────────────────────────

class TestJaccard:
    def test_identical_sets(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        assert abs(_jaccard({"a", "b", "c"}, {"b", "c", "d"}) - 0.5) < 1e-9

    def test_both_empty(self):
        assert _jaccard(set(), set()) == 0.0

    def test_one_empty(self):
        assert _jaccard({"a"}, set()) == 0.0


# ─────────────────────────────────────────────
# Unit tests: Aggregation with decay
# ─────────────────────────────────────────────

class TestAggregationDecay:
    def test_single_chunk(self):
        assert abs(_aggregate_with_decay([0.8], 3, 0.85) - 0.8) < 1e-9

    def test_cap_limits_contributions(self):
        sims = [0.9, 0.8, 0.7, 0.6, 0.5]
        result = _aggregate_with_decay(sims, chunk_cap=3, decay=0.85)
        expected = 0.9 + 0.8 * 0.85 + 0.7 * 0.85**2
        assert abs(result - expected) < 1e-9

    def test_decay_penalizes_hub_dominance(self):
        many_generic = [0.5] * 20
        few_specific = [0.9, 0.85]
        score_many = _aggregate_with_decay(many_generic, chunk_cap=3, decay=0.85)
        score_few = _aggregate_with_decay(few_specific, chunk_cap=3, decay=0.85)
        assert score_few > score_many

    def test_empty_sims(self):
        assert _aggregate_with_decay([], 3, 0.85) == 0.0


# ─────────────────────────────────────────────
# Unit tests: Posture bucketization
# ─────────────────────────────────────────────

class TestPostureBucket:
    def test_summary_judgment(self):
        assert _bucketize_posture("appeal from summary judgment") == "summary_judgment"

    def test_12b6(self):
        assert _bucketize_posture("motion to dismiss under 12(b)(6)") == "12b6"

    def test_sentencing(self):
        assert _bucketize_posture("appeal from sentencing") == "sentencing"

    def test_none(self):
        assert _bucketize_posture(None) is None

    def test_no_match(self):
        assert _bucketize_posture("completely novel posture xyz") is None


# ─────────────────────────────────────────────
# Unit tests: Case type normalization
# ─────────────────────────────────────────────

class TestCaseTypeNormalization:
    def test_criminal_variants(self):
        assert _normalize_case_type("Criminal") == "criminal"
        assert _normalize_case_type("criminal sentencing") == "criminal"
        assert _normalize_case_type("appeal from conviction") == "criminal"

    def test_civil_variants(self):
        assert _normalize_case_type("Civil") == "civil"
        assert _normalize_case_type("prisoner civil rights 1983") == "civil"

    def test_immigration(self):
        assert _normalize_case_type("immigration removal") == "immigration"

    def test_none(self):
        assert _normalize_case_type(None) is None

    def test_passthrough(self):
        assert _normalize_case_type("tax") == "tax"


class TestInferCaseType:
    def test_civil_1983_text(self):
        text = (
            "42 u.s.c. section 1983 deliberate indifference excessive force "
            "prison conditions civil rights equal protection "
        )
        assert _infer_case_type_from_text(text) == "civil"

    def test_criminal_text(self):
        text = (
            "criminal sentencing guilty plea restitution "
            "guidelines range 18 u.s.c conviction "
        )
        assert _infer_case_type_from_text(text) == "criminal"

    def test_ambiguous_returns_none(self):
        assert _infer_case_type_from_text("hello world") is None


# ─────────────────────────────────────────────
# Unit tests: Signal extraction
# ─────────────────────────────────────────────

class TestExtractSignals:
    def test_civil_1983_signals(self):
        text = (
            "This is a civil rights case under 42 U.S.C. § 1983 alleging "
            "deliberate indifference to serious medical needs in violation of "
            "the Eighth Amendment. The plaintiff is a state prisoner seeking "
            "damages for cruel and unusual punishment. The district court "
            "granted summary judgment based on qualified immunity. "
            "PLRA exhaustion requirements are at issue."
        )
        signals = extract_query_signals([text])
        assert signals.case_type == "civil"
        assert any("1983" in t for t in signals.statute_tags)
        assert "qualified immunity" in signals.doctrine_tags
        assert "cruel and unusual" in signals.doctrine_tags
        assert "prison conditions" in signals.issue_tags or "medical care" in signals.issue_tags

    def test_criminal_signals(self):
        text = (
            "Defendant appeals conviction and sentence for conspiracy to "
            "distribute controlled substances in violation of 21 U.S.C. § 846. "
            "The guidelines range was calculated based on drug quantity. "
            "Defendant challenges the restitution order and sentencing."
        )
        signals = extract_query_signals([text])
        assert signals.case_type == "criminal"

    def test_llm_metadata_override(self):
        text = "some ambiguous text"
        meta = {"case_type": "civil", "procedural_posture": "summary judgment"}
        signals = extract_query_signals([text], llm_metadata=meta)
        assert signals.case_type == "civil"
        assert signals.procedural_posture == "summary judgment"
        assert signals.posture_bucket == "summary_judgment"


# ─────────────────────────────────────────────
# Unit tests: Fingerprint builder
# ─────────────────────────────────────────────

class TestFingerprint:
    def test_includes_signals(self):
        signals = QuerySignals(
            case_type="civil",
            statute_tags=["42 U.S.C. § 1983"],
            doctrine_tags=["qualified immunity"],
        )
        fp = build_query_fingerprint(
            [BriefInput(label="test", text="brief text here")],
            query_signals=signals,
        )
        assert "CASE TYPE: civil" in fp
        assert "42 U.S.C. § 1983" in fp
        assert "qualified immunity" in fp

    def test_backward_compat_without_signals(self):
        fp = build_query_fingerprint(
            [BriefInput(label="test", text="brief text here")],
        )
        assert "brief text here" in fp

    def test_strips_certificate(self):
        text = "Main argument text.\nCERTIFICATE OF COMPLIANCE\nI hereby certify..."
        fp = build_query_fingerprint([BriefInput(label="b", text=text)])
        assert "CERTIFICATE OF COMPLIANCE" not in fp


# ─────────────────────────────────────────────
# Integration tests: Reranking
# ─────────────────────────────────────────────

class TestReranking:
    def test_doctrine_overlap_boosts_ranking(self):
        """Case B (lower embed, high doctrine overlap) should beat Case A (higher embed, no overlap)."""
        chunk_rows = [
            {"case_id": "A", "chunk_id": "a1", "sim": 0.95},
            {"case_id": "A", "chunk_id": "a2", "sim": 0.90},
            {"case_id": "B", "chunk_id": "b1", "sim": 0.88},
            {"case_id": "B", "chunk_id": "b2", "sim": 0.85},
        ]
        meta_rows = [
            {
                "case_id": "A", "case_type": "civil",
                "procedural_posture": "summary judgment",
                "case_card_text": "Case A card",
                "issue_tags": [], "statute_tags": ["42 U.S.C. § 1983"],
                "doctrine_tags": [],
            },
            {
                "case_id": "B", "case_type": "civil",
                "procedural_posture": "summary judgment",
                "case_card_text": "Case B card",
                "issue_tags": [], "statute_tags": ["42 U.S.C. § 1983"],
                "doctrine_tags": ["qualified immunity", "deliberate indifference"],
            },
        ]
        conn = _make_mock_conn(chunk_rows, meta_rows)
        signals = QuerySignals(
            case_type="civil",
            doctrine_tags=["qualified immunity", "deliberate indifference"],
            statute_tags=["42 U.S.C. § 1983"],
        )
        cfg = RetrievalConfig(k=2, candidate_n=10, chunk_cap=3, decay=0.85)
        results = retrieve_top_k_case_cards(
            conn, [0.0] * 1536, k=2, config=cfg, query_signals=signals,
        )
        assert len(results) == 2
        assert results[0].case_id == "B"

    def test_statute_overlap_soft_boost(self):
        """Statute overlap should boost Y above X when embed scores are close."""
        chunk_rows = [
            {"case_id": "X", "chunk_id": "x1", "sim": 0.90},
            {"case_id": "Y", "chunk_id": "y1", "sim": 0.89},
        ]
        meta_rows = [
            {
                "case_id": "X", "case_type": "civil",
                "procedural_posture": None, "case_card_text": "Case X",
                "issue_tags": [], "statute_tags": [],
                "doctrine_tags": [],
            },
            {
                "case_id": "Y", "case_type": "civil",
                "procedural_posture": None, "case_card_text": "Case Y",
                "issue_tags": [],
                "statute_tags": ["42 u.s.c. § 1983", "28 u.s.c. § 1915"],
                "doctrine_tags": [],
            },
        ]
        conn = _make_mock_conn(chunk_rows, meta_rows)
        signals = QuerySignals(
            case_type="civil",
            statute_tags=["42 U.S.C. § 1983", "28 U.S.C. § 1915"],
        )
        cfg = RetrievalConfig(k=2)
        results = retrieve_top_k_case_cards(
            conn, [0.0] * 1536, k=2, config=cfg, query_signals=signals,
        )
        assert results[0].case_id == "Y"

    def test_posture_match_mild_boost(self):
        """Matching posture bucket provides a mild boost."""
        chunk_rows = [
            {"case_id": "P1", "chunk_id": "p1a", "sim": 0.80},
            {"case_id": "P2", "chunk_id": "p2a", "sim": 0.80},
        ]
        meta_rows = [
            {
                "case_id": "P1", "case_type": "civil",
                "procedural_posture": "sentencing",
                "case_card_text": "P1", "issue_tags": [],
                "statute_tags": [], "doctrine_tags": [],
            },
            {
                "case_id": "P2", "case_type": "civil",
                "procedural_posture": "appeal from summary judgment",
                "case_card_text": "P2", "issue_tags": [],
                "statute_tags": [], "doctrine_tags": [],
            },
        ]
        conn = _make_mock_conn(chunk_rows, meta_rows)
        signals = QuerySignals(
            case_type="civil",
            procedural_posture="summary judgment",
            posture_bucket="summary_judgment",
        )
        cfg = RetrievalConfig(k=2)
        results = retrieve_top_k_case_cards(
            conn, [0.0] * 1536, k=2, config=cfg, query_signals=signals,
        )
        assert results[0].case_id == "P2"


# ─────────────────────────────────────────────
# Integration tests: Case-type gating
# ─────────────────────────────────────────────

class TestCaseTypeGating:
    def test_criminal_filtered_from_civil_query(self):
        """Criminal cases must not appear in final results for a civil query."""
        chunk_rows = [
            {"case_id": "CRIM", "chunk_id": "c1", "sim": 0.95},
            {"case_id": "CIVIL", "chunk_id": "v1", "sim": 0.80},
        ]
        meta_rows = [
            {
                "case_id": "CRIM", "case_type": "criminal",
                "procedural_posture": "sentencing",
                "case_card_text": "Criminal card",
                "issue_tags": [], "statute_tags": [],
                "doctrine_tags": [],
            },
            {
                "case_id": "CIVIL", "case_type": "civil",
                "procedural_posture": "summary judgment",
                "case_card_text": "Civil card",
                "issue_tags": [], "statute_tags": [],
                "doctrine_tags": [],
            },
        ]
        conn = _make_mock_conn(chunk_rows, meta_rows)
        signals = QuerySignals(case_type="civil")
        cfg = RetrievalConfig(k=2, allow_cross_type=False)
        results = retrieve_top_k_case_cards(
            conn, [0.0] * 1536, k=2, config=cfg, query_signals=signals,
        )
        for r in results:
            assert _normalize_case_type(r.case_type) != "criminal"

    def test_cross_type_allowed(self):
        """When allow_cross_type=True, criminal cases can appear for civil queries."""
        chunk_rows = [
            {"case_id": "CRIM", "chunk_id": "c1", "sim": 0.95},
            {"case_id": "CIVIL", "chunk_id": "v1", "sim": 0.80},
        ]
        meta_rows = [
            {
                "case_id": "CRIM", "case_type": "criminal",
                "procedural_posture": "sentencing",
                "case_card_text": "Criminal card",
                "issue_tags": [], "statute_tags": [],
                "doctrine_tags": [],
            },
            {
                "case_id": "CIVIL", "case_type": "civil",
                "procedural_posture": "summary judgment",
                "case_card_text": "Civil card",
                "issue_tags": [], "statute_tags": [],
                "doctrine_tags": [],
            },
        ]
        conn = _make_mock_conn(chunk_rows, meta_rows)
        signals = QuerySignals(case_type="civil")
        cfg = RetrievalConfig(k=2, allow_cross_type=True)
        results = retrieve_top_k_case_cards(
            conn, [0.0] * 1536, k=2, config=cfg, query_signals=signals,
        )
        ids = {r.case_id for r in results}
        assert "CRIM" in ids

    def test_unknown_type_passes_gate(self):
        """Cases with NULL case_type should pass through the gate."""
        chunk_rows = [
            {"case_id": "UNK", "chunk_id": "u1", "sim": 0.90},
            {"case_id": "CRIM", "chunk_id": "c1", "sim": 0.85},
        ]
        meta_rows = [
            {
                "case_id": "UNK", "case_type": None,
                "procedural_posture": None,
                "case_card_text": "Unknown card",
                "issue_tags": [], "statute_tags": [],
                "doctrine_tags": [],
            },
            {
                "case_id": "CRIM", "case_type": "criminal",
                "procedural_posture": "sentencing",
                "case_card_text": "Criminal card",
                "issue_tags": [], "statute_tags": [],
                "doctrine_tags": [],
            },
        ]
        conn = _make_mock_conn(chunk_rows, meta_rows)
        signals = QuerySignals(case_type="civil")
        cfg = RetrievalConfig(k=2, allow_cross_type=False)
        results = retrieve_top_k_case_cards(
            conn, [0.0] * 1536, k=2, config=cfg, query_signals=signals,
        )
        ids = {r.case_id for r in results}
        assert "UNK" in ids
        assert "CRIM" not in ids


# ─────────────────────────────────────────────
# End-to-end test: Civil §1983 query
# ─────────────────────────────────────────────

class TestEndToEndCivil1983:
    """
    Simulate a civil §1983 prison-condition / Eighth Amendment query against a
    candidate pool containing criminal sentencing, criminal RICO, and civil
    rights cases. Verify zero criminal cases in final top-k.
    """

    def test_civil_1983_excludes_all_criminal(self):
        chunk_rows = [
            {"case_id": "crim-sent", "chunk_id": "cs1", "sim": 0.95},
            {"case_id": "crim-sent", "chunk_id": "cs2", "sim": 0.93},
            {"case_id": "crim-rico", "chunk_id": "cr1", "sim": 0.92},
            {"case_id": "civil-qi",  "chunk_id": "cq1", "sim": 0.85},
            {"case_id": "civil-qi",  "chunk_id": "cq2", "sim": 0.83},
            {"case_id": "civil-ef",  "chunk_id": "ce1", "sim": 0.82},
            {"case_id": "civil-ef",  "chunk_id": "ce2", "sim": 0.80},
            {"case_id": "civil-emp", "chunk_id": "cm1", "sim": 0.78},
        ]
        meta_rows = [
            {
                "case_id": "crim-sent", "case_type": "criminal",
                "procedural_posture": "sentencing",
                "case_card_text": "Criminal sentencing—drug quantity, guidelines",
                "issue_tags": ["sentencing"],
                "statute_tags": ["18 U.S.C. § 3553", "21 U.S.C. § 841"],
                "doctrine_tags": [],
            },
            {
                "case_id": "crim-rico", "case_type": "criminal",
                "procedural_posture": "post-trial",
                "case_card_text": "RICO conspiracy conviction",
                "issue_tags": ["rico"],
                "statute_tags": ["18 U.S.C. § 1962"],
                "doctrine_tags": [],
            },
            {
                "case_id": "civil-qi", "case_type": "civil",
                "procedural_posture": "appeal from summary judgment on qualified immunity",
                "case_card_text": "Section 1983 prison conditions—deliberate indifference",
                "issue_tags": ["prison conditions", "medical care"],
                "statute_tags": ["42 U.S.C. § 1983"],
                "doctrine_tags": ["qualified immunity", "deliberate indifference"],
            },
            {
                "case_id": "civil-ef", "case_type": "civil",
                "procedural_posture": "12(b)(6) dismissal",
                "case_card_text": "Section 1983 excessive force claim",
                "issue_tags": ["use of force"],
                "statute_tags": ["42 U.S.C. § 1983"],
                "doctrine_tags": ["qualified immunity", "excessive force"],
            },
            {
                "case_id": "civil-emp", "case_type": "civil",
                "procedural_posture": "summary judgment",
                "case_card_text": "Title VII employment discrimination",
                "issue_tags": ["employment discrimination"],
                "statute_tags": ["Title VII"],
                "doctrine_tags": [],
            },
        ]

        conn = _make_mock_conn(chunk_rows, meta_rows)

        signals = QuerySignals(
            case_type="civil",
            procedural_posture="appeal from summary judgment",
            posture_bucket="summary_judgment",
            statute_tags=["42 U.S.C. § 1983"],
            doctrine_tags=["qualified immunity", "deliberate indifference"],
            issue_tags=["prison conditions", "medical care"],
        )

        cfg = RetrievalConfig(k=3, allow_cross_type=False)
        results = retrieve_top_k_case_cards(
            conn, [0.0] * 1536, k=3, config=cfg, query_signals=signals,
        )

        # Zero criminal cases in final top-k
        for r in results:
            norm_type = _normalize_case_type(r.case_type)
            assert norm_type != "criminal", (
                f"Criminal case '{r.case_id}' leaked into civil §1983 results"
            )

        # Should have results
        assert len(results) > 0

        # civil-qi should rank first (doctrine + statute + posture match)
        assert results[0].case_id == "civil-qi"

        # Every result should carry a score breakdown
        for r in results:
            assert r.score_breakdown is not None
            assert r.score_breakdown.embed >= 0
            assert r.score_breakdown.doctrine >= 0

    def test_score_breakdown_components_sum(self):
        """Verify final score ≈ weighted sum of component scores."""
        chunk_rows = [
            {"case_id": "A", "chunk_id": "a1", "sim": 0.90},
        ]
        meta_rows = [
            {
                "case_id": "A", "case_type": "civil",
                "procedural_posture": "summary judgment",
                "case_card_text": "card",
                "issue_tags": [],
                "statute_tags": ["42 U.S.C. § 1983"],
                "doctrine_tags": ["qualified immunity"],
            },
        ]
        conn = _make_mock_conn(chunk_rows, meta_rows)
        signals = QuerySignals(
            case_type="civil",
            posture_bucket="summary_judgment",
            statute_tags=["42 U.S.C. § 1983"],
            doctrine_tags=["qualified immunity"],
        )
        cfg = RetrievalConfig(k=1)
        results = retrieve_top_k_case_cards(
            conn, [0.0] * 1536, k=1, config=cfg, query_signals=signals,
        )
        assert len(results) == 1
        r = results[0]
        bd = r.score_breakdown
        expected = (
            cfg.w_embed * bd.embed
            + cfg.w_doctrine * bd.doctrine
            + cfg.w_statute * bd.statute
            + cfg.w_posture * bd.posture
        )
        assert abs(r.score - expected) < 1e-6


# ─────────────────────────────────────────────
# Boilerplate stripping
# ─────────────────────────────────────────────

class TestStripBoilerplate:
    def test_removes_certificate_of_service(self):
        text = "Argument.\nCERTIFICATE OF SERVICE\nI certify that on this day..."
        result = _strip_boilerplate(text)
        assert "CERTIFICATE OF SERVICE" not in result
        assert "Argument." in result

    def test_preserves_substantive_text(self):
        text = "The defendant violated 42 U.S.C. § 1983 through deliberate indifference."
        assert _strip_boilerplate(text) == text


def run_tests(verbose: bool = True) -> int:
    """
    Run all retrieval tests. Returns exit code (0 = pass, non-zero = fail).
    Usage: from tests.test_retrieval import run_tests; run_tests()
    Or: python tests/test_retrieval.py
    """
    return pytest.main(
        ["-v" if verbose else "-q", __file__],
        plugins=[],
    )


if __name__ == "__main__":
    exit(run_tests())
