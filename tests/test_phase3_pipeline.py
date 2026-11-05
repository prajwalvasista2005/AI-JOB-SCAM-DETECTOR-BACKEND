"""
Phase 3 — Canonical Detection Pipeline End-to-End & Failure Boundary Test Suite.
Verifies:
  1. Canonical 6-stage pipeline execution: analyze_job_posting().
  2. Failure Boundary Taxonomy (F-06 Categories 1-7).
  3. Anti-double-counting and multi-category compounding through full pipeline.
  4. Reproducible Performance Benchmark Protocol (F-07).
"""
import time
from unittest.mock import patch
import pytest

from app.services.detection import (
    CandidateEligibility,
    DetectionResult,
    DetectionSource,
    EvidenceCompleteness,
    EvidenceItem,
    FindingSeverity,
    FindingStatus,
    ScamCategory,
    analyze_job_posting,
    extract_regex_candidates,
    extract_semantic_candidates,
    reconcile_detection_engines,
)
from app.services.detection.reconciliation import RejectedCandidateRecord


# =============================================================================
# 1. CANONICAL PIPELINE END-TO-END EXECUTION TESTS
# =============================================================================

def test_pipeline_clean_job_posting():
    """Benign job posting without scam triggers returns 0 points (Low Risk)."""
    raw = (
        "We are looking for a Senior Software Engineer to build scalable backend systems. "
        "Requirements: 5+ years of experience with Python, FastAPI, and PostgreSQL. "
        "Competitive salary, full healthcare benefits, and remote work options provided."
    )
    result = analyze_job_posting(raw)
    assert isinstance(result, DetectionResult)
    assert result.rule_score == 0
    assert len(result.categories_detected) == 0
    assert len(result.evidence_items) == 0
    assert result.evidence_completeness == EvidenceCompleteness.MODERATE
    assert "Low Risk" in result.fallback_explanation


def test_pipeline_active_fee_demand():
    """Posting demanding upfront registration fee evaluates to 65 points (High Risk)."""
    raw = (
        "Selected candidates must pay a registration fee of Rs. 1000 before onboarding. "
        "Send your resume to apply."
    )
    result = analyze_job_posting(raw)
    assert isinstance(result, DetectionResult)
    assert result.rule_score == 65
    assert ScamCategory.ADVANCE_FEE in result.categories_detected
    assert len(result.evidence_items) > 0
    assert "High Risk" in result.fallback_explanation

    # Verify that quotation integrity is preserved on emitted items
    for item in result.evidence_items:
        if item.start_pos is not None and item.end_pos is not None:
            assert raw[item.start_pos:item.end_pos] == item.matched_text


def test_pipeline_corroborated_occurrence_anti_double_counting():
    """When both regex and semantic engines detect the same occurrence, score is 65 (not 130)."""
    raw = "Candidates must pay a registration fee before attending the interview."
    result = analyze_job_posting(raw)

    assert result.rule_score == 65
    # Both items are preserved for audit/evidence, but secondary has wt=0
    assert len(result.evidence_items) >= 2

    scorable_items = [
        it for it in result.evidence_items
        if it.eligibility in (CandidateEligibility.SCORABLE_ACTIVE, CandidateEligibility.SCORABLE_AMBIGUOUS)
    ]
    # Exactly one item is scorable for that occurrence
    assert len(scorable_items) == 1

    secondary_items = [
        it for it in result.evidence_items
        if it.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    ]
    assert len(secondary_items) >= 1
    # Check that secondary carries corroboration metadata
    sec = secondary_items[0]
    assert "corroborated_with:" in (sec.context_required or "")
    assert "scoring weight suppressed to prevent double-counting" in sec.explanation


def test_pipeline_multi_category_compounding():
    """Fee demand (65) + Fake recruitment (45) compounds to 75 points."""
    raw = (
        "Candidates must pay a registration fee before attending the interview. "
        "We offer 100% guaranteed job placement."
    )
    result = analyze_job_posting(raw)

    assert result.rule_score == 75  # 65 anchor + 10 residual boost
    assert ScamCategory.ADVANCE_FEE in result.categories_detected
    assert ScamCategory.FAKE_RECRUITMENT in result.categories_detected


# =============================================================================
# 2. FAILURE BOUNDARY TAXONOMY TESTS (F-06 Categories 1 to 7)
# =============================================================================

def test_pipeline_category_1_invalid_candidate_rejected_does_not_abort():
    """F-06 Cat 1: Individual malformed candidate is rejected into rejected_candidates; pipeline continues."""
    raw = "Candidates must pay a registration fee of Rs. 1000."

    # Intercept extract_semantic_candidates to inject a non-EvidenceItem and a valid one
    malformed_candidate = {"invalid": "not_an_evidence_item"}
    valid_candidate = EvidenceItem(
        id="SEM_ADV_FEE_DIRECTIVE",
        category=ScamCategory.ADVANCE_FEE,
        source=DetectionSource.DETERMINISTIC_RULE,
        severity=FindingSeverity.CRITICAL,
        status=FindingStatus.EXPLICIT_CLAIM,
        matched_text="pay a registration fee of Rs. 1000",
        explanation="Valid semantic candidate",
        start_pos=16,
        end_pos=50,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
    )

    with patch(
        "app.services.detection.service.extract_semantic_candidates",
        return_value=[malformed_candidate, valid_candidate],
    ):
        result = analyze_job_posting(raw)
        # Pipeline must not crash, must return valid DetectionResult
        assert isinstance(result, DetectionResult)
        assert result.rule_score == 65


def test_pipeline_category_2_empty_semantic_substep_proceeds_normally():
    """F-06 Cat 2: Semantic engine yielding empty list is not an error; regex proceeds normally."""
    raw = "Candidates must pay a registration fee of Rs. 1000."
    with patch("app.services.detection.service.extract_semantic_candidates", return_value=[]):
        result = analyze_job_posting(raw)
        assert isinstance(result, DetectionResult)
        assert result.rule_score == 65
        assert ScamCategory.ADVANCE_FEE in result.categories_detected


def test_pipeline_category_3_invalid_raw_content_raises_value_error():
    """F-06 Cat 3: None, non-string, or whitespace-only raises ValueError immediately before any stage."""
    with pytest.raises(ValueError, match="raw_content must be a non-empty string"):
        analyze_job_posting(None)

    with pytest.raises(ValueError, match="raw_content must be a non-empty string"):
        analyze_job_posting("")

    with pytest.raises(ValueError, match="raw_content must be a non-empty string"):
        analyze_job_posting("   \n\t  ")

    with pytest.raises(ValueError, match="raw_content must be a non-empty string"):
        analyze_job_posting(12345)


def test_pipeline_category_4_quotation_validation_failure_candidate():
    """F-06 Cat 4: Candidate with corrupt bounds fails quotation validation and is rejected fail-closed."""
    raw = "Candidates must pay a registration fee of Rs. 1000."
    # Inject a candidate with invalid bounds that will fail validate_evidence_quotation
    corrupt_candidate = EvidenceItem(
        id="SEM_ADV_FEE_DIRECTIVE",
        category=ScamCategory.ADVANCE_FEE,
        source=DetectionSource.DETERMINISTIC_RULE,
        severity=FindingSeverity.CRITICAL,
        status=FindingStatus.EXPLICIT_CLAIM,
        matched_text="this text does not match offsets",
        explanation="Corrupt bounds candidate",
        start_pos=10,
        end_pos=25,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
    )

    with patch("app.services.detection.service.extract_semantic_candidates", return_value=[corrupt_candidate]):
        result = analyze_job_posting(raw)
        assert isinstance(result, DetectionResult)
        # The corrupt candidate must not be scorable
        assert all(it.explanation != "Corrupt bounds candidate" for it in result.evidence_items)


def test_pipeline_category_5_semantic_exception_falls_back_to_regex_only():
    """F-06 Cat 5: Unhandled exception in semantic engine logs error and cleanly completes with regex-only."""
    raw = "Candidates must pay a registration fee of Rs. 1000."
    with patch(
        "app.services.detection.service.tokenize_and_tag",
        side_effect=RuntimeError("Simulated unexpected semantic failure"),
    ):
        result = analyze_job_posting(raw)
        # Pipeline must complete with regex-only findings and zero raised exception
        assert isinstance(result, DetectionResult)
        assert result.rule_score == 65
        assert ScamCategory.ADVANCE_FEE in result.categories_detected


def test_pipeline_category_6_phase2_invariant_violation_raises_runtime_error():
    """F-06 Cat 6: A finding missing required model_fields_set reaches scoring -> raises RuntimeError."""
    raw = "Candidates must pay a registration fee of Rs. 1000."

    class InvariantViolatingItem:
        id = "VIOLATING_ITEM"
        category = ScamCategory.ADVANCE_FEE
        finding_id = "VIOLATING_1"
        model_fields_set = {"category"}  # Missing source, severity, status

    with patch(
        "app.services.detection.service.reconcile_contradictions",
        return_value=[InvariantViolatingItem()],
    ):
        with pytest.raises(RuntimeError, match="Phase 2 invariant violation"):
            analyze_job_posting(raw)


def test_pipeline_category_7_orphaned_item_in_reconciler_logged_and_rejected():
    """F-06 Cat 7: Orphaned unvalidated item reaching reconciler is logged as error and rejected."""
    raw = "Candidates must pay a registration fee of Rs. 1000."
    valid_item = EvidenceItem(
        id="ADV_FEE_REGISTRATION",
        category=ScamCategory.ADVANCE_FEE,
        source=DetectionSource.DETERMINISTIC_RULE,
        severity=FindingSeverity.CRITICAL,
        status=FindingStatus.EXPLICIT_CLAIM,
        matched_text="pay a registration fee of Rs. 1000",
        explanation="Valid item",
        finding_id="ADV_FEE_REGISTRATION_1",
        start_pos=16,
        end_pos=50,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
    )
    orphaned_item = EvidenceItem(
        id="ORPHANED_UNVALIDATED",
        category=ScamCategory.ADVANCE_FEE,
        source=DetectionSource.DETERMINISTIC_RULE,
        severity=FindingSeverity.CRITICAL,
        status=FindingStatus.EXPLICIT_CLAIM,
        matched_text="pay",
        explanation="Orphaned item",
        finding_id="ORPHANED_1",
        start_pos=16,
        end_pos=19,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
    )

    # Validated set contains only valid_item.finding_id
    validated_ids = {valid_item.finding_id}
    result = reconcile_detection_engines(
        regex_items=[valid_item, orphaned_item],
        semantic_items=[],
        raw_content=raw,
        validated_item_ids=validated_ids,
    )

    # Orphaned item is rejected and logged
    assert len(result.rejected_candidates) == 1
    assert result.rejected_candidates[0].finding_id == "ORPHANED_1"
    assert result.rejected_candidates[0].rejection_reason == "ORPHANED_UNVALIDATED_ITEM"

    # Valid item is retained
    assert len(result.reconciled_evidence) == 1
    assert result.reconciled_evidence[0].finding_id == valid_item.finding_id


# =============================================================================
# 3. PERFORMANCE & REDOS BENCHMARK TESTS (F-07)
# =============================================================================

def test_performance_benchmark_all_payload_sizes():
    """
    F-07 Reproducible Benchmark Protocol:
    Measures execution latency across four payload sizes:
      - Tiny: ~100 characters
      - Typical: ~500 characters
      - Large: ~2,000 characters
      - Maximum: ~10,000 characters
    Design target: <20 ms for typical payload on standard hardware.
    """
    tiny_payload = "Candidates must pay a registration fee of Rs. 500."
    typical_payload = (
        "We are looking for Data Entry Operators. Candidates must submit documents and pay "
        "a registration fee of Rs. 1000 before onboarding. Payment must be transferred directly "
        "to our company account. Beware of third-party agencies asking for additional fees. "
        "Our company does not authorize any consultancy to collect money."
    )
    large_payload = typical_payload * 4
    maximum_payload = typical_payload * 20

    payloads = [
        ("Tiny (~100 chars)", tiny_payload),
        ("Typical (~500 chars)", typical_payload),
        ("Large (~2000 chars)", large_payload),
        ("Maximum (~10000 chars)", maximum_payload),
    ]

    results = []
    for label, payload in payloads:
        # Warmup run
        analyze_job_posting(payload)

        # Benchmark 5 iterations
        iterations = 5
        start_t = time.perf_counter()
        for _ in range(iterations):
            analyze_job_posting(payload)
        end_t = time.perf_counter()

        avg_latency_ms = ((end_t - start_t) / iterations) * 1000.0
        results.append((label, avg_latency_ms))
        print(f"\n[BENCHMARK F-07] {label}: {avg_latency_ms:.2f} ms")

    # Assert typical payload completes comfortably (target <20ms, safety guardrail <150ms on CI/Windows)
    typical_latency = next(lat for lbl, lat in results if "Typical" in lbl)
    assert typical_latency < 150.0, f"Typical payload latency {typical_latency:.2f}ms exceeded guardrail"
