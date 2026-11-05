"""
Phase 3 — Multi-Engine Reconciliation Engine Unit & Integration Test Suite.
Verifies:
  1. The 10 Reconciliation Behavior Matrix combinations (Section 9 of implementation plan).
  2. The 7 Reconciliation Provenance Acceptance Criteria (Section 10.5 of specification).
  3. Five-Tier contradiction graph interactions (Stage 5 bypass for secondary items).
  4. Durable Rejected Candidate Provenance (F-04).
"""
import pytest

from app.services.detection.models import (
    CandidateEligibility,
    DetectionSource,
    EvidenceItem,
    FindingSeverity,
    FindingStatus,
    RejectionReason,
    ScamCategory,
)
from app.services.detection.quotation import reconcile_contradictions
from app.services.detection.reconciliation import (
    EngineConflictRecord,
    EngineCorroborationRecord,
    ReconciliationResult,
    RejectedCandidateRecord,
    reconcile_detection_engines,
)
from app.services.detection.scoring import calculate_evidence_score, get_effective_weight


def _make_evidence_item(
    item_id: str,
    category: ScamCategory = ScamCategory.ADVANCE_FEE,
    severity: FindingSeverity = FindingSeverity.CRITICAL,
    status: FindingStatus = FindingStatus.EXPLICIT_CLAIM,
    matched_text: str = "pay registration fee",
    start_pos: int = 10,
    end_pos: int = 30,
    finding_id: str | None = None,
    eligibility: CandidateEligibility = CandidateEligibility.SCORABLE_ACTIVE,
    explanation: str = "Mandatory fee payment directive.",
    context_required: str | None = None,
    rejection_reason: RejectionReason | None = None,
) -> EvidenceItem:
    f_id = finding_id or f"{item_id}_{start_pos}_{end_pos}"
    return EvidenceItem(
        id=item_id,
        category=category,
        source=DetectionSource.DETERMINISTIC_RULE,
        severity=severity,
        status=status,
        matched_text=matched_text,
        explanation=explanation,
        context_required=context_required,
        finding_id=f_id,
        contradicts_finding_ids=[],
        start_pos=start_pos,
        end_pos=end_pos,
        eligibility=eligibility,
        rejection_reason=rejection_reason,
    )


# =============================================================================
# 1. THE 10 RECONCILIATION BEHAVIOR MATRIX COMBINATIONS (Section 9)
# =============================================================================

def test_combination_1_both_active_same_occurrence():
    """Row 1: Regex Active (65) vs Semantic Active (65) -> Regex is Primary, Semantic wt=0."""
    raw = "Candidates must pay a registration fee of Rs. 1000 before onboarding."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=16, end_pos=50, severity=FindingSeverity.CRITICAL)
    s = _make_evidence_item("SEM_ADV_FEE_DIRECTIVE", start_pos=16, end_pos=50, severity=FindingSeverity.CRITICAL)

    result = reconcile_detection_engines([r], [s], raw)
    assert len(result.reconciled_evidence) == 2
    assert len(result.corroborations) == 1
    assert len(result.conflicts) == 0

    corrob = result.corroborations[0]
    assert corrob.primary_engine == "REGEX"
    assert corrob.secondary_engine == "SEMANTIC"
    assert corrob.primary_finding_id == r.finding_id
    assert corrob.secondary_finding_id == s.finding_id
    assert corrob.secondary_engine_original_qualification == CandidateEligibility.SCORABLE_ACTIVE
    assert corrob.scoring_weight_assigned == 0

    # Primary regex finding retains scorable active status
    emitted_r = next(item for item in result.reconciled_evidence if item.finding_id == r.finding_id)
    assert emitted_r.eligibility == CandidateEligibility.SCORABLE_ACTIVE
    assert get_effective_weight(emitted_r) == 65

    # Secondary semantic finding emitted as NON_SCORABLE_INFORMATIONAL
    emitted_s = next(item for item in result.reconciled_evidence if item.finding_id == s.finding_id)
    assert emitted_s.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert emitted_s.severity == FindingSeverity.CRITICAL  # F-03: severity preserved
    assert emitted_s.status == FindingStatus.EXPLICIT_CLAIM  # F-03: status preserved
    assert emitted_s.rejection_reason is None
    assert get_effective_weight(emitted_s) == 0

    # Total score contributes exactly once (65)
    score = calculate_evidence_score(result.reconciled_evidence)
    assert score == 65


def test_combination_2_regex_active_semantic_ambiguous():
    """Row 2: Regex Active (65) vs Semantic Ambiguous (10) -> Regex is Primary (65), Semantic wt=0."""
    raw = "Candidates must pay an onboarding charge to our placement firm."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=16, end_pos=40, severity=FindingSeverity.CRITICAL)
    s = _make_evidence_item(
        "SEM_ADV_FEE_AMBIGUOUS",
        start_pos=16,
        end_pos=40,
        severity=FindingSeverity.LOW,
        status=FindingStatus.AMBIGUOUS_SIGNAL,
        eligibility=CandidateEligibility.SCORABLE_AMBIGUOUS,
    )

    result = reconcile_detection_engines([r], [s], raw)
    assert len(result.reconciled_evidence) == 2
    assert len(result.corroborations) == 1

    corrob = result.corroborations[0]
    assert corrob.primary_engine == "REGEX"
    assert corrob.secondary_engine == "SEMANTIC"
    assert corrob.primary_engine_qualification == CandidateEligibility.SCORABLE_ACTIVE
    assert corrob.secondary_engine_original_qualification == CandidateEligibility.SCORABLE_AMBIGUOUS

    emitted_s = next(item for item in result.reconciled_evidence if item.finding_id == s.finding_id)
    assert emitted_s.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert get_effective_weight(emitted_s) == 0

    score = calculate_evidence_score(result.reconciled_evidence)
    assert score == 65


def test_combination_3_regex_ambiguous_semantic_active():
    """Row 3: Regex Ambiguous (10) vs Semantic Active (65) -> Semantic is Primary (65, Max-Weight Policy)."""
    raw = "Registration fee of Rs. 1000 must be paid immediately to proceed."
    r = _make_evidence_item(
        "ADV_FEE_AMBIGUOUS_MENTION",
        start_pos=0,
        end_pos=28,
        severity=FindingSeverity.LOW,
        status=FindingStatus.AMBIGUOUS_SIGNAL,
        eligibility=CandidateEligibility.SCORABLE_AMBIGUOUS,
    )
    s = _make_evidence_item(
        "SEM_ADV_FEE_DIRECTIVE",
        start_pos=0,
        end_pos=45,
        severity=FindingSeverity.CRITICAL,
        status=FindingStatus.EXPLICIT_CLAIM,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
    )

    result = reconcile_detection_engines([r], [s], raw)
    assert len(result.reconciled_evidence) == 2
    assert len(result.corroborations) == 1

    corrob = result.corroborations[0]
    assert corrob.primary_engine == "SEMANTIC"
    assert corrob.secondary_engine == "REGEX"
    assert corrob.primary_finding_id == s.finding_id
    assert corrob.secondary_finding_id == r.finding_id
    assert corrob.primary_engine_qualification == CandidateEligibility.SCORABLE_ACTIVE
    assert corrob.secondary_engine_original_qualification == CandidateEligibility.SCORABLE_AMBIGUOUS

    # Semantic is preserved as primary scorable (65)
    emitted_s = next(item for item in result.reconciled_evidence if item.finding_id == s.finding_id)
    assert emitted_s.eligibility == CandidateEligibility.SCORABLE_ACTIVE
    assert get_effective_weight(emitted_s) == 65

    # Regex is emitted as secondary informational (0)
    emitted_r = next(item for item in result.reconciled_evidence if item.finding_id == r.finding_id)
    assert emitted_r.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert get_effective_weight(emitted_r) == 0

    # Total score preserves stronger qualification (65, not 10)
    score = calculate_evidence_score(result.reconciled_evidence)
    assert score == 65


def test_combination_4_conflict_active_vs_warning():
    """Row 4: Regex Active (65) vs Semantic Warning (0) -> Conflict: Regex scorable, Semantic advisory wt=0."""
    raw = "Beware of fraudsters asking you to pay a registration fee."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=35, end_pos=58, severity=FindingSeverity.CRITICAL)
    s = _make_evidence_item(
        "ADVISORY_SCAM_WARNING",
        start_pos=0,
        end_pos=58,
        severity=FindingSeverity.INFORMATIONAL,
        status=FindingStatus.AMBIGUOUS_SIGNAL,
        eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
    )

    result = reconcile_detection_engines([r], [s], raw)
    assert len(result.reconciled_evidence) == 2
    assert len(result.corroborations) == 0
    assert len(result.conflicts) == 1

    conflict = result.conflicts[0]
    assert conflict.conflict_type == "ACTIVE_VS_WARNING"
    assert conflict.regex_qualification == CandidateEligibility.SCORABLE_ACTIVE
    assert conflict.semantic_qualification == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert conflict.resolution_action == "PRESERVE_BOTH_EMIT_ADVISORY"

    emitted_r = next(item for item in result.reconciled_evidence if item.finding_id == r.finding_id)
    assert emitted_r.eligibility == CandidateEligibility.SCORABLE_ACTIVE

    emitted_s = next(item for item in result.reconciled_evidence if item.finding_id == s.finding_id)
    assert emitted_s.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert f"conflicts_with:{r.finding_id}" in (emitted_s.context_required or "")

    # Regex scored; handed to Stage 5 contradiction resolution
    score = calculate_evidence_score(result.reconciled_evidence)
    assert score == 65


def test_combination_5_regex_active_semantic_rejected():
    """Row 5: Regex Active (65) vs Semantic Rejected (0) -> Regex is retained, Semantic in rejected_candidates."""
    raw = "Candidates must pay an onboarding fee."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=16, end_pos=37, severity=FindingSeverity.CRITICAL)
    s = _make_evidence_item(
        "SEM_ADV_FEE_DIRECTIVE",
        start_pos=16,
        end_pos=37,
        eligibility=CandidateEligibility.REJECTED,
        rejection_reason=RejectionReason.NON_FINANCIAL_ACTION,
    )

    result = reconcile_detection_engines([r], [s], raw)
    # Only the valid regex item is emitted in reconciled_evidence
    assert len(result.reconciled_evidence) == 1
    assert result.reconciled_evidence[0].finding_id == r.finding_id
    assert result.reconciled_evidence[0].eligibility == CandidateEligibility.SCORABLE_ACTIVE

    # Semantic candidate recorded in durable rejected_candidates
    assert len(result.rejected_candidates) == 1
    rej = result.rejected_candidates[0]
    assert rej.engine == "SEMANTIC"
    assert rej.finding_id == s.finding_id
    assert rej.original_eligibility == CandidateEligibility.REJECTED

    # Regex finding is NOT suppressed
    score = calculate_evidence_score(result.reconciled_evidence)
    assert score == 65


def test_combination_6_different_predicate_same_span():
    """Row 6: Active Fee (65) vs Active Doc (25) -> Different predicate: both retained independently."""
    raw = "Candidates must submit documents and pay a registration fee."
    # Regex fee finding
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=37, end_pos=60, severity=FindingSeverity.CRITICAL)
    # Semantic document collection finding
    s = _make_evidence_item(
        "DOC_COLLECTION_MANDATORY",
        category=ScamCategory.ADVANCE_FEE,
        start_pos=16,
        end_pos=45,  # Overlaps in span
        severity=FindingSeverity.MEDIUM,
        status=FindingStatus.EXPLICIT_CLAIM,
        matched_text="submit documents",
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
    )

    result = reconcile_detection_engines([r], [s], raw)
    assert len(result.reconciled_evidence) == 2
    assert len(result.corroborations) == 0
    assert len(result.conflicts) == 0

    # Both retain scorable eligibility independently
    emitted_r = next(item for item in result.reconciled_evidence if item.finding_id == r.finding_id)
    emitted_s = next(item for item in result.reconciled_evidence if item.finding_id == s.finding_id)
    assert emitted_r.eligibility == CandidateEligibility.SCORABLE_ACTIVE
    assert emitted_s.eligibility == CandidateEligibility.SCORABLE_ACTIVE


def test_combination_7_different_category_same_span():
    """Row 7: Active Fee (65) vs Active Credential (45) -> Different category: both retained independently."""
    raw = "Pay the fee and share your OTP to confirm."
    r = _make_evidence_item(
        "ADV_FEE_REGISTRATION",
        category=ScamCategory.ADVANCE_FEE,
        start_pos=0,
        end_pos=25,
        severity=FindingSeverity.CRITICAL,
    )
    s = _make_evidence_item(
        "CRED_HARVEST_AUTH_SECRETS",
        category=ScamCategory.CREDENTIAL_HARVESTING,
        start_pos=10,
        end_pos=35,
        severity=FindingSeverity.HIGH,
    )

    result = reconcile_detection_engines([r], [s], raw)
    assert len(result.reconciled_evidence) == 2
    assert len(result.corroborations) == 0
    assert len(result.conflicts) == 0

    # Compounded multi-category score (65 anchor + 10 residual boost = 75)
    score = calculate_evidence_score(result.reconciled_evidence)
    assert score == 75


def test_combination_8_no_span_overlap():
    """Row 8: Active Fee (65) vs Active Fee (65), no span overlap -> Disjoint: both retained independently."""
    raw = "Pay registration fee of Rs. 500. Later, deposit uniform charge of Rs. 2000."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=0, end_pos=31, severity=FindingSeverity.CRITICAL)
    s = _make_evidence_item("ADV_FEE_SECURITY_DEPOSIT", start_pos=40, end_pos=74, severity=FindingSeverity.CRITICAL)

    result = reconcile_detection_engines([r], [s], raw)
    assert len(result.reconciled_evidence) == 2
    assert len(result.corroborations) == 0
    assert len(result.conflicts) == 0

    # Same category, so anchored at 65 (not double counted since same category)
    score = calculate_evidence_score(result.reconciled_evidence)
    assert score == 65


def test_combination_9_semantic_only():
    """Row 9: Semantic Only -> Retained with original eligibility, full semantic weight."""
    raw = "Candidates must transfer payment to the agency."
    s = _make_evidence_item("SEM_ADV_FEE_DIRECTIVE", start_pos=16, end_pos=46, severity=FindingSeverity.CRITICAL)

    result = reconcile_detection_engines([], [s], raw)
    assert len(result.reconciled_evidence) == 1
    assert result.reconciled_evidence[0].finding_id == s.finding_id
    assert result.reconciled_evidence[0].eligibility == CandidateEligibility.SCORABLE_ACTIVE
    assert calculate_evidence_score(result.reconciled_evidence) == 65


def test_combination_10_regex_only():
    """Row 10: Regex Only -> Retained with original eligibility, full regex weight."""
    raw = "Candidates must pay a registration fee."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=16, end_pos=39, severity=FindingSeverity.CRITICAL)

    result = reconcile_detection_engines([r], [], raw)
    assert len(result.reconciled_evidence) == 1
    assert result.reconciled_evidence[0].finding_id == r.finding_id
    assert result.reconciled_evidence[0].eligibility == CandidateEligibility.SCORABLE_ACTIVE
    assert calculate_evidence_score(result.reconciled_evidence) == 65


# =============================================================================
# 2. THE 7 RECONCILIATION PROVENANCE ACCEPTANCE CRITERIA (Section 10.5)
# =============================================================================

def test_provenance_criterion_1_caller_owned_object_immutability():
    """Criterion 1: Inputs to reconcile_detection_engines() are NOT mutated in place."""
    raw = "Candidates must pay a registration fee."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=16, end_pos=39)
    s = _make_evidence_item("SEM_ADV_FEE_DIRECTIVE", start_pos=16, end_pos=39)

    orig_r_elig = r.eligibility
    orig_s_elig = s.eligibility

    result = reconcile_detection_engines([r], [s], raw)

    # Assert caller-owned objects were NOT mutated
    assert r.eligibility == orig_r_elig
    assert s.eligibility == orig_s_elig
    # Assert emitted objects are distinct instances
    assert result.reconciled_evidence[0] is not r
    assert result.reconciled_evidence[1] is not s


def test_provenance_criterion_2_capture_before_transformation_invariant():
    """Criterion 2: Original secondary qualification is captured before secondary item eligibility mutation."""
    raw = "Candidates must pay a registration fee."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=16, end_pos=39)
    s = _make_evidence_item("SEM_ADV_FEE_DIRECTIVE", start_pos=16, end_pos=39, eligibility=CandidateEligibility.SCORABLE_ACTIVE)

    result = reconcile_detection_engines([r], [s], raw)
    corrob = result.corroborations[0]

    # Captured before transformation in memory
    assert corrob.secondary_engine_original_qualification == CandidateEligibility.SCORABLE_ACTIVE
    emitted_s = next(it for it in result.reconciled_evidence if it.finding_id == s.finding_id)
    assert emitted_s.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL


def test_provenance_criterion_3_structured_record_consistency():
    """Criterion 3: EngineCorroborationRecord retains exact original semantic eligibility and weight=0."""
    raw = "Candidates must pay a registration fee."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=16, end_pos=39)
    s = _make_evidence_item("SEM_ADV_FEE_DIRECTIVE", start_pos=16, end_pos=39, eligibility=CandidateEligibility.SCORABLE_ACTIVE)

    result = reconcile_detection_engines([r], [s], raw)
    rec = result.corroborations[0]

    assert rec.primary_finding_id == r.finding_id
    assert rec.secondary_finding_id == s.finding_id
    assert rec.primary_engine_qualification == CandidateEligibility.SCORABLE_ACTIVE
    assert rec.secondary_engine_original_qualification == CandidateEligibility.SCORABLE_ACTIVE
    assert rec.scoring_weight_assigned == 0


def test_provenance_criterion_4_no_independent_finding_suppression():
    """Criterion 4: Span overlap alone never suppresses or merges an independent finding on different predicate."""
    raw = "Candidates must submit documents and pay a registration fee."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=16, end_pos=60)
    s = _make_evidence_item("DOC_COLLECTION_MANDATORY", start_pos=16, end_pos=40, matched_text="submit documents")

    result = reconcile_detection_engines([r], [s], raw)
    assert len(result.corroborations) == 0
    assert len(result.reconciled_evidence) == 2
    assert all(it.eligibility == CandidateEligibility.SCORABLE_ACTIVE for it in result.reconciled_evidence)


def test_provenance_criterion_5_score_correctness_anti_double_counting():
    """Criterion 5: Corroborated occurrence contributes exactly one score (65, not 130)."""
    raw = "Candidates must pay a registration fee."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=16, end_pos=39, severity=FindingSeverity.CRITICAL)
    s = _make_evidence_item("SEM_ADV_FEE_DIRECTIVE", start_pos=16, end_pos=39, severity=FindingSeverity.CRITICAL)

    result = reconcile_detection_engines([r], [s], raw)
    score = calculate_evidence_score(result.reconciled_evidence)
    assert score == 65


def test_provenance_criterion_6_conflict_preservation_invariant():
    """Criterion 6: Semantic warning never silently overwrites or deletes regex finding."""
    raw = "Beware of fraudsters asking you to pay a registration fee."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=35, end_pos=58, severity=FindingSeverity.CRITICAL)
    s = _make_evidence_item(
        "ADVISORY_SCAM_WARNING",
        start_pos=0,
        end_pos=58,
        severity=FindingSeverity.INFORMATIONAL,
        eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
    )

    result = reconcile_detection_engines([r], [s], raw)
    assert len(result.conflicts) == 1
    assert any(it.finding_id == r.finding_id and it.eligibility == CandidateEligibility.SCORABLE_ACTIVE for it in result.reconciled_evidence)
    assert any(it.finding_id == s.finding_id and it.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL for it in result.reconciled_evidence)


def test_provenance_criterion_7_metadata_consistency():
    """Criterion 7: Secondary EvidenceItem context_required and explanation match corroboration record."""
    raw = "Candidates must pay a registration fee."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=16, end_pos=39)
    s = _make_evidence_item("SEM_ADV_FEE_DIRECTIVE", start_pos=16, end_pos=39)

    result = reconcile_detection_engines([r], [s], raw)
    emitted_s = next(it for it in result.reconciled_evidence if it.finding_id == s.finding_id)

    expected_context = f"corroborated_with:{r.finding_id};original_eligibility:SCORABLE_ACTIVE"
    assert emitted_s.context_required == expected_context
    assert f"corroborated_with:{r.finding_id}" in emitted_s.context_required
    assert "scoring weight suppressed to prevent double-counting" in emitted_s.explanation


# =============================================================================
# 3. FIVE-TIER CONTRADICTION GRAPH INTERACTION TESTS (Stage 5 Bypass)
# =============================================================================

def test_secondary_informational_item_bypasses_contradiction_graph():
    """Secondary informational items (NON_SCORABLE_INFORMATIONAL) must bypass Stage 5 contradiction links."""
    raw = (
        "Our company does not charge any registration fees for recruitment. "
        "However, our firm requires candidates to pay a registration fee of Rs. 1000."
    )
    # Company disclaimer finding
    disc = _make_evidence_item(
        "DISCLAIMER_COMPANY_NO_FEE",
        start_pos=0,
        end_pos=66,
        severity=FindingSeverity.INFORMATIONAL,
        status=FindingStatus.AMBIGUOUS_SIGNAL,
        matched_text="Our company does not charge any registration fees",
        eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
    )
    # Active fee demand (Regex)
    r = _make_evidence_item(
        "ADV_FEE_REGISTRATION",
        start_pos=107,
        end_pos=133,
        severity=FindingSeverity.CRITICAL,
        matched_text="pay a registration fee",
    )
    # Active fee demand (Semantic)
    s = _make_evidence_item(
        "SEM_ADV_FEE_DIRECTIVE",
        start_pos=107,
        end_pos=133,
        severity=FindingSeverity.CRITICAL,
        matched_text="pay a registration fee",
    )

    reconciled = reconcile_detection_engines([r], [s, disc], raw)
    # Stage 5 Contradiction Resolution
    stage5_items = reconcile_contradictions(reconciled.reconciled_evidence, raw)

    # Primary regex finding is linked to disclaimer (Tier 1 contradiction)
    stage5_r = next(it for it in stage5_items if it.finding_id == r.finding_id)
    assert disc.finding_id in stage5_r.contradicts_finding_ids

    # Secondary informational finding is NOT falsely linked (bypasses Stage 5 edge generation)
    stage5_s = next(it for it in stage5_items if it.finding_id == s.finding_id)
    assert disc.finding_id not in stage5_s.contradicts_finding_ids


def test_disclaimer_preserves_immune_patterns_through_reconciliation():
    """Immune patterns (e.g. ADV_FEE_PAYMENT_RAIL_DIRECT) are never contradicted even after reconciliation."""
    raw = (
        "Our company does not charge any recruitment fee. "
        "Candidates must pay Rs. 500 directly to test@oksbi."
    )
    disc = _make_evidence_item(
        "DISCLAIMER_COMPANY_NO_FEE",
        start_pos=0,
        end_pos=47,
        severity=FindingSeverity.INFORMATIONAL,
        matched_text="Our company does not charge any recruitment fee",
        eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
    )
    r = _make_evidence_item(
        "ADV_FEE_PAYMENT_RAIL_DIRECT",
        start_pos=65,
        end_pos=99,
        severity=FindingSeverity.CRITICAL,
        matched_text="pay Rs. 500 directly to test@oksbi",
    )
    s = _make_evidence_item(
        "SEM_ADV_FEE_DIRECTIVE",
        start_pos=65,
        end_pos=99,
        severity=FindingSeverity.CRITICAL,
        matched_text="pay Rs. 500 directly to test@oksbi",
    )

    reconciled = reconcile_detection_engines([r], [s, disc], raw)
    stage5_items = reconcile_contradictions(reconciled.reconciled_evidence, raw)

    stage5_r = next(it for it in stage5_items if it.finding_id == r.finding_id)
    # Payment rail direct is in DISCLAIMER_IMMUNE_IDS; must NOT have contradiction edge
    assert disc.finding_id not in stage5_r.contradicts_finding_ids


# =============================================================================
# 4. DURABLE REJECTED CANDIDATE PROVENANCE (F-04)
# =============================================================================

def test_rejected_candidate_does_not_suppress_valid_regex():
    """A rejected semantic candidate does not suppress a valid regex finding for the same span."""
    raw = "Candidates must pay an onboarding fee."
    r = _make_evidence_item("ADV_FEE_REGISTRATION", start_pos=16, end_pos=37)
    s = _make_evidence_item(
        "SEM_ADV_FEE_DIRECTIVE",
        start_pos=16,
        end_pos=37,
        eligibility=CandidateEligibility.REJECTED,
        rejection_reason=RejectionReason.NON_FINANCIAL_ACTION,
    )

    result = reconcile_detection_engines([r], [s], raw)
    assert len(result.reconciled_evidence) == 1
    assert result.reconciled_evidence[0].finding_id == r.finding_id
    assert result.reconciled_evidence[0].eligibility == CandidateEligibility.SCORABLE_ACTIVE


def test_rejected_candidates_list_non_empty_when_rejected():
    """The rejected_candidates list is non-empty when a candidate is rejected."""
    raw = "Candidates must pay an onboarding fee."
    s = _make_evidence_item(
        "SEM_ADV_FEE_DIRECTIVE",
        start_pos=16,
        end_pos=37,
        eligibility=CandidateEligibility.REJECTED,
        rejection_reason=RejectionReason.DISCRETIONARY_NOT_MANDATORY,
    )

    result = reconcile_detection_engines([], [s], raw)
    assert len(result.rejected_candidates) == 1
    rej = result.rejected_candidates[0]
    assert rej.engine == "SEMANTIC"
    assert rej.rejection_reason == RejectionReason.DISCRETIONARY_NOT_MANDATORY.value
    assert rej.original_eligibility == CandidateEligibility.REJECTED
    assert rej.occurrence_span == (16, 37)


def test_rejected_candidate_provenance_recoverable_without_logs():
    """Provenance from RejectedCandidateRecord is recoverable independently of log output."""
    raw = "Candidates must pay an onboarding fee."
    s = _make_evidence_item(
        "SEM_ADV_FEE_DIRECTIVE",
        start_pos=16,
        end_pos=37,
        eligibility=CandidateEligibility.REJECTED,
        rejection_reason=RejectionReason.NO_RECRUITMENT_CONTEXT,
    )

    result = reconcile_detection_engines([], [s], raw)
    assert len(result.rejected_candidates) == 1
    record = result.rejected_candidates[0]

    # Directly inspectable structured attributes without log capture
    assert record.engine == "SEMANTIC"
    assert record.finding_id == s.finding_id
    assert record.occurrence_span == (16, 37)
    assert record.rejection_reason == "NO_RECRUITMENT_CONTEXT"
    assert record.original_eligibility == CandidateEligibility.REJECTED
    assert record.rejection_context == "semantic_qualification"
