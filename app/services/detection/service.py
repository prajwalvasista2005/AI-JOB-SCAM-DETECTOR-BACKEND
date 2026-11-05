"""
Canonical Detection Subsystem Orchestrator (Step 3.7).
Implements analyze_job_posting() executing the 6-stage canonical pipeline:
  Stage 1: Regex Pattern Matching (patterns.py)
  Stage 2: Semantic Qualification Engine (semantic.py)
  Stage 3: Unified Quotation & Span Validation (quotation.py)
  Stage 4: Multi-Engine Reconciliation (reconciliation.py)
  Stage 5: Five-Tier Contradiction Graph (quotation.py)
  Stage 6: Multi-Category Scoring & Provenance Gate (scoring.py)

Follows Specification v1.6.0-DRAFT Section 2, Section 10 and Implementation Plan v1.1.0 Section 5.
"""
import logging
from typing import Any

from app.services.detection.models import (
    CandidateEligibility,
    ContentSecurityWarning,
    DetectionResult,
    DetectionSource,
    EvidenceCompleteness,
    EvidenceItem,
    FindingSeverity,
    FindingStatus,
    RejectionReason,
    ScamCategory,
)
from app.services.detection.patterns import (
    COMPANY_FEE_DISCLAIMER_RE,
    DETECTION_PATTERNS,
)
from app.services.detection.quotation import (
    DocumentContext,
    reconcile_contradictions,
    validate_evidence_quotation,
)
from app.services.detection.reconciliation import (
    EngineConflictRecord,
    EngineCorroborationRecord,
    ReconciliationResult,
    RejectedCandidateRecord,
    reconcile_detection_engines,
)
from app.services.detection.scoring import (
    calculate_evidence_score,
    get_effective_weight,
    get_verdict,
)
from app.services.detection.semantic import (
    SemanticEvidenceAdapter,
    associate_tokens_scoped,
    evaluate_recruitment_context_gating,
    qualify_semantic_association,
    tokenize_and_tag,
)

logger = logging.getLogger(__name__)


def extract_regex_candidates(raw_content: str, max_per_pattern: int = 50, max_total: int = 200) -> list[EvidenceItem]:
    """Stage 1: Generates candidate EvidenceItem instances from frozen regex patterns."""
    candidates: list[EvidenceItem] = []
    pattern_counts: dict[str, int] = {}
    total_count = 0

    # 1. Standard detection patterns
    for p in DETECTION_PATTERNS:
        if total_count >= max_total:
            break
        for m in p.regex.finditer(raw_content):
            if total_count >= max_total:
                break
            count = pattern_counts.get(p.pattern_id, 0) + 1
            if count > max_per_pattern:
                continue

            item = p.to_evidence_item(
                matched_text=m.group(0),
                finding_id=f"{p.pattern_id}_{count}",
                start_pos=m.start(),
                end_pos=m.end(),
            )
            candidates.append(item)
            pattern_counts[p.pattern_id] = count
            total_count += 1

    # 2. Company-wide fee disclaimers
    disc_count = 0
    for m in COMPANY_FEE_DISCLAIMER_RE.finditer(raw_content):
        if total_count >= max_total:
            break
        disc_count += 1
        if disc_count > max_per_pattern:
            continue

        item = EvidenceItem(
            id="DISCLAIMER_COMPANY_NO_FEE",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.INFORMATIONAL,
            status=FindingStatus.AMBIGUOUS_SIGNAL,
            matched_text=m.group(0),
            explanation="Company-wide disclaimer explicitly stating recruitment is free and no fees are charged.",
            finding_id=f"DISCLAIMER_COMPANY_NO_FEE_{disc_count}",
            start_pos=m.start(),
            end_pos=m.end(),
            eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
        )
        candidates.append(item)
        total_count += 1

    return candidates


def extract_semantic_candidates(raw_content: str, max_per_pattern: int = 50, max_total: int = 200) -> list[EvidenceItem]:
    """Stage 2: Generates candidate EvidenceItem instances via internal 7-stage semantic pipeline."""
    try:
        tokens = tokenize_and_tag(raw_content)
        if not tokens:
            return []

        associations = associate_tokens_scoped(tokens, max_window=10)
        if not associations:
            return []

        candidates: list[EvidenceItem] = []
        pattern_counts: dict[str, int] = {}
        total_count = 0

        for assoc in associations:
            if total_count >= max_total:
                break

            has_recruitment_context = evaluate_recruitment_context_gating(assoc, raw_content)
            decision = qualify_semantic_association(assoc, has_recruitment_context=has_recruitment_context)
            p_id = decision.pattern_id or "SEM_GENERIC"
            if pattern_counts.get(p_id, 0) >= max_per_pattern:
                continue

            item = SemanticEvidenceAdapter.adapt(decision, raw_content)
            if item is not None:
                candidates.append(item)
                pattern_counts[p_id] = pattern_counts.get(p_id, 0) + 1
                total_count += 1

        return candidates
    except Exception as e:
        logger.exception("Semantic engine unexpected exception fail-closed; falling back to regex-only: %s", e)
        return []


def analyze_job_posting(raw_content: str) -> DetectionResult:
    """
    Canonical detection subsystem orchestrator.
    Executes the 6-stage detection pipeline:
      Stage 1: Regex Pattern Matching (patterns.py)
      Stage 2: Semantic Qualification Engine (semantic.py)
      Stage 3: Unified Quotation Validation (quotation.py: validate_evidence_quotation)
      Stage 4: Multi-Engine Reconciliation (reconciliation.py: reconcile_detection_engines)
      Stage 5: Contradiction Resolution Graph (quotation.py: reconcile_contradictions)
      Stage 6: Scoring Gate (scoring.py: calculate_evidence_score)

    Raises:
      ValueError: if raw_content is None, not a string, or whitespace-only (F-06 Category 3).
      RuntimeError: if a Phase 2 invariant violation is detected (F-06 Category 6).
    """
    # F-06 Category 3: Validate input text entrypoint
    if raw_content is None or not isinstance(raw_content, str) or not raw_content.strip():
        raise ValueError("raw_content must be a non-empty string")

    # OPT-01 & OPT-02: Instantiate per-request DocumentContext
    doc_ctx = DocumentContext(raw_content)

    # Stage 1: Deterministic Regex Matching
    raw_regex_items = extract_regex_candidates(raw_content)

    # Stage 2: Semantic Qualification (Exception-isolated with regex fallback)
    raw_semantic_items = extract_semantic_candidates(raw_content)

    # Stage 3: Unified Quotation Validation
    validated_regex: list[EvidenceItem] = []
    validated_semantic: list[EvidenceItem] = []
    validated_ids: set[str] = set()
    rejected_records: list[RejectedCandidateRecord] = []

    for item in raw_regex_items:
        if not isinstance(item, EvidenceItem):
            rejected_records.append(
                RejectedCandidateRecord(
                    engine="REGEX",
                    finding_id=getattr(item, "finding_id", None),
                    occurrence_span=None,
                    rejection_reason="INVALID_CANDIDATE_OBJECT",
                    original_eligibility=getattr(item, "eligibility", None),
                    rejection_context="candidate_type_check",
                )
            )
            continue

        val_item = validate_evidence_quotation(item, raw_content, doc_ctx=doc_ctx)
        # Quotation validation failed check
        if val_item.start_pos is None or val_item.end_pos is None:
            rejected_records.append(
                RejectedCandidateRecord(
                    engine="REGEX",
                    finding_id=item.finding_id,
                    occurrence_span=(item.start_pos, item.end_pos) if item.start_pos is not None and item.end_pos is not None else None,
                    rejection_reason="QUOTATION_VALIDATION_FAILED",
                    original_eligibility=item.eligibility,
                    rejection_context="quotation_validation",
                )
            )
        else:
            validated_regex.append(val_item)
            validated_ids.add(val_item.finding_id or val_item.id)

    for item in raw_semantic_items:
        if not isinstance(item, EvidenceItem):
            rejected_records.append(
                RejectedCandidateRecord(
                    engine="SEMANTIC",
                    finding_id=getattr(item, "finding_id", None),
                    occurrence_span=None,
                    rejection_reason="INVALID_CANDIDATE_OBJECT",
                    original_eligibility=getattr(item, "eligibility", None),
                    rejection_context="candidate_type_check",
                )
            )
            continue

        val_item = validate_evidence_quotation(item, raw_content, doc_ctx=doc_ctx)
        if val_item.start_pos is None or val_item.end_pos is None:
            rejected_records.append(
                RejectedCandidateRecord(
                    engine="SEMANTIC",
                    finding_id=item.finding_id,
                    occurrence_span=(item.start_pos, item.end_pos) if item.start_pos is not None and item.end_pos is not None else None,
                    rejection_reason="QUOTATION_VALIDATION_FAILED",
                    original_eligibility=item.eligibility,
                    rejection_context="quotation_validation",
                )
            )
        else:
            validated_semantic.append(val_item)
            validated_ids.add(val_item.finding_id or val_item.id)

    # Stage 4: Multi-Engine Reconciliation
    reconciliation_result = reconcile_detection_engines(
        regex_items=validated_regex,
        semantic_items=validated_semantic,
        raw_content=raw_content,
        rejected_candidates=rejected_records,
        validated_item_ids=validated_ids,
    )

    # Stage 5: Contradiction Resolution Graph
    reconciled_items = reconciliation_result.reconciled_evidence
    contradiction_resolved = reconcile_contradictions(reconciled_items, raw_content, doc_ctx=doc_ctx)

    # Stage 6: Scoring Gate & Phase 2 Invariant Verification (F-06 Category 6)
    required_provenance_fields = {"category", "source", "severity", "status"}
    for it in contradiction_resolved:
        fields_set = getattr(it, "model_fields_set", set())
        if not required_provenance_fields.issubset(fields_set):
            missing = required_provenance_fields - fields_set
            raise RuntimeError(
                f"Phase 2 invariant violation: item {it.finding_id or it.id} missing required provenance fields {missing}"
            )

    risk_score = calculate_evidence_score(contradiction_resolved)
    verdict = get_verdict(risk_score)

    # Categories detected
    categories_detected: list[ScamCategory] = []
    seen_cats: set[ScamCategory] = set()
    for it in contradiction_resolved:
        if it.category != ScamCategory.UNKNOWN and it.category not in seen_cats:
            if get_effective_weight(it) > 0 or it.eligibility in (
                CandidateEligibility.SCORABLE_ACTIVE,
                CandidateEligibility.SCORABLE_AMBIGUOUS,
            ):
                seen_cats.add(it.category)
                categories_detected.append(it.category)

    explanation = f"Deterministic rule engine scored {risk_score} points ({verdict})."
    completeness = EvidenceCompleteness.HIGH if contradiction_resolved else EvidenceCompleteness.MODERATE

    return DetectionResult(
        rule_score=risk_score,
        categories_detected=categories_detected,
        evidence_items=contradiction_resolved,
        fallback_explanation=explanation,
        evidence_completeness=completeness,
        content_security=ContentSecurityWarning(),
    )
