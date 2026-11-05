"""
Multi-Engine Reconciliation Engine (Step 3.6).
Reconciles quotation-validated findings from regex and semantic engines conservatively.
Follows Specification v1.6.0-DRAFT Section 10 and Implementation Plan v1.1.0 Section 9.

Core Guarantees:
  1. Operates strictly on quotation-validated input items.
  2. Caller-owned object immutability (deep-copy inputs; no in-place mutation).
  3. Strict occurrence identity: span overlap AND same predicate AND compatible category.
  4. Max-Effective-Weight Primary Selection Policy (preserves stronger qualification).
  5. Anti-double-counting scoring workaround (secondary copy emitted as NON_SCORABLE_INFORMATIONAL, wt=0).
  6. Lossless audit trail in EngineCorroborationRecord and EngineConflictRecord.
  7. Durable rejected candidate provenance in RejectedCandidateRecord (F-04).
"""
import copy
from dataclasses import dataclass, field
import logging
from typing import Any

from app.services.detection.models import (
    CandidateEligibility,
    DetectionSource,
    EvidenceItem,
    FindingSeverity,
    FindingStatus,
    RejectionReason,
    ScamCategory,
)
from app.services.detection.scoring import get_effective_weight

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineCorroborationRecord:
    primary_finding_id: str
    secondary_finding_id: str
    occurrence_span: tuple[int, int]
    primary_engine: str  # "REGEX" or "SEMANTIC"
    secondary_engine: str  # "SEMANTIC" or "REGEX"
    primary_engine_qualification: CandidateEligibility
    secondary_engine_original_qualification: CandidateEligibility
    resolution_action: str = "CORROBORATED_DUAL_RETENTION_SCORING_WORKAROUND"
    scoring_weight_assigned: int = 0


@dataclass(frozen=True)
class EngineConflictRecord:
    regex_finding_id: str
    semantic_finding_id: str
    occurrence_span: tuple[int, int]
    regex_qualification: CandidateEligibility
    semantic_qualification: CandidateEligibility
    conflict_type: str  # e.g. "ACTIVE_VS_WARNING", "ACTIVE_DIRECTIVE_VS_SCAM_WARNING"
    resolution_action: str = "PRESERVE_BOTH_EMIT_ADVISORY"


@dataclass(frozen=True)
class RejectedCandidateRecord:
    engine: str  # "SEMANTIC" or "REGEX"
    finding_id: str | None
    occurrence_span: tuple[int, int] | None
    rejection_reason: str
    original_eligibility: CandidateEligibility | None
    rejection_context: str  # e.g. "quotation_validation", "span_bounds_check", "semantic_qualification"


@dataclass
class ReconciliationResult:
    reconciled_evidence: list[EvidenceItem]
    corroborations: list[EngineCorroborationRecord]
    conflicts: list[EngineConflictRecord]
    rejected_candidates: list[RejectedCandidateRecord] = field(default_factory=list)


def _spans_overlap(item1: EvidenceItem, item2: EvidenceItem) -> bool:
    """Returns True iff item1 and item2 character spans overlap by > 0 characters."""
    if item1.start_pos is None or item1.end_pos is None or item2.start_pos is None or item2.end_pos is None:
        return False
    return max(item1.start_pos, item2.start_pos) < min(item1.end_pos, item2.end_pos)


def _compatible_categories(item1: EvidenceItem, item2: EvidenceItem) -> bool:
    """Returns True iff item1 and item2 belong to compatible scam categories."""
    if item1.category == ScamCategory.UNKNOWN or item2.category == ScamCategory.UNKNOWN:
        return False
    return item1.category == item2.category


def _extract_predicate_family(item: EvidenceItem) -> str:
    """Categorizes finding into predicate family to ensure same-predicate matching."""
    item_id = (item.id or "").upper()
    text = (item.matched_text or "").lower()

    if "DOC" in item_id or any(w in text for w in ("submit document", "upload id", "id proof", "submit docs")):
        return "PRED_DOC"
    if "CRED" in item_id or "OTP" in item_id or "PIN" in item_id or any(w in text for w in ("upi pin", "bank otp", "sms otp")):
        return "PRED_CRED"
    if "REV_AUTH" in item_id or "REVERSE_AUTH" in item_id or "collect request" in text:
        return "PRED_REV_AUTH"
    if "EQUIP" in item_id or "laptop" in text or "vendor" in text:
        return "PRED_EQUIP"
    if "COMM" in item_id or "telegram" in text or "whatsapp" in text:
        return "PRED_COMM"
    if "SALARY" in item_id or "unrealistic" in text:
        return "PRED_SALARY"
    if item.category == ScamCategory.ADVANCE_FEE or "FEE" in item_id or "PAYMENT" in item_id or "DEPOSIT" in item_id:
        return "PRED_FEE"

    return item.category.value


def _same_predicate(item1: EvidenceItem, item2: EvidenceItem) -> bool:
    """Returns True iff item1 and item2 refer to the identical predicate/action family."""
    return _extract_predicate_family(item1) == _extract_predicate_family(item2)


def _is_warning_or_disclaimer(item: EvidenceItem) -> bool:
    """Returns True if finding represents an informational advisory, warning, or disclaimer."""
    item_id = (item.id or "").upper()
    if item_id.startswith("DISCLAIMER_") or item_id.startswith("ADVISORY_"):
        return True
    if item.severity == FindingSeverity.INFORMATIONAL:
        return True
    if item.eligibility in (
        CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
        CandidateEligibility.NON_SCORABLE_CONTEXTUAL,
    ):
        return True
    return False


def _copy_evidence_item(
    item: EvidenceItem,
    *,
    eligibility: CandidateEligibility | None = None,
    explanation: str | None = None,
    context_required: str | None = None,
    rejection_reason: RejectionReason | None = None,
) -> EvidenceItem:
    """
    Creates a deep-copy EvidenceItem with optional overriding fields,
    ensuring 100% caller-owned object immutability and complete adherence to F-03.
    """
    return EvidenceItem(
        id=item.id,
        category=item.category,
        source=item.source,
        severity=item.severity,
        status=item.status,
        matched_text=item.matched_text,
        explanation=explanation if explanation is not None else item.explanation,
        context_required=context_required if context_required is not None else item.context_required,
        finding_id=item.finding_id,
        contradicts_finding_ids=list(item.contradicts_finding_ids),
        start_pos=item.start_pos,
        end_pos=item.end_pos,
        eligibility=eligibility if eligibility is not None else item.eligibility,
        rejection_reason=rejection_reason if rejection_reason is not None else item.rejection_reason,
    )


def reconcile_detection_engines(
    regex_items: list[EvidenceItem],
    semantic_items: list[EvidenceItem],
    raw_content: str,
    rejected_candidates: list[RejectedCandidateRecord] | None = None,
    validated_item_ids: set[str] | None = None,
) -> ReconciliationResult:
    """
    Standardized Multi-Engine Reconciliation Interface (Step 3.6).
    Reconciles quotation-validated findings from regex and semantic engines conservatively.

    Guarantees:
      - Operates strictly on quotation-validated input items.
      - Caller-owned object immutability (deep copy inputs; zero mutation of arguments).
      - Same-occurrence evaluation (span overlap + same predicate + compatible category).
      - Max-weight primary selection (preserves stronger qualification).
      - Anti-double-counting (secondary item emitted with NON_SCORABLE_INFORMATIONAL, wt=0).
      - Lossless audit trail in corroborations and conflicts.
      - Durable rejected candidate provenance in RejectedCandidateRecord (F-04).
    """
    reconciled_evidence: list[EvidenceItem] = []
    corroborations: list[EngineCorroborationRecord] = []
    conflicts: list[EngineConflictRecord] = []
    durable_rejected: list[RejectedCandidateRecord] = list(rejected_candidates or [])

    try:
        # 1. Input validation & defensive fail-closed checks
        if not isinstance(raw_content, str):
            logger.warning("reconcile_detection_engines fail-closed: raw_content is not str")
            return ReconciliationResult(
                reconciled_evidence=[],
                corroborations=[],
                conflicts=[],
                rejected_candidates=durable_rejected,
            )

        safe_regex_items = list(regex_items) if isinstance(regex_items, list) else []
        safe_semantic_items = list(semantic_items) if isinstance(semantic_items, list) else []

        # 2. Partition inputs: separate rejected candidates from valid scorable/informational items
        valid_regex: list[EvidenceItem] = []
        for r in safe_regex_items:
            if not isinstance(r, EvidenceItem):
                durable_rejected.append(
                    RejectedCandidateRecord(
                        engine="REGEX",
                        finding_id=getattr(r, "finding_id", None),
                        occurrence_span=None,
                        rejection_reason="INVALID_CANDIDATE_OBJECT",
                        original_eligibility=getattr(r, "eligibility", None),
                        rejection_context="input_type_check",
                    )
                )
                continue

            # Check for orphaned unvalidated items (F-06 Category 7)
            item_key = r.finding_id or r.id
            is_orphaned = validated_item_ids is not None and item_key not in validated_item_ids
            has_invalid_bounds = (
                r.start_pos is None
                or r.end_pos is None
                or not (0 <= r.start_pos <= r.end_pos <= len(raw_content))
            )
            if is_orphaned or has_invalid_bounds:
                logger.error("Orphaned or unvalidated item reached reconciler: %s", item_key)
                durable_rejected.append(
                    RejectedCandidateRecord(
                        engine="REGEX",
                        finding_id=r.finding_id,
                        occurrence_span=(r.start_pos, r.end_pos) if r.start_pos is not None and r.end_pos is not None else None,
                        rejection_reason="ORPHANED_UNVALIDATED_ITEM" if is_orphaned else "INVALID_SPAN_BOUNDS",
                        original_eligibility=r.eligibility,
                        rejection_context="reconciler_validation_check",
                    )
                )
                continue

            if r.eligibility == CandidateEligibility.REJECTED:
                reason_str = r.rejection_reason.value if hasattr(r.rejection_reason, "value") else str(r.rejection_reason or "REJECTED")
                durable_rejected.append(
                    RejectedCandidateRecord(
                        engine="REGEX",
                        finding_id=r.finding_id,
                        occurrence_span=(r.start_pos, r.end_pos),
                        rejection_reason=reason_str,
                        original_eligibility=r.eligibility,
                        rejection_context="regex_qualification",
                    )
                )
            else:
                valid_regex.append(r)

        valid_semantic: list[EvidenceItem] = []
        for s in safe_semantic_items:
            if not isinstance(s, EvidenceItem):
                durable_rejected.append(
                    RejectedCandidateRecord(
                        engine="SEMANTIC",
                        finding_id=getattr(s, "finding_id", None),
                        occurrence_span=None,
                        rejection_reason="INVALID_CANDIDATE_OBJECT",
                        original_eligibility=getattr(s, "eligibility", None),
                        rejection_context="input_type_check",
                    )
                )
                continue

            # Check for orphaned unvalidated items (F-06 Category 7)
            item_key = s.finding_id or s.id
            is_orphaned = validated_item_ids is not None and item_key not in validated_item_ids
            has_invalid_bounds = (
                s.start_pos is None
                or s.end_pos is None
                or not (0 <= s.start_pos <= s.end_pos <= len(raw_content))
            )
            if is_orphaned or has_invalid_bounds:
                logger.error("Orphaned or unvalidated item reached reconciler: %s", item_key)
                durable_rejected.append(
                    RejectedCandidateRecord(
                        engine="SEMANTIC",
                        finding_id=s.finding_id,
                        occurrence_span=(s.start_pos, s.end_pos) if s.start_pos is not None and s.end_pos is not None else None,
                        rejection_reason="ORPHANED_UNVALIDATED_ITEM" if is_orphaned else "INVALID_SPAN_BOUNDS",
                        original_eligibility=s.eligibility,
                        rejection_context="reconciler_validation_check",
                    )
                )
                continue

            if s.eligibility == CandidateEligibility.REJECTED:
                reason_str = s.rejection_reason.value if hasattr(s.rejection_reason, "value") else str(s.rejection_reason or "REJECTED")
                durable_rejected.append(
                    RejectedCandidateRecord(
                        engine="SEMANTIC",
                        finding_id=s.finding_id,
                        occurrence_span=(s.start_pos, s.end_pos),
                        rejection_reason=reason_str,
                        original_eligibility=s.eligibility,
                        rejection_context="semantic_qualification",
                    )
                )
            else:
                valid_semantic.append(s)

        # 3. Same-occurrence matching between valid_regex and valid_semantic
        matched_semantic_indices: set[int] = set()
        matched_regex_indices: set[int] = set()

        for r_idx, r in enumerate(valid_regex):
            best_s_idx: int | None = None
            best_overlap_len = 0

            for s_idx, s in enumerate(valid_semantic):
                if s_idx in matched_semantic_indices:
                    continue

                if _spans_overlap(r, s) and _compatible_categories(r, s) and _same_predicate(r, s):
                    overlap_len = min(r.end_pos, s.end_pos) - max(r.start_pos, s.start_pos)
                    if overlap_len > best_overlap_len:
                        best_overlap_len = overlap_len
                        best_s_idx = s_idx

            if best_s_idx is not None:
                s = valid_semantic[best_s_idx]
                matched_regex_indices.add(r_idx)
                matched_semantic_indices.add(best_s_idx)

                span = (max(r.start_pos, s.start_pos), min(r.end_pos, s.end_pos))

                # Check for Conflict vs Corroboration
                r_is_active = r.eligibility in (CandidateEligibility.SCORABLE_ACTIVE, CandidateEligibility.SCORABLE_AMBIGUOUS)
                s_is_active = s.eligibility in (CandidateEligibility.SCORABLE_ACTIVE, CandidateEligibility.SCORABLE_AMBIGUOUS)

                if r_is_active and _is_warning_or_disclaimer(s):
                    # Combination 4: Conflict (Active vs Warning/Advisory)
                    conflict_record = EngineConflictRecord(
                        regex_finding_id=r.finding_id or r.id,
                        semantic_finding_id=s.finding_id or s.id,
                        occurrence_span=span,
                        regex_qualification=r.eligibility,
                        semantic_qualification=s.eligibility,
                        conflict_type="ACTIVE_VS_WARNING",
                        resolution_action="PRESERVE_BOTH_EMIT_ADVISORY",
                    )
                    conflicts.append(conflict_record)

                    # Preserve regex finding as scorable
                    reconciled_evidence.append(_copy_evidence_item(r))

                    # Emit semantic finding as advisory (wt=0)
                    sem_context = s.context_required or ""
                    conflict_tag = f"conflicts_with:{r.finding_id or r.id}"
                    updated_context = f"{sem_context};{conflict_tag}".strip(";") if sem_context else conflict_tag
                    reconciled_evidence.append(_copy_evidence_item(s, context_required=updated_context))

                elif s_is_active and _is_warning_or_disclaimer(r):
                    # Inverted conflict: Semantic is active, regex is warning/advisory
                    conflict_record = EngineConflictRecord(
                        regex_finding_id=r.finding_id or r.id,
                        semantic_finding_id=s.finding_id or s.id,
                        occurrence_span=span,
                        regex_qualification=r.eligibility,
                        semantic_qualification=s.eligibility,
                        conflict_type="ACTIVE_VS_WARNING",
                        resolution_action="PRESERVE_BOTH_EMIT_ADVISORY",
                    )
                    conflicts.append(conflict_record)

                    reconciled_evidence.append(_copy_evidence_item(s))
                    reg_context = r.context_required or ""
                    conflict_tag = f"conflicts_with:{s.finding_id or s.id}"
                    updated_context = f"{reg_context};{conflict_tag}".strip(";") if reg_context else conflict_tag
                    reconciled_evidence.append(_copy_evidence_item(r, context_required=updated_context))

                else:
                    # Combinations 1, 2, 3: Corroboration
                    # Apply Max-Effective-Weight Policy
                    w_r = get_effective_weight(r)
                    w_s = get_effective_weight(s)

                    if w_s > w_r:
                        # Row 3: Semantic has higher weight (e.g. Semantic Active 65 vs Regex Ambiguous 10)
                        primary_item = s
                        secondary_item = r
                        primary_engine = "SEMANTIC"
                        secondary_engine = "REGEX"
                    else:
                        # Rows 1, 2: Regex has equal or higher weight
                        primary_item = r
                        secondary_item = s
                        primary_engine = "REGEX"
                        secondary_engine = "SEMANTIC"

                    # Capture original secondary qualification BEFORE mutation
                    sec_orig_eligibility = secondary_item.eligibility
                    sec_orig_explanation = secondary_item.explanation
                    sec_orig_context = secondary_item.context_required or ""

                    corroboration_record = EngineCorroborationRecord(
                        primary_finding_id=primary_item.finding_id or primary_item.id,
                        secondary_finding_id=secondary_item.finding_id or secondary_item.id,
                        occurrence_span=span,
                        primary_engine=primary_engine,
                        secondary_engine=secondary_engine,
                        primary_engine_qualification=primary_item.eligibility,
                        secondary_engine_original_qualification=sec_orig_eligibility,
                        resolution_action="CORROBORATED_DUAL_RETENTION_SCORING_WORKAROUND",
                        scoring_weight_assigned=0,
                    )
                    corroborations.append(corroboration_record)

                    # Emit Primary Item with original eligibility intact
                    reconciled_evidence.append(_copy_evidence_item(primary_item))

                    # Emit Secondary Item with F-03 Transformation Contract
                    context_suffix = f"corroborated_with:{primary_item.finding_id or primary_item.id};original_eligibility:{sec_orig_eligibility.value}"
                    new_context = f"{sec_orig_context};{context_suffix}".strip(";") if sec_orig_context else context_suffix
                    new_explanation = (
                        f"[CORROBORATED occurrence; original semantic qualification={sec_orig_eligibility.value}; "
                        f"scoring weight suppressed to prevent double-counting with {primary_item.finding_id or primary_item.id}] "
                        f"{sec_orig_explanation}"
                    )

                    secondary_copy = _copy_evidence_item(
                        secondary_item,
                        eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
                        explanation=new_explanation,
                        context_required=new_context,
                        rejection_reason=None,
                    )
                    reconciled_evidence.append(secondary_copy)

        # 4. Emit unmatched regex findings with original eligibility intact (Combinations 6, 7, 8, 10)
        for r_idx, r in enumerate(valid_regex):
            if r_idx not in matched_regex_indices:
                reconciled_evidence.append(_copy_evidence_item(r))

        # 5. Emit unmatched semantic findings with original eligibility intact (Combinations 6, 7, 8, 9)
        for s_idx, s in enumerate(valid_semantic):
            if s_idx not in matched_semantic_indices:
                reconciled_evidence.append(_copy_evidence_item(s))

        return ReconciliationResult(
            reconciled_evidence=reconciled_evidence,
            corroborations=corroborations,
            conflicts=conflicts,
            rejected_candidates=durable_rejected,
        )

    except Exception as e:
        logger.exception("reconcile_detection_engines unexpected failure fail-closed: %s", e)
        # Fallback: preserve valid regex items intact
        fallback_evidence = [_copy_evidence_item(r) for r in regex_items if isinstance(r, EvidenceItem)]
        return ReconciliationResult(
            reconciled_evidence=fallback_evidence,
            corroborations=[],
            conflicts=[],
            rejected_candidates=durable_rejected,
        )
