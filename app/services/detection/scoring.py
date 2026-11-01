from app.services.detection.models import (
    DetectionSource,
    EvidenceItem,
    FindingSeverity,
    FindingStatus,
    ScamCategory,
)

SEVERITY_WEIGHTS: dict[FindingSeverity, int] = {
    FindingSeverity.CRITICAL: 65,
    FindingSeverity.HIGH: 45,
    FindingSeverity.MEDIUM: 25,
    FindingSeverity.LOW: 10,
    FindingSeverity.INFORMATIONAL: 0,
}


def get_effective_weight(item: EvidenceItem) -> int:
    """
    Computes effective weight according to frozen V3.2 provenance and evidence rules:
      - UNKNOWN category: 0 (unrecognized scam category cannot anchor or compound).
      - AMBIGUOUS_SIGNAL status: capped at LOW (10) because benign interpretations exist.
      - UNKNOWN source: capped at LOW (10) because untrusted provenance has no special trust.
      - Valid findings: full severity weight.
    """
    if item.category == ScamCategory.UNKNOWN:
        return 0

    base_weight = SEVERITY_WEIGHTS.get(item.severity, 0)
    if base_weight == 0:
        return 0

    # Ambiguous signals cannot anchor above LOW
    if item.status == FindingStatus.AMBIGUOUS_SIGNAL:
        base_weight = min(base_weight, SEVERITY_WEIGHTS[FindingSeverity.LOW])

    # Unknown provenance cannot anchor above LOW
    if item.source == DetectionSource.UNKNOWN:
        base_weight = min(base_weight, SEVERITY_WEIGHTS[FindingSeverity.LOW])

    return base_weight


def calculate_evidence_score(evidence_items: list[EvidenceItem]) -> int:
    """
    Calculates unified risk score using Multi-Category Anchor Exclusion
    while enforcing frozen V3.2 provenance and evidence bounding.
    """
    if not evidence_items:
        return 0

    item_weights = [(item, get_effective_weight(item)) for item in evidence_items]
    valid_weights = [w for _, w in item_weights if w > 0]
    if not valid_weights:
        return 0

    # 1. Base Anchor: Maximum effective severity weight
    max_severity_weight = max(valid_weights)
    if max_severity_weight == 0:
        return 0

    # 2. Multi-Category Anchor Identification:
    # Identify EVERY valid category that has a finding tied at max_severity_weight.
    anchor_categories = {
        item.category
        for item, w in item_weights
        if w == max_severity_weight and item.category != ScamCategory.UNKNOWN
    }

    # 3. Remaining Categories:
    # Only known categories NOT in anchor_categories are eligible for residual compounding.
    remaining_categories = {
        item.category
        for item, w in item_weights
        if item.category != ScamCategory.UNKNOWN and item.category not in anchor_categories and w > 0
    }

    # 4. Bounded Residual Compounding across distinct remaining categories
    residual_boost = 0
    for cat in sorted(remaining_categories):
        highest_cat_weight = max(w for item, w in item_weights if item.category == cat)
        if highest_cat_weight >= 45:      # CRITICAL or HIGH in remaining category
            residual_boost += 10
        elif highest_cat_weight >= 25:    # MEDIUM in remaining category
            residual_boost += 5
        # LOW and INFORMATIONAL contribute 0 residual boost

    # 5. Cap residual boost at 30, and total score at 100
    capped_residual = min(30, residual_boost)
    return min(100, max_severity_weight + capped_residual)


def get_verdict(risk_score: int) -> str:
    """
    Classifies a unified risk score into standardized verdicts:
      - Low Risk: 0 to 20
      - Medium Risk: 21 to 50
      - High Risk: 51 to 100
    """
    if risk_score <= 20:
        return "Low Risk"
    elif risk_score <= 50:
        return "Medium Risk"
    else:
        return "High Risk"
