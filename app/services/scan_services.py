import logging
import re
from urllib.parse import urlparse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.scan import Scan
from app.schemas.scan import AnalysisMode, InputType
from app.services.gemini_service import analyze_job_content, parse_extracted_details

logger = logging.getLogger(__name__)

SCAM_PATTERNS: dict[str, int] = {
    "registration fee": 60,
    "guaranteed job": 25,
    "no interview": 20,
    "whatsapp only": 15,
    "urgent hiring": 10,
}


def detect_input_type(content: str) -> InputType:
    stripped_content = content.strip()

    # Standalone URL check: no unencoded whitespace or newlines, valid http/https scheme and netloc
    if not any(c.isspace() for c in stripped_content):
        parsed = urlparse(stripped_content)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return InputType.URL
        elif stripped_content.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
            return InputType.IMAGE

    # Standalone image filename check: single line, at most two words (e.g. "job ad.png"), ending with image extension
    if (
        "\n" not in stripped_content
        and "\r" not in stripped_content
        and len(stripped_content.split()) <= 2
        and stripped_content.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
    ):
        return InputType.IMAGE

    return InputType.TEXT


def _is_negated_match(prefix: str) -> bool:
    """
    Checks if a pattern match is negated within the same clause or sentence.
    Matches negation words (no, never, without, not, zero) preceding the match
    within 0-4 intervening words.
    """
    last_clause = re.split(r"[\.\!\?\;\n]", prefix)[-1].strip()
    negation_regex = r"\b(?:no|never|without|not|zero)\b(?:\s+\w+){0,4}\s*$"
    return bool(re.search(negation_regex, last_clause))


def _is_pattern_present(content_lower: str, pattern: str) -> bool:
    """
    Determines whether a scam pattern is present in the text and not negated.
    Patterns that inherently express negation (e.g., 'no interview') are matched
    directly. For other patterns, word boundaries and plural forms are supported,
    and negation prefixes within the same clause suppress the match unless an
    unnegated occurrence also exists.
    """
    if pattern.startswith("no "):
        return pattern in content_lower

    escaped = re.escape(pattern)
    pattern_regex = rf"\b{escaped}s?\b"
    matches = list(re.finditer(pattern_regex, content_lower))
    if not matches:
        return False

    for m in matches:
        prefix = content_lower[max(0, m.start() - 60) : m.start()]
        if not _is_negated_match(prefix):
            return True
    return False


def find_matched_patterns(content: str) -> list[str]:
    """
    Identifies the scam patterns present in the content that are not negated.
    Returns matched pattern names in the order defined in SCAM_PATTERNS.
    """
    content_lower = content.lower()
    return [
        pattern
        for pattern in SCAM_PATTERNS
        if _is_pattern_present(content_lower, pattern)
    ]


def calculate_risk_score(content: str) -> int:
    matched = find_matched_patterns(content)
    score = sum(SCAM_PATTERNS[p] for p in matched)
    return min(score, 100)


def get_verdict(risk_score: int) -> str:
    """
    Classifies a risk score into a standardized verdict.

    Verdict Policy:
      - Low Risk (0 to 20):
          Benign postings (0 points) or isolated mild urgency/communication signals
          (e.g., "urgent hiring" [10], "whatsapp only" [15], "no interview" [20]).
      - Medium Risk (21 to 50):
          Suspicious postings with bold claims (e.g., "guaranteed job" [25])
          or combined low/moderate signals (e.g., "urgent hiring" + "whatsapp only" [25],
          "guaranteed job" + "no interview" [45]).
      - High Risk (51 to 100):
          Critical fraud indicators like upfront advance fees (e.g., "registration fee" [60])
          or multiple compounding red flags (e.g., "guaranteed job" + "no interview" + "urgent hiring" [55],
          or all patterns combined [capped at 100]).
    """
    if risk_score <= 20:
        return "Low Risk"
    elif risk_score <= 50:
        return "Medium Risk"
    else:
        return "High Risk"


def _combine_indicators(
    rule_patterns: list[str], gemini_indicators: list[str]
) -> list[str]:
    """
    Combines rule pattern indicators with Gemini indicators without duplicates.
    Preserves case from the first occurrence and maintains insertion order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in list(rule_patterns) + list(gemini_indicators):
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _build_rule_elevated_explanation(
    matched_patterns: list[str],
    rule_score: int,
    verdict: str,
    gemini_score: int,
    gemini_explanation: str,
) -> str:
    """
    Builds an explanation acknowledging rule-based warning signs when the rule
    score exceeds Gemini's contextual score. Avoids claiming definitive fraud,
    describes matched patterns cautiously as warning signs, and preserves
    useful context from Gemini.
    """
    patterns_str = ", ".join(f"'{p}'" for p in matched_patterns)
    warning_phrase = "warning sign" if len(matched_patterns) == 1 else "warning signs"
    explanation = (
        f"Rule-based detection identified {warning_phrase} ({patterns_str}), "
        f"elevating the risk score to {rule_score} ({verdict}) over the contextual score of {gemini_score}. "
        f"While detected phrases represent warning signs rather than definitive proof of fraud, they warrant caution."
    )
    trimmed_gemini = gemini_explanation.strip()
    if trimmed_gemini:
        explanation += f" Contextual analysis noted: {trimmed_gemini}"
    return explanation


def reconcile_scan_findings(
    content: str,
    gemini_score: int,
    gemini_explanation: str,
    gemini_indicators: list[str],
) -> tuple[int, str, str, list[str]]:
    """
    Reconciles rule-based pattern detection with Gemini contextual analysis.
    Preserves max(rule_score, gemini_score).
    When rule_score > gemini_score, produces an explanation and indicators that
    acknowledge the matched rule-based findings without discarding Gemini's context.
    When gemini_score >= rule_score, preserves Gemini's contextual explanation and indicators.
    Returns: (risk_score, verdict, explanation, indicators)
    """
    matched_patterns = find_matched_patterns(content)
    rule_score = min(sum(SCAM_PATTERNS[p] for p in matched_patterns), 100)
    risk_score = max(rule_score, gemini_score)
    verdict = get_verdict(risk_score)

    if rule_score > gemini_score:
        explanation = _build_rule_elevated_explanation(
            matched_patterns=matched_patterns,
            rule_score=rule_score,
            verdict=verdict,
            gemini_score=gemini_score,
            gemini_explanation=gemini_explanation,
        )
        indicators = _combine_indicators(matched_patterns, gemini_indicators)
    else:
        explanation = gemini_explanation
        indicators = _combine_indicators([], gemini_indicators)

    return risk_score, verdict, explanation, indicators


def create_scan(content: str, user_id: int, db: Session) -> Scan:
    input_type = detect_input_type(content)

    # Existing rule-based detection
    rule_score = calculate_risk_score(content)

    explanation: str | None = None
    indicators: list[str] | None = None
    job_details: dict[str, str | None] | None = None
    analysis_mode: str = AnalysisMode.RULES_ONLY.value

    # Gemini provides contextual analysis
    try:
        gemini_result = analyze_job_content(content)
        if not isinstance(gemini_result, dict):
            raise ValueError("Gemini analysis returned unexpected format")

        gemini_score = gemini_result.get("risk_score")
        if (
            isinstance(gemini_score, bool)
            or not isinstance(gemini_score, int)
            or not 0 <= gemini_score <= 100
        ):
            raise ValueError("Gemini returned an invalid risk score")

        gemini_explanation = gemini_result.get("explanation")
        if not isinstance(gemini_explanation, str):
            raise ValueError("Gemini returned an invalid explanation")

        gemini_indicators = gemini_result.get("indicators")
        if (
            not isinstance(gemini_indicators, list)
            or not all(isinstance(item, str) for item in gemini_indicators)
        ):
            raise ValueError("Gemini returned invalid indicators")

        risk_score, verdict, explanation, indicators = reconcile_scan_findings(
            content=content,
            gemini_score=gemini_score,
            gemini_explanation=gemini_explanation,
            gemini_indicators=gemini_indicators,
        )
        job_details = parse_extracted_details(gemini_result.get("job_details"))
        analysis_mode = AnalysisMode.AI_ASSISTED.value

    except Exception:
        logger.exception(
            "Gemini analysis failed; using rule-based scoring"
        )
        risk_score = rule_score
        verdict = get_verdict(risk_score)
        explanation = None
        indicators = None
        job_details = None
        analysis_mode = AnalysisMode.RULES_ONLY.value

    new_scan = Scan(
        user_id=user_id,
        input_type=input_type.value,
        content=content,
        risk_score=risk_score,
        verdict=verdict,
        analysis_mode=analysis_mode,
        explanation=explanation,
        indicators=indicators,
        job_details=job_details,
    )

    try:
        db.add(new_scan)
        db.commit()
        db.refresh(new_scan)
        return new_scan
    except Exception:
        db.rollback()
        raise


def get_user_scans(user_id: int, db: Session) -> list[Scan]:
    return list(
        db.scalars(
            select(Scan).where(Scan.user_id == user_id).order_by(Scan.created_at.desc())
        ).all()
    )


def get_scan_by_id(scan_id: int, user_id: int, db: Session) -> Scan | None:
    return db.scalars(
        select(Scan).where(Scan.id == scan_id, Scan.user_id == user_id)
    ).first()


def delete_scan(scan_id: int, user_id: int, db: Session) -> bool:
    scan = get_scan_by_id(scan_id=scan_id, user_id=user_id, db=db)
    if not scan:
        return False
    try:
        db.delete(scan)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
