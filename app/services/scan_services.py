import json
import logging
import re
import time
from typing import Any
import unicodedata
from urllib.parse import urlparse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.scan import Scan
from app.schemas.scan import AnalysisMode, InputType
from app.services.detection.models import DetectionResult, EvidenceItem
from app.services.detection.scoring import get_effective_weight
from app.services.detection.service import analyze_job_posting
from app.services.gemini_service import (
    EXTRACTION_FIELDS,
    analyze_job_content,
    parse_extracted_details,
)

logger = logging.getLogger(__name__)


def _emit_scan_telemetry(event: str, level: int, payload: dict[str, Any]) -> None:
    """
    Emits structured scan operational telemetry.
    Completely isolated so that telemetry errors can NEVER interfere with or fail a scan.
    """
    try:
        json_payload = json.dumps(payload, default=str)
        logger.log(
            level,
            "scan_telemetry_event: %s | data=%s",
            event,
            json_payload,
            extra={"telemetry": payload},
        )
    except Exception:
        # If logger fails (e.g. disk full, broken stream, serialization error),
        # silently suppress to guarantee scan execution is never failed by telemetry.
        pass


class ScanValidationError(ValueError):
    """Base exception for client-input validation failures in the scan service."""
    pass


class UnsupportedInputModeError(ScanValidationError):
    """Raised when an unsupported input mode (e.g. standalone URL or image file) is submitted."""
    pass


class ScanServiceUnavailableError(Exception):
    """Raised when both detection engines fail and the scan cannot be completed."""

    def __init__(
        self,
        message: str = "Scan service is temporarily unavailable. Please retry shortly.",
        retry_after: int = 30,
    ):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


SCAM_PATTERNS: dict[str, int] = {
    "registration fee": 60,
    "guaranteed job": 25,
    "no interview": 20,
    "whatsapp only": 15,
    "urgent hiring": 10,
}


def detect_input_type(content: str) -> InputType:
    stripped_content = content.strip()

    # Standalone URL check: no unencoded whitespace or newlines, valid scheme and netloc
    if not any(c.isspace() for c in stripped_content):
        parsed = urlparse(stripped_content)
        if parsed.scheme in ("http", "https", "ftp", "sftp") and parsed.netloc:
            return InputType.URL
        if stripped_content.lower().startswith("www.") and "." in stripped_content[4:]:
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
    """
    Backward-compatible wrapper delegating to the Phase 3 canonical detection subsystem.
    Calculates unified deterministic rule risk score.
    """
    if not content or not content.strip():
        return 0
    try:
        return analyze_job_posting(content).rule_score
    except Exception:
        logger.exception("analyze_job_posting failed in calculate_risk_score wrapper")
        return 0


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


def format_detection_indicators(evidence_items: list[EvidenceItem]) -> list[str]:
    """
    Formats scorable Phase 3 findings into standard list-of-strings indicators.
    Enforces Modified Option A format:
      f'{item.explanation} — Evidence: "{item.matched_text}"'
    Filters out zero-weight and non-scorable findings. Preserves first-seen case
    and maintains insertion order without duplicates.
    """
    seen: set[str] = set()
    indicators: list[str] = []
    for item in evidence_items:
        if get_effective_weight(item) <= 0:
            continue
        matched = (item.matched_text or "").strip()
        explanation = (item.explanation or "").strip()
        if matched and explanation:
            indicator_str = f'{explanation} — Evidence: "{matched}"'
        elif explanation:
            indicator_str = explanation
        elif matched:
            indicator_str = f'Evidence: "{matched}"'
        else:
            continue

        key = indicator_str.lower()
        if key not in seen:
            seen.add(key)
            indicators.append(indicator_str)
    return indicators


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
    detection_result: DetectionResult | None,
    rule_score: int,
    verdict: str,
    gemini_score: int,
    gemini_explanation: str,
    matched_patterns: list[str] | None = None,
) -> str:
    """
    Builds an explanation acknowledging rule-based warning signs when the rule
    score exceeds Gemini's contextual score. Avoids claiming definitive fraud,
    describes matched patterns cautiously as warning signs, and preserves
    useful context from Gemini.
    """
    patterns: list[str] = []
    seen_pats: set[str] = set()
    if detection_result and detection_result.evidence_items:
        for it in detection_result.evidence_items:
            if get_effective_weight(it) > 0 and it.matched_text:
                pat_str = f"'{it.matched_text}'"
                if pat_str.lower() not in seen_pats:
                    seen_pats.add(pat_str.lower())
                    patterns.append(pat_str)
    elif matched_patterns:
        for p in matched_patterns:
            pat_str = f"'{p}'"
            if pat_str.lower() not in seen_pats:
                seen_pats.add(pat_str.lower())
                patterns.append(pat_str)

    if patterns:
        patterns_str = ", ".join(patterns)
        warning_phrase = "warning sign" if len(patterns) == 1 else "warning signs"
        explanation = (
            f"Rule-based detection identified {warning_phrase} ({patterns_str}), "
            f"elevating the risk score to {rule_score} ({verdict}) over the contextual score of {gemini_score}. "
            f"While detected phrases represent warning signs rather than definitive proof of fraud, they warrant caution."
        )
    else:
        explanation = (
            f"Rule-based detection elevated the risk score to {rule_score} ({verdict}) "
            f"over the contextual score of {gemini_score}. Caution is advised."
        )

    trimmed_gemini = gemini_explanation.strip() if gemini_explanation else ""
    if trimmed_gemini:
        explanation += f" Contextual analysis noted: {trimmed_gemini}"
    return explanation


def reconcile_scan_findings(
    detection_or_content: DetectionResult | str | None = None,
    gemini_score: int = 0,
    gemini_explanation: str = "",
    gemini_indicators: list[str] | None = None,
    *,
    content: str | None = None,
    detection_result: DetectionResult | None = None,
) -> tuple[int, str, str, list[str]]:
    """
    Reconciles deterministic detection findings with Gemini contextual analysis.
    Preserves max(rule_score, gemini_score).
    When rule_score > gemini_score, produces an explanation and indicators that
    acknowledge the matched rule-based findings without discarding Gemini's context.
    When gemini_score >= rule_score, preserves Gemini's contextual explanation and
    combines all grounded indicators.
    Returns: (risk_score, verdict, explanation, indicators)
    """
    actual_detection: DetectionResult | None = detection_result
    raw_content: str | None = content

    if actual_detection is None:
        if isinstance(detection_or_content, DetectionResult):
            actual_detection = detection_or_content
        elif isinstance(detection_or_content, str):
            raw_content = detection_or_content

    if actual_detection is None and raw_content:
        try:
            actual_detection = analyze_job_posting(raw_content)
        except Exception:
            logger.exception("Failed to execute analyze_job_posting in reconcile_scan_findings")
            actual_detection = None

    if actual_detection is not None:
        rule_score = actual_detection.rule_score
        rule_indicators = format_detection_indicators(actual_detection.evidence_items)
    else:
        rule_score = 0
        rule_indicators = []

    risk_score = max(rule_score, gemini_score)
    verdict = get_verdict(risk_score)
    safe_gemini_indicators = list(gemini_indicators) if gemini_indicators else []

    if rule_score > gemini_score:
        explanation = _build_rule_elevated_explanation(
            detection_result=actual_detection,
            rule_score=rule_score,
            verdict=verdict,
            gemini_score=gemini_score,
            gemini_explanation=gemini_explanation,
        )
        indicators = _combine_indicators(rule_indicators, safe_gemini_indicators)
    else:
        explanation = gemini_explanation
        indicators = _combine_indicators(rule_indicators, safe_gemini_indicators)

    return risk_score, verdict, explanation, indicators


def create_scan(content: str, user_id: int, db: Session) -> Scan:
    start_time = time.perf_counter()

    # Control character validation
    if any(unicodedata.category(c) == "Cc" and c not in ("\n", "\r", "\t") for c in content):
        raise ScanValidationError("Content contains invalid control characters")

    # Detect and handle input modes
    input_type = detect_input_type(content)
    if input_type == InputType.IMAGE:
        raise UnsupportedInputModeError(
            "Image scanning is not currently supported. Image vision analysis is not implemented; please paste the job posting text directly."
        )

    if input_type == InputType.URL:
        from app.services.url_fetch_service import fetch_job_url

        original_url = content.strip()
        scannable_text, final_url = fetch_job_url(original_url)
        # Sanitize fetched text: strip null bytes and inappropriate control characters
        scannable_text = "".join(
            c for c in scannable_text
            if not (unicodedata.category(c) == "Cc" and c not in ("\n", "\r", "\t"))
        )
    else:
        original_url = None
        final_url = None
        scannable_text = content

    # Length and whitespace validation for scannable text content
    non_ws_count = sum(1 for c in scannable_text if not c.isspace()) if scannable_text else 0
    if non_ws_count < 10:
        raise ScanValidationError("Content must contain at least 10 non-whitespace characters")
    if len(scannable_text) > 50000:
        raise ScanValidationError("Content exceeds maximum length of 50000 characters")

    # Phase 3 Deterministic Detection Engine (synchronous, ~35ms)
    detection_result: DetectionResult | None = None
    phase3_failed = False
    phase3_error: Exception | None = None
    try:
        detection_result = analyze_job_posting(scannable_text)
        rule_score = detection_result.rule_score
    except Exception as exc:
        logger.exception("Phase 3 detection engine failed defensively; defaulting rule score to 0")
        phase3_failed = True
        phase3_error = exc
        detection_result = None
        rule_score = 0

    explanation: str | None = None
    indicators: list[str] | None = None
    job_details: dict[str, str | None] | None = None
    analysis_mode: str = AnalysisMode.RULES_ONLY.value

    # Gemini contextual analysis
    gemini_result: dict[str, Any] | None = None
    gemini_failed = False
    gemini_error: Exception | None = None

    try:
        raw_gemini = analyze_job_content(scannable_text)
        if not isinstance(raw_gemini, dict):
            raise ValueError("Gemini analysis returned unexpected format")

        gemini_score = raw_gemini.get("risk_score")
        if (
            isinstance(gemini_score, bool)
            or not isinstance(gemini_score, int)
            or not 0 <= gemini_score <= 100
        ):
            raise ValueError("Gemini returned an invalid risk score")

        gemini_explanation = raw_gemini.get("explanation")
        if not isinstance(gemini_explanation, str):
            raise ValueError("Gemini returned an invalid explanation")

        gemini_indicators = raw_gemini.get("indicators")
        if (
            not isinstance(gemini_indicators, list)
            or not all(isinstance(item, str) for item in gemini_indicators)
        ):
            raise ValueError("Gemini returned invalid indicators")

        gemini_result = raw_gemini
    except Exception as exc:
        logger.exception(
            "Gemini analysis failed; falling back to deterministic rule scoring"
        )
        gemini_failed = True
        gemini_error = exc

    # Dual-engine failure protection (Option 2)
    if phase3_failed and gemini_failed:
        logger.critical(
            "Both detection engines failed for scan request: Phase 3 error: %r | Gemini error: %r",
            phase3_error,
            gemini_error,
        )
        try:
            db.rollback()
        except Exception as rb_exc:
            logger.error(
                "Database rollback failed during dual-engine failure cleanup: %s",
                rb_exc,
            )
        _emit_scan_telemetry(
            event="scan_dual_engine_failure",
            level=logging.CRITICAL,
            payload={
                "event": "scan_dual_engine_failure",
                "phase3_error_type": type(phase3_error).__name__ if phase3_error else "UnknownError",
                "gemini_error_type": type(gemini_error).__name__ if gemini_error else "UnknownError",
                "scan_service_duration_ms": round((time.perf_counter() - start_time) * 1000),
                "retry_after": 30,
            },
        )
        raise ScanServiceUnavailableError(
            message="Scan service is temporarily unavailable. Please retry shortly.",
            retry_after=30,
        )

    # Post-engine orchestration: cross-engine reconciliation or rules-only fallback
    if not gemini_failed and gemini_result is not None:
        if phase3_failed:
            _emit_scan_telemetry(
                event="scan_engine_degraded",
                level=logging.WARNING,
                payload={
                    "event": "scan_engine_degraded",
                    "fallback_mode": AnalysisMode.AI_ASSISTED.value,
                    "failed_engine": "phase3",
                    "error_type": type(phase3_error).__name__ if phase3_error else "UnknownError",
                    "scan_service_duration_ms": round((time.perf_counter() - start_time) * 1000),
                },
            )
        risk_score, verdict, explanation, indicators = reconcile_scan_findings(
            detection_result=detection_result,
            gemini_score=gemini_result["risk_score"],
            gemini_explanation=gemini_result["explanation"],
            gemini_indicators=gemini_result["indicators"],
        )
        job_details = parse_extracted_details(gemini_result.get("job_details"))
        analysis_mode = AnalysisMode.AI_ASSISTED.value
    else:
        _emit_scan_telemetry(
            event="scan_engine_degraded",
            level=logging.WARNING,
            payload={
                "event": "scan_engine_degraded",
                "fallback_mode": AnalysisMode.RULES_ONLY.value,
                "failed_engine": "gemini",
                "error_type": type(gemini_error).__name__ if gemini_error else "UnknownError",
                "scan_service_duration_ms": round((time.perf_counter() - start_time) * 1000),
            },
        )
        risk_score = rule_score
        verdict = get_verdict(risk_score)
        analysis_mode = AnalysisMode.RULES_ONLY.value
        job_details = None

        if detection_result is not None and rule_score > 0:
            explanation = (
                f"{detection_result.fallback_explanation} Note: AI contextual "
                f"analysis was unavailable; scored using deterministic rule engine."
            )
            rule_indicators = format_detection_indicators(detection_result.evidence_items)
            indicators = rule_indicators if rule_indicators else None
        else:
            explanation = None
            indicators = None

    if input_type == InputType.URL:
        if job_details is None:
            job_details = {field: None for field in EXTRACTION_FIELDS}
        # M-1: Always record the exact original URL submitted by the user
        job_details["scanned_url"] = original_url
        if not job_details.get("application_url"):
            job_details["application_url"] = final_url or original_url

    new_scan = Scan(
        user_id=user_id,
        input_type=input_type.value,
        content=scannable_text,
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
        _emit_scan_telemetry(
            event="scan_completed",
            level=logging.INFO,
            payload={
                "event": "scan_completed",
                "scan_id": new_scan.id,
                "input_type": input_type.value,
                "analysis_mode": analysis_mode,
                "risk_score": risk_score,
                "rule_score": rule_score,
                "gemini_score": gemini_result.get("risk_score") if gemini_result else None,
                "indicator_count": len(indicators) if indicators else 0,
                "scan_service_duration_ms": round((time.perf_counter() - start_time) * 1000),
                "engine_status": {
                    "phase3": "failed" if phase3_failed else "ok",
                    "gemini": "failed" if gemini_failed else "ok",
                },
            },
        )
        return new_scan
    except Exception:
        db.rollback()
        raise


def get_user_scans(
    user_id: int,
    db: Session,
    limit: int = 20,
    offset: int = 0,
    analysis_mode: str | None = None,
    verdict: str | None = None,
) -> list[Scan]:
    # Extract defaults if FastAPI Query objects are passed during direct function calls
    actual_limit = limit.default if hasattr(limit, "default") else limit
    actual_offset = offset.default if hasattr(offset, "default") else offset
    actual_mode = analysis_mode.default if hasattr(analysis_mode, "default") else analysis_mode
    actual_verdict = verdict.default if hasattr(verdict, "default") else verdict

    query = select(Scan).where(Scan.user_id == user_id)

    if actual_mode is not None:
        mode_val = actual_mode.value if hasattr(actual_mode, "value") else str(actual_mode)
        query = query.where(Scan.analysis_mode == mode_val)

    if actual_verdict is not None:
        verdict_val = actual_verdict.value if hasattr(actual_verdict, "value") else str(actual_verdict)
        query = query.where(Scan.verdict == verdict_val)

    query = (
        query
        .order_by(Scan.created_at.desc(), Scan.id.desc())
        .limit(actual_limit)
        .offset(actual_offset)
    )

    return list(db.scalars(query).all())


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
