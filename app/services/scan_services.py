import re
from urllib.parse import urlparse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.scan import Scan
from app.schemas.scan import InputType

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


def calculate_risk_score(content: str) -> int:
    content_lower = content.lower()
    score = 0
    for pattern, pattern_score in SCAM_PATTERNS.items():
        if _is_pattern_present(content_lower, pattern):
            score += pattern_score
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


def create_scan(content: str, user_id: int, db: Session) -> Scan:
    input_type = detect_input_type(content)
    risk_score = calculate_risk_score(content)
    verdict = get_verdict(risk_score)

    new_scan = Scan(
        user_id=user_id,
        input_type=input_type.value,
        content=content,
        risk_score=risk_score,
        verdict=verdict,
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
