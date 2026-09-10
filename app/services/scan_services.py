from urllib.parse import urlparse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.scan import Scan
from app.schemas.scan import InputType

SCAM_PATTERNS: dict[str, int] = {
    "registration fee": 30,
    "guaranteed job": 25,
    "no interview": 20,
    "whatsapp only": 15,
    "urgent hiring": 10,
}


def detect_input_type(content: str) -> InputType:
    stripped_content = content.strip()
    parsed = urlparse(stripped_content)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return InputType.URL
    elif stripped_content.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
        return InputType.IMAGE
    else:
        return InputType.TEXT


def calculate_risk_score(content: str) -> int:
    content_lower = content.lower()
    score = 0
    for pattern, pattern_score in SCAM_PATTERNS.items():
        if pattern in content_lower:
            score += pattern_score
    return min(score, 100)


def get_verdict(risk_score: int) -> str:
    if risk_score <= 30:
        return "Low Risk"
    elif risk_score <= 60:
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
