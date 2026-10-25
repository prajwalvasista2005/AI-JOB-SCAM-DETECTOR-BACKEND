from datetime import datetime, timezone
from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    input_type: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    verdict: Mapped[str] = mapped_column(String(255), nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    indicators: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    job_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    analysis_mode: Mapped[str] = mapped_column(
        String(50), nullable=False, default="legacy_unknown"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user = relationship("User", back_populates="scans")

    def __init__(self, **kwargs):
        if "analysis_mode" not in kwargs or kwargs["analysis_mode"] is None:
            kwargs["analysis_mode"] = "legacy_unknown"
        super().__init__(**kwargs)
