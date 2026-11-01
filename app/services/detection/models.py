from datetime import datetime
from enum import Enum
from typing import Any
import logging
from pydantic import BaseModel, Field, ConfigDict, field_validator

logger = logging.getLogger(__name__)


class ScamCategory(str, Enum):
    ADVANCE_FEE = "advance_fee"
    TASK_OPTIMIZATION = "task_optimization"
    FAKE_RECRUITMENT = "fake_recruitment"
    EQUIPMENT_OVERPAYMENT = "equipment_overpayment"
    SALARY_MANIPULATION = "salary_manipulation"
    CREDENTIAL_HARVESTING = "credential_harvesting"
    SUSPICIOUS_COMMUNICATION = "suspicious_communication"
    UNKNOWN = "unknown"


class FindingSeverity(str, Enum):
    CRITICAL = "CRITICAL"          # Base Anchor = 65 (Immediate High Risk)
    HIGH = "HIGH"                  # Base Anchor = 45 (Upper Medium Risk)
    MEDIUM = "MEDIUM"              # Base Anchor = 25 (Lower Medium Risk)
    LOW = "LOW"                    # Base Anchor = 10 (Low Risk)
    INFORMATIONAL = "INFORMATIONAL"# Base Anchor = 0  (Zero Score Contribution)


class FindingStatus(str, Enum):
    EXPLICIT_CLAIM = "EXPLICIT_CLAIM"              # Verbatim unnegated statement directly quoted
    CONTEXTUAL_INFERENCE = "CONTEXTUAL_INFERENCE"  # Logical deduction from document syntax/context
    AMBIGUOUS_SIGNAL = "AMBIGUOUS_SIGNAL"          # Pattern present but benign interpretations exist


class DetectionSource(str, Enum):
    DETERMINISTIC_RULE = "DETERMINISTIC_RULE"      # Found via deterministic regex / syntax rules
    AI_INFERENCE = "AI_INFERENCE"                  # Identified via Gemini contextual analysis
    RULE_AND_AI = "RULE_AND_AI"                    # Flagged by deterministic rule and supported by AI
    UNKNOWN = "UNKNOWN"                            # Unrecognized historical provenance; no special trust


class EvidenceCompleteness(str, Enum):
    HIGH = "HIGH"          # Extensive context: employer, role, compensation, and workflow clear
    MODERATE = "MODERATE"  # Standard context: core parameters evaluable with reasonable certainty
    LOW = "LOW"            # Sparse or truncated text: high uncertainty due to missing information


class EvidenceItem(BaseModel):
    id: str = Field(default="FINDING_UNKNOWN", description="Machine-readable finding identifier, e.g. ADV_FEE_UPI_DIRECT")
    category: ScamCategory = ScamCategory.UNKNOWN
    source: DetectionSource = DetectionSource.UNKNOWN
    severity: FindingSeverity = FindingSeverity.LOW
    status: FindingStatus = FindingStatus.AMBIGUOUS_SIGNAL
    matched_text: str | None = None
    explanation: str = Field(default="No explanation provided")
    context_required: str | None = None
    finding_id: str | None = Field(default=None, description="Deterministic instance ID, e.g. ADV_FEE_REGISTRATION_1")
    contradicts_finding_ids: list[str] = Field(default_factory=list, description="IDs of contradicted evidence instances")
    start_pos: int | None = Field(default=None, exclude=True, description="Internal start character offset")
    end_pos: int | None = Field(default=None, exclude=True, description="Internal end character offset")

    model_config = ConfigDict(from_attributes=True, extra="ignore")

    @field_validator("id", mode="before")
    @classmethod
    def sanitize_id(cls, v: Any) -> str:
        if v is None:
            return "FINDING_UNKNOWN"
        val = str(v).strip()
        return val if val else "FINDING_UNKNOWN"

    @field_validator("explanation", mode="before")
    @classmethod
    def sanitize_explanation(cls, v: Any) -> str:
        if v is None:
            return "No explanation provided"
        val = str(v).strip()
        return val if val else "No explanation provided"

    @field_validator("category", mode="before")
    @classmethod
    def sanitize_category(cls, v: Any) -> ScamCategory:
        valid = {e.value for e in ScamCategory}
        val = v.value if hasattr(v, "value") else str(v).strip().lower()
        if val in valid:
            return ScamCategory(val)
        logger.warning("Unknown ScamCategory '%s'; fallback to UNKNOWN", v)
        return ScamCategory.UNKNOWN

    @field_validator("severity", mode="before")
    @classmethod
    def sanitize_severity(cls, v: Any) -> FindingSeverity:
        valid = {e.value for e in FindingSeverity}
        val = v.value if hasattr(v, "value") else str(v).strip().upper()
        if val in valid:
            return FindingSeverity(val)
        return FindingSeverity.LOW

    @field_validator("status", mode="before")
    @classmethod
    def sanitize_status(cls, v: Any) -> FindingStatus:
        valid = {e.value for e in FindingStatus}
        val = v.value if hasattr(v, "value") else str(v).strip().upper()
        if val in valid:
            return FindingStatus(val)
        return FindingStatus.AMBIGUOUS_SIGNAL

    @field_validator("source", mode="before")
    @classmethod
    def sanitize_source(cls, v: Any) -> DetectionSource:
        valid = {e.value for e in DetectionSource}
        val = v.value if hasattr(v, "value") else str(v).strip().upper()
        if val in valid:
            return DetectionSource(val)
        logger.info("Unrecognized DetectionSource '%s'; mapped safely to UNKNOWN", v)
        return DetectionSource.UNKNOWN

    @field_validator("contradicts_finding_ids", mode="before")
    @classmethod
    def sanitize_contradicts_finding_ids(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if x and str(x).strip()]

    @classmethod
    def safe_parse(cls, data: Any) -> "EvidenceItem | None":
        """
        Safely instantiates an EvidenceItem from arbitrary data (dict or obj).
        Returns None if data cannot be coerced into a dictionary, preserving the
        caller from unexpected crashes while maintaining provenance integrity.
        """
        if not isinstance(data, dict):
            return None
        try:
            return cls.model_validate(data)
        except Exception as e:
            logger.warning("Failed to safely deserialize EvidenceItem: %s", e)
            return None


class ContentSecurityWarning(BaseModel):
    adversarial_text_detected: bool = False
    details: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class DetectionResult(BaseModel):
    rule_score: int
    categories_detected: list[ScamCategory]
    evidence_items: list[EvidenceItem]
    fallback_explanation: str
    evidence_completeness: EvidenceCompleteness
    content_security: ContentSecurityWarning = Field(default_factory=ContentSecurityWarning)

    model_config = ConfigDict(from_attributes=True, extra="ignore")

    @field_validator("evidence_items", mode="before")
    @classmethod
    def sanitize_evidence_items(cls, v: Any) -> list[Any]:
        if not isinstance(v, list):
            return []
        sanitized: list[Any] = []
        for item in v:
            if isinstance(item, EvidenceItem):
                sanitized.append(item)
            elif isinstance(item, dict):
                parsed = EvidenceItem.safe_parse(item)
                if parsed is not None:
                    sanitized.append(parsed)
            else:
                logger.warning("Discarded non-dictionary evidence item: %s", item)
        return sanitized

    @field_validator("categories_detected", mode="before")
    @classmethod
    def sanitize_categories(cls, v: Any) -> list[ScamCategory]:
        if not isinstance(v, list):
            return []
        valid = {e.value for e in ScamCategory}
        res: list[ScamCategory] = []
        for cat in v:
            val = cat.value if hasattr(cat, "value") else str(cat).strip().lower()
            if val in valid:
                res.append(ScamCategory(val))
            else:
                res.append(ScamCategory.UNKNOWN)
        return res
