from datetime import datetime
from enum import Enum
from typing import Any, NamedTuple
import logging
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

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
    SUSPICIOUS = "MEDIUM"          # Backward-compatibility alias for lower-medium severity
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


class CandidateEligibility(str, Enum):
    SCORABLE_ACTIVE = "SCORABLE_ACTIVE"
    SCORABLE_AMBIGUOUS = "SCORABLE_AMBIGUOUS"
    NON_SCORABLE_INFORMATIONAL = "NON_SCORABLE_INFORMATIONAL"
    NON_SCORABLE_CONTEXTUAL = "NON_SCORABLE_CONTEXTUAL"
    REJECTED = "REJECTED"


class RejectionReason(str, Enum):
    INTERVAL_INVALID = "INTERVAL_INVALID"
    INTERVAL_EMPTY = "INTERVAL_EMPTY"
    BOUNDS_OUT_OF_RANGE = "BOUNDS_OUT_OF_RANGE"
    SEGMENT_BOUNDARY_VIOLATED = "SEGMENT_BOUNDARY_VIOLATED"
    QUOTATION_MISMATCH = "QUOTATION_MISMATCH"
    QUOTATION_EMPTY_OR_WHITESPACE = "QUOTATION_EMPTY_OR_WHITESPACE"
    QUOTATION_TOO_LONG = "QUOTATION_TOO_LONG"
    NORMALIZED_MISMATCH = "NORMALIZED_MISMATCH"
    DESCRIPTIVE_NOT_DIRECTIVE = "DESCRIPTIVE_NOT_DIRECTIVE"
    HISTORICAL_REPORT = "HISTORICAL_REPORT"
    DISCRETIONARY_NOT_MANDATORY = "DISCRETIONARY_NOT_MANDATORY"
    NON_FINANCIAL_ACTION = "NON_FINANCIAL_ACTION"
    COMPETING_OBJECT_BOUND = "COMPETING_OBJECT_BOUND"
    SUBORDINATE_CLAUSE_ISOLATED = "SUBORDINATE_CLAUSE_ISOLATED"
    SELF_AUTH_NOT_DISCLOSURE = "SELF_AUTH_NOT_DISCLOSURE"
    NO_RECRUITMENT_CONTEXT = "NO_RECRUITMENT_CONTEXT"
    NO_GOVERNING_VERB = "NO_GOVERNING_VERB"
    UNVERIFIED_PROVENANCE = "UNVERIFIED_PROVENANCE"
    CONTRADICTORY_PROVENANCE = "CONTRADICTORY_PROVENANCE"
    UNTRUSTED_SOURCE = "UNTRUSTED_SOURCE"
    UNSUPPORTED_CATEGORY = "UNSUPPORTED_CATEGORY"
    NON_ACTIONABLE_SEVERITY = "NON_ACTIONABLE_SEVERITY"


class TextSegment(NamedTuple):
    text: str
    source_intervals: list[tuple[int, int]]
    raw_start: int
    raw_end: int
    segment_index: int
    script: str


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
    eligibility: CandidateEligibility = Field(
        default=CandidateEligibility.REJECTED,
        exclude=True,
        description="Internal scoring eligibility decision",
    )
    rejection_reason: RejectionReason | None = Field(
        default=None,
        exclude=True,
        description="Internal provenance or quotation rejection reason",
    )

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

    @field_validator("eligibility", mode="before")
    @classmethod
    def sanitize_eligibility(cls, v: Any) -> CandidateEligibility:
        valid = {e.value for e in CandidateEligibility}
        val = v.value if hasattr(v, "value") else str(v).strip().upper()
        if val in valid:
            return CandidateEligibility(val)
        return CandidateEligibility.REJECTED

    @model_validator(mode="after")
    def _validate_eligibility_and_provenance(self) -> "EvidenceItem":
        # 1. Capture whether caller explicitly passed eligibility BEFORE any mutation
        was_explicitly_set = "eligibility" in self.model_fields_set

        # 2. If caller explicitly passed a non-scorable state (or explicit None / invalid string), preserve rejection
        if was_explicitly_set:
            if self.eligibility in (
                CandidateEligibility.REJECTED,
                CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
                CandidateEligibility.NON_SCORABLE_CONTEXTUAL,
            ):
                return self

        # 3. Provenance completeness check for scorable eligibility
        required_provenance = {"category", "source", "severity", "status"}
        if not required_provenance.issubset(self.model_fields_set):
            self.eligibility = CandidateEligibility.REJECTED
            self.rejection_reason = RejectionReason.UNVERIFIED_PROVENANCE
            return self

        # 4. Provenance validity checks
        active_severities = {
            FindingSeverity.SUSPICIOUS,
            FindingSeverity.HIGH,
            FindingSeverity.CRITICAL,
        }
        trusted_sources = {
            DetectionSource.DETERMINISTIC_RULE,
            DetectionSource.RULE_AND_AI,
            DetectionSource.AI_INFERENCE,
        }

        # Branch A: SCORABLE_ACTIVE (Strictly active severities & explicit claims)
        if (
            self.category != ScamCategory.UNKNOWN
            and self.source in trusted_sources
            and self.severity in active_severities
            and self.status == FindingStatus.EXPLICIT_CLAIM
        ):
            if was_explicitly_set:
                if self.eligibility == CandidateEligibility.SCORABLE_ACTIVE:
                    self.rejection_reason = None
                    return self
                else:
                    self.eligibility = CandidateEligibility.REJECTED
                    self.rejection_reason = RejectionReason.CONTRADICTORY_PROVENANCE
                    return self
            self.eligibility = CandidateEligibility.SCORABLE_ACTIVE
            self.rejection_reason = None
            return self

        # Branch B: SCORABLE_AMBIGUOUS (Independent branches: Active Ambiguous vs. Low Ambiguous)
        is_active_ambiguous = (
            self.category != ScamCategory.UNKNOWN
            and self.severity in active_severities
            and (self.status == FindingStatus.AMBIGUOUS_SIGNAL or self.source == DetectionSource.UNKNOWN)
        )
        is_low_ambiguous = (
            self.category != ScamCategory.UNKNOWN
            and self.source in trusted_sources
            and self.severity == FindingSeverity.LOW
            and self.status == FindingStatus.AMBIGUOUS_SIGNAL
        )

        if is_active_ambiguous or is_low_ambiguous:
            if was_explicitly_set:
                if self.eligibility == CandidateEligibility.SCORABLE_AMBIGUOUS:
                    self.rejection_reason = None
                    return self
                else:
                    self.eligibility = CandidateEligibility.REJECTED
                    self.rejection_reason = RejectionReason.CONTRADICTORY_PROVENANCE
                    return self
            self.eligibility = CandidateEligibility.SCORABLE_AMBIGUOUS
            self.rejection_reason = None
            return self

        # Fail-closed rejection for all other combinations
        self.eligibility = CandidateEligibility.REJECTED
        if self.category == ScamCategory.UNKNOWN:
            self.rejection_reason = RejectionReason.UNSUPPORTED_CATEGORY
        elif self.source not in trusted_sources:
            self.rejection_reason = RejectionReason.UNTRUSTED_SOURCE
        elif self.severity not in active_severities and self.severity != FindingSeverity.LOW:
            self.rejection_reason = RejectionReason.NON_ACTIONABLE_SEVERITY
        else:
            self.rejection_reason = RejectionReason.UNVERIFIED_PROVENANCE
        return self

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
