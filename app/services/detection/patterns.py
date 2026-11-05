import re
from typing import NamedTuple

from app.services.detection.models import (
    DetectionSource,
    EvidenceItem,
    FindingSeverity,
    FindingStatus,
    ScamCategory,
)


class PatternDefinition(NamedTuple):
    pattern_id: str
    category: ScamCategory
    severity: FindingSeverity
    regex: re.Pattern
    explanation: str
    context_required: str | None = None
    default_status: FindingStatus = FindingStatus.EXPLICIT_CLAIM

    def to_evidence_item(
        self,
        matched_text: str | None = None,
        finding_id: str | None = None,
        start_pos: int | None = None,
        end_pos: int | None = None,
    ) -> EvidenceItem:
        return EvidenceItem(
            id=self.pattern_id,
            category=self.category,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=self.severity,
            status=self.default_status,
            matched_text=matched_text,
            explanation=self.explanation,
            context_required=self.context_required,
            finding_id=finding_id,
            start_pos=start_pos,
            end_pos=end_pos,
        )


# Finding IDs immune to text-based generic disclaimers
DISCLAIMER_IMMUNE_IDS: set[str] = {
    "ADV_FEE_PAYMENT_RAIL_DIRECT",  # Peer-to-peer VPA/UPI rails can never be excused
    "CRED_HARVEST_AUTH_SECRETS",    # OTP / PIN credential theft can never be excused
    "EQUIP_OVERPAY_CHECK_VENDOR",   # Cashier's check overpayment fraud can never be excused
    "ADV_FEE_SECURITY_DEPOSIT",     # Equipment/hardware deposits cannot be excused by recruitment disclaimer
    "ADV_FEE_REVERSE_AUTH",         # Deceptive reverse collect requests cannot be excused
}

# Pre-compiled regex for company-wide no-fee disclaimers
COMPANY_FEE_DISCLAIMER_RE = re.compile(
    r"\b(?:our\s+company|the\s+company|we|organization|employer|recruiter|firm)\s+"
    r"(?:does\s+not|do\s+not|never)\s+"
    r"(?:charge|asks?\s+(?:candidates?|applicants?|anyone)?\s*for|collect\s+(?:from\s+(?:candidates?|applicants?))?|demand|require|accept)\s+"
    r"(?:(?:candidates?|applicants?|job\s*seekers?|anyone)\s+)?"
    r"(?:any\s+)?"
    r"(?:(?:registration|recruitment|application|placement|processing|hiring|interview|selection|onboarding|upfront)\s*(?:(?:or|and|,)\s*)?){0,6}\s*"
    r"(?:fees?|charges?|money|deposits?|payments?)\b"
    r"|\b(?:our\s+company|the\s+company|we|organization|employer|recruiter|firm)\s+(?:does\s+not|do\s+not|never)\s+charge\s+(?:candidates?|applicants?|job\s*seekers?|anyone)\b"
    r"|\b(?:we\s+do\s+not|never)\s+(?:charge|ask\s+for|collect|demand)\s+(?:any\s+)?(?:fees?|charges?|money|deposits?)\s+(?:for|during|at|towards)\s+(?:recruitment|hiring|interviews?|selection|placement|jobs?|applications?)\b"
    r"|\b(?:we|our\s+company|the\s+company)\s+(?:do\s+not|does\s+not|never)\s+(?:authorize|permit|allow|employ)\s+(?:any\s+)?(?:agenc(?:y|ies)|consultanc(?:y|ies)|consultants?|agents?|third\s*part(?:y|ies))\s+(?:to\s+)?(?:charge|collect|demand|ask\s+for|take)\s+(?:any\s+)?(?:(?:registration|recruitment|placement|application|processing|onboarding)\s+)?(?:fees?|charges?|money|deposits?|payments?)\b"
    r"|\bno\s+(?:agenc(?:y|ies)|consultanc(?:y|ies)|consultants?|agents?|third\s*part(?:y|ies))\s+(?:is|are)\s+authorized\s+to\s+(?:charge|collect|demand|ask\s+for|accept)\s+(?:any\s+)?(?:(?:registration|recruitment|placement|application|processing|onboarding)\s+)?(?:fees?|charges?|money|payments?)\b"
    r"|\b(?:no|zero)\s+(?:registration|recruitment|application|placement|processing|hiring|interview)\s+(?:fees?|charges?)\s+(?:are|is)\s+(?:charged|collected|demanded|required|applicable)\b"
    r"|\b(?:recruitment|hiring|selection|application)\s+(?:process\s+)?is\s+(?:completely\s+|entirely\s+)?(?:free\s+of\s+(?:cost|charge)|free)\b",
    re.IGNORECASE,
)


# Pre-compiled category detection patterns (India + International)
DETECTION_PATTERNS: list[PatternDefinition] = [
    # -------------------------------------------------------------------------
    # 1. ADVANCE-FEE & DEPOSITS (advance_fee)
    # -------------------------------------------------------------------------
    PatternDefinition(
        pattern_id="ADV_FEE_REGISTRATION",
        category=ScamCategory.ADVANCE_FEE,
        severity=FindingSeverity.CRITICAL,
        regex=re.compile(
            # Explicit candidate payment demand for registration, application, onboarding, document, gate pass or uniform fee
            r"\b(?:pay|deposit|transfer|remit)\s+(?:(?:to\s+)?(?:our\s+company|the\s+company|us)\s+)?(?:(?:a|an|the)?\s*(?:non[-\s]refundable\s+)?)?(?:registration|recruitment|placement|application(?:\s+processing)?|onboarding|documentation|processing|gate\s+pass|uniform)\s+(?:fees?|charges?)\b"
            r"|\b(?:registration|recruitment|placement|application(?:\s+processing)?|onboarding|documentation|processing|gate\s+pass|uniform)\s+(?:fees?|charges?)\s*(?:of\s+(?:rs\.?|inr|\$|£|€)?\s*\d+(?:,\d+)*)?\s*(?:is\s+)?(?:required|mandatory|compulsory|payable|must\s+be\s+(?:paid|remitted|deposited|transferred))\b"
            r"|\b(?:candidates?|applicants?)\s+must\s+(?:pay|deposit|remit|submit)\s+(?:(?:to\s+)?(?:our\s+company|the\s+company|us)\s+)?(?:(?:a|an|the)?\s*)?(?:(?:non[-\s]refundable|mandatory|compulsory|official)\s+)?(?:registration|recruitment|placement|application(?:\s+processing)?|onboarding|documentation|processing)\s+(?:fee|charge)s?\s*(?:of\s+(?:rs\.?|inr|\$|£|€)?\s*\d+(?:,\d+)*)?\b"
            r"|\b(?:pay|deposit|transfer|remit)\s+(?:(?:rs\.?|inr|\$|£|€)\s*\d+(?:,\d+)*\s+)?(?:directly\s+)?(?:to\s+|into\s+)?(?:our\s+company|the\s+company|our\s+firm|the\s+firm|our\s+organization|us)\b"
            r"|\b(?:our\s+company|the\s+company|our\s+firm|our\s+organization|our\s+hr|we)\s+(?:collects?|charges?|requires?|demands?|asks?\s+for|takes?)\s+(?:(?:a|an|the|any)?\s*(?:non[-\s]refundable|mandatory|compulsory|upfront)?\s*)?(?:(?:rs\.?|inr|\$|£|€)?\s*\d+(?:,\d+)*\s+)?(?:registration|recruitment|placement|application(?:\s+processing)?|onboarding|documentation|processing)\s+(?:fees?|charges?)\b"
            r"|\b(?:pay|deposit|transfer|remit)\s+(?:(?:rs\.?|inr|\$|£|€)\s*\d+(?:,\d+)*\s+)?(?:to\s+)?(?:(?!pay\b|to\b)[a-zA-Z0-9_\-]+\s+){0,4}(?:agency|consultancy|placement\s+agency|staffing\s+firm|firm)\s*(?:(?:a|an|the)?\s*(?:non[-\s]refundable\s+)?(?:fee|charge)s?\s*(?:of\s+)?)?(?:(?:rs\.?|inr|\$|£|€)\s*\d+(?:,\d+)*)?\b"
            r"|\b(?:(?:our|the|[a-zA-Z0-9\.\-_]+)\s+)?(?:agency|consultancy|placement\s+agency|staffing\s+firm)\s+(?:collects?|charges?|demands?|requires?|takes?)\s+(?:(?:a|an|the|any)?\s*(?:non[-\s]refundable|mandatory|compulsory|upfront)?\s*)?(?:(?:rs\.?|inr|\$|£|€)?\s*\d+(?:,\d+)*\s+)?(?:registration|recruitment|placement|application(?:\s+processing)?|onboarding|documentation|processing)?\s*(?:fees?|charges?)\b",
            re.IGNORECASE,
        ),
        explanation="Explicit demand for candidate to pay upfront registration, document, or onboarding fees.",
    ),
    PatternDefinition(
        pattern_id="ADV_FEE_AMBIGUOUS_MENTION",
        category=ScamCategory.ADVANCE_FEE,
        severity=FindingSeverity.LOW,
        regex=re.compile(
            # Informational mention of fee without explicit extortion directive (e.g. public examination or licensing fee)
            r"\b(?:examination|exam|licensing|application|registration)\s+(?:fees?|charges?)\s+(?:is|of)\s+(?:rs\.?|inr|\$|£|€)?\s*\d+(?:,\d+)*\b",
            re.IGNORECASE,
        ),
        explanation="Informational mention of application or examination fee; context verification required to distinguish legitimate public-sector exam fees from scam demands.",
        context_required="Verify whether the fee is charged by an accredited public examination authority or licensing board.",
        default_status=FindingStatus.AMBIGUOUS_SIGNAL,
    ),
    PatternDefinition(
        pattern_id="ADV_FEE_SECURITY_DEPOSIT",
        category=ScamCategory.ADVANCE_FEE,
        severity=FindingSeverity.CRITICAL,
        regex=re.compile(
            r"\b(?:refundable\s+)?(?:security|laptop|training|kit)\s+deposits?\b"
            r"|\bdeposit\s+of\s+(?:rs\.?|inr|\$|£|€)?\s*\d+\s*(?:is\s+required|must\s+be\s+paid)\b",
            re.IGNORECASE,
        ),
        explanation="Demands upfront security or training deposit before beginning employment.",
    ),
    PatternDefinition(
        pattern_id="ADV_FEE_PAYMENT_RAIL_DIRECT",
        category=ScamCategory.ADVANCE_FEE,
        severity=FindingSeverity.CRITICAL,
        regex=re.compile(
            # Require explicit payment demand verb when directing funds to payment rails or peer-to-peer VPAs
            r"\b(?:pay|send|transfer|deposit|remit)\b(?:\s+\S+){0,7}\s+(?:via|to|through|into)\s+(?:(?:(?:upi(?:\s+id)?|qr\s*code|cashapp|zelle|gift\s*cards?|crypto)(?:\s+(?:is|at|to|:|id)?\s*)?)?[a-zA-Z0-9.\-_]{2,64}@(oksbi|okhdfcbank|okaxis|paytm|ybl|apl|postbank)|(?:upi(?:\s+id)?|qr\s*code|cashapp|zelle|gift\s*cards?|crypto))\b"
            r"|\b(?:pay|send|transfer|deposit|remit)\s+(?:(?:rs\.?|inr|\$|£|€)?\s*\d+(?:,\d+)*\s+)?(?:to|via|through)\s+[a-zA-Z0-9.\-_]{2,64}@(oksbi|okhdfcbank|okaxis|paytm|ybl|apl|postbank)\b"
            r"|\b(?:upi\s*id|vpa)\s*(?::|is|\-)\s*[a-zA-Z0-9.\-_]{2,64}@(oksbi|okhdfcbank|okaxis|paytm|ybl|apl|postbank)\s+(?:for|to\s+pay)\b",
            re.IGNORECASE,
        ),
        explanation="Direct payment demand via peer-to-peer UPI, QR code, or informal payment rail for employment.",
    ),
    PatternDefinition(
        pattern_id="ADV_FEE_REVERSE_AUTH",
        category=ScamCategory.ADVANCE_FEE,
        severity=FindingSeverity.CRITICAL,
        regex=re.compile(
            r"\b(?:approve|accept)\s+(?:(?:the|any|your)?\s*)?(?:upi\s+)?collect\s+requests?\s+(?:to\s+(?:receive|get|claim)\s+(?:money|payment|salary|funds?)|for\s+salary)\b"
            r"|\b(?:salary|funds?|payment)\s+(?:credit|transfer|receive)\s+karne\s+ke\s+liye\s+upi\s+collect\s+request\s+approve\s+karo\b",
            re.IGNORECASE,
        ),
        explanation="Deceptive reverse collect request instructing candidate to approve UPI debit to receive funds.",
    ),

    # -------------------------------------------------------------------------
    # 2. CREDENTIAL HARVESTING & SENSITIVE SECRETS (credential_harvesting)
    # -------------------------------------------------------------------------
    PatternDefinition(
        pattern_id="CRED_HARVEST_AUTH_SECRETS",
        category=ScamCategory.CREDENTIAL_HARVESTING,
        severity=FindingSeverity.CRITICAL,
        regex=re.compile(
            # Require disclosure context (enter/share/provide/submit) rather than standalone technical references (e.g. 2FA system design)
            r"\b(?:enter|share|provide|send|submit|disclose|give|input|tell\s+us)\s+(?:your\s+)?(?:upi\s*pin|bank\s*otp|sms\s*otp|otp|net\s*banking\s*password|debit\s*card\s*(?:pin|cvv)|atm\s*pin|banking\s*credentials|card\s*cvv)\b"
            r"|\b(?:upi\s*pin|bank\s*otp|sms\s*otp|net\s*banking\s*password|debit\s*card\s*(?:pin|cvv)|atm\s*pin)\s+(?:is\s+)?(?:required|needed|mandatory|must\s+be\s+(?:shared|entered|provided))\b"
            r"|\b(?:share|enter)\s+(?:the\s+)?otp\s+(?:(?:received\s+on|sent\s+to)\s+your\s+(?:phone|mobile|device|number)|for\s+salary\s+verification|to\s+claim)\b"
            r"|\benter\s+(?:your\s+)?upi\s*pin\s+to\s+receive\s+(?:salary|funds?|payment|bonus)\b",
            re.IGNORECASE,
        ),
        explanation="Requests banking authentication secrets, UPI PIN, or SMS OTP under the pretext of job verification.",
    ),
    PatternDefinition(
        pattern_id="CRED_HARVEST_PREMATURE_DOCS",
        category=ScamCategory.CREDENTIAL_HARVESTING,
        severity=FindingSeverity.HIGH,
        regex=re.compile(
            r"\b(?:send|upload|email|share)\s+(?:copy\s+of\s+)?(?:aadhaar|pan|ssn|social\s+security\s+number|passport)\s+(?:(?:card\s+)?(?:photo|scan|details|copy)|card)?\s*(?:before\s+interview|for\s+screening|immediately)\b"
            r"|\b(?:aadhaar|pan)\s+(?:card\s+)?(?:photo|scan)\s+required\s+(?:before\s+interview|for\s+pre-screening)\b",
            re.IGNORECASE,
        ),
        explanation="Demands government identity documents (Aadhaar, PAN, SSN) prematurely before an interview or vetting.",
    ),

    # -------------------------------------------------------------------------
    # 3. EQUIPMENT & CHEQUE OVERPAYMENT (equipment_overpayment)
    # -------------------------------------------------------------------------
    PatternDefinition(
        pattern_id="EQUIP_OVERPAY_CHECK_VENDOR",
        category=ScamCategory.EQUIPMENT_OVERPAYMENT,
        severity=FindingSeverity.CRITICAL,
        regex=re.compile(
            r"\b(?:cashier'?s?\s+check|cheque)\s+(?:for\s+home\s+office|will\s+be\s+sent|to\s+buy\s+equipment)\b"
            r"|\b(?:deposit\s+the\s+check|deposit\s+cheque)\s+and\s+(?:wire|send|transfer|refund)\s+(?:back\s+)?(?:the\s+)?(?:excess|balance|remaining)\b"
            r"|\b(?:purchase|buy)\s+(?:equipment|hardware|supplies)\s+from\s+(?:our\s+)?approved\s+vendor\b",
            re.IGNORECASE,
        ),
        explanation="Fake check or advance credit overpayment scheme instructing candidate to wire funds to a designated vendor.",
    ),

    # -------------------------------------------------------------------------
    # 4. TASK & OPTIMIZATION FRAUD (task_optimization)
    # -------------------------------------------------------------------------
    PatternDefinition(
        pattern_id="TASK_RATING_OPTIMIZATION_QUOTA",
        category=ScamCategory.TASK_OPTIMIZATION,
        severity=FindingSeverity.HIGH,
        regex=re.compile(
            r"\b(?:rating|reviewing)\s+(?:apps|hotels|products|maps)\b[^\n.!?]{0,80}\b(?:daily\s+tasks?|daily\s+commissions?|payouts?)\b"
            r"|\bcomplete\s+\d+\s+(?:sets?\s+of\s+)?(?:tasks?|orders?)\s+daily\b"
            r"|\b(?:like|subscribe)\s+(?:youtube\s+videos?|channels?)\s+(?:and\s+earn|earn\s+rs\.?)\b",
            re.IGNORECASE,
        ),
        explanation="Gamified task loop requiring repetitive app reviews, video likes, or daily task quotas for commissions.",
    ),
    PatternDefinition(
        pattern_id="TASK_CRYPTO_VIP_COMMISSION",
        category=ScamCategory.TASK_OPTIMIZATION,
        severity=FindingSeverity.HIGH,
        regex=re.compile(
            r"\b(?:usdt|crypto\s*wallet|cryptocurrency)\s+(?:payout|commission|withdrawal|account)\b"
            r"|\b(?:vip\s+level|unlock\s+vip\s+tier|prepaid\s+task\s+order)\b",
            re.IGNORECASE,
        ),
        explanation="Structured prepaid commission task scheme promising daily USDT/crypto payouts.",
    ),

    # -------------------------------------------------------------------------
    # 5. FAKE RECRUITMENT & GUARANTEED PLACEMENT (fake_recruitment)
    # -------------------------------------------------------------------------
    PatternDefinition(
        pattern_id="FAKE_RECRUIT_GUARANTEED_JOB",
        category=ScamCategory.FAKE_RECRUITMENT,
        severity=FindingSeverity.HIGH,
        regex=re.compile(
            # Qualify "no interview" to prevent flagging disclaimers like "no interview fee"
            r"\b(?:100%|direct)\s+guaranteed\s+(?:job|placement|joining|selection)\b"
            r"|\b(?:direct\s+appointment\s+letter|direct\s+joining)\s+without\s+(?:any\s+)?(?:interview|test|screening)\b"
            r"|\bno\s+interview\s+(?:required|needed|conducted|necessary|scheduled)\b"
            r"|\b(?:direct\s+selection|direct\s+hiring|appointment\s+letter)\s+(?:with\s+)?no\s+interview\b",
            re.IGNORECASE,
        ),
        explanation="Promises unconditional employment guarantees or direct appointment letters without vetting.",
    ),

    # -------------------------------------------------------------------------
    # 6. SUSPICIOUS COMMUNICATION (suspicious_communication)
    # -------------------------------------------------------------------------
    PatternDefinition(
        pattern_id="COMM_TELEGRAM_ONLY_RECRUITMENT",
        category=ScamCategory.SUSPICIOUS_COMMUNICATION,
        severity=FindingSeverity.MEDIUM,
        regex=re.compile(
            r"\b(?:contact|message|reach\s+out\s+to)\s+(?:hr|hiring\s+manager|recruiter)?\s*(?:only\s+)?on\s+telegram\s*@\w+\b"
            r"|\btelegram\s*:\s*@\w+\b",
            re.IGNORECASE,
        ),
        explanation="Recruiting directs applicants exclusively to off-platform Telegram messaging handles.",
        context_required="Verify whether the business maintains an official verified corporate communication channel.",
    ),
    PatternDefinition(
        pattern_id="COMM_WHATSAPP_ONLY_RECRUITMENT",
        category=ScamCategory.SUSPICIOUS_COMMUNICATION,
        severity=FindingSeverity.LOW,
        regex=re.compile(
            r"\bwhatsapp\s+only\b"
            r"|\bcontact\s+(?:via\s+)?whatsapp\s+only\b",
            re.IGNORECASE,
        ),
        explanation="Informal contact method relying exclusively on WhatsApp messaging.",
        context_required="Informal messaging is common in local small businesses; evaluate in combination with other signals.",
    ),

    # -------------------------------------------------------------------------
    # 7. SALARY MANIPULATION (salary_manipulation)
    # -------------------------------------------------------------------------
    PatternDefinition(
        pattern_id="SALARY_UNSKILLED_EXORBITANT",
        category=ScamCategory.SALARY_MANIPULATION,
        severity=FindingSeverity.MEDIUM,
        regex=re.compile(
            r"\bearn\s+(?:rs\.?|inr|\$)\s*\d{3,6}\s*(?:daily|per\s+day)\s+for\s+(?:simple\s+typing|30\s+mins?|1\s+hour|copy\s+paste)\b"
            r"|\bearn\s+\$\s*1[0-9]{3,4}\s*(?:daily|per\s+day)\b",
            re.IGNORECASE,
        ),
        explanation="Extreme disparity between compensation claims and minimal, non-specialized workload.",
    ),
]
