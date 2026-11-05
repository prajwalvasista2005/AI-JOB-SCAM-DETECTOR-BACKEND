"""
Semantic Qualification Layer (Fix 2 V5.2 Specification).
Implements pure semantic qualification decoupled from candidate emission:
  1. Tokenization and lexical tagging with raw source coordinates.
  2. Bounded semantic association with action-scoped governors and sub-clause isolation.
  3. Pure semantic qualification with fail-safe outcome mapping.
  4. Candidate emission with tightly bounded spans.
  5. Category-preserving advisory classification.
"""
from enum import Enum
import logging
import re
from typing import Any, NamedTuple

from app.services.detection.models import (
    CandidateEligibility,
    DetectionSource,
    EvidenceItem,
    FindingSeverity,
    FindingStatus,
    RejectionReason,
    ScamCategory,
    TextSegment,
)

logger = logging.getLogger(__name__)


# Lexical Tag Taxonomy
TAG_FEE_NOUN = "TAG_FEE_NOUN"
TAG_CRED_NOUN = "TAG_CRED_NOUN"
TAG_REV_NOUN = "TAG_REV_NOUN"
TAG_PAY_VERB = "TAG_PAY_VERB"
TAG_CRED_VERB = "TAG_CRED_VERB"
TAG_REV_VERB = "TAG_REV_VERB"
TAG_OBLIGATION_MANDATORY = "TAG_OBLIGATION_MANDATORY"
TAG_OBLIGATION_OPTIONAL = "TAG_OBLIGATION_OPTIONAL"
TAG_HISTORICAL_MARKER = "TAG_HISTORICAL_MARKER"
TAG_AMOUNT = "TAG_AMOUNT"
TAG_PRECOND = "TAG_PRECOND"
TAG_SUBJECT_CANDIDATE = "TAG_SUBJECT_CANDIDATE"
TAG_RECIPIENT_AGENT = "TAG_RECIPIENT_AGENT"
TAG_POSSESSIVE = "TAG_POSSESSIVE"
TAG_PROHIBITIVE = "TAG_PROHIBITIVE"
TAG_DESCRIPTIVE = "TAG_DESCRIPTIVE"
TAG_SELF_AUTH_MARKER = "TAG_SELF_AUTH_MARKER"
TAG_NON_FINANCIAL_OBJ = "TAG_NON_FINANCIAL_OBJ"
TAG_UNRELATED_FIN_OBJ = "TAG_UNRELATED_FIN_OBJ"
TAG_CLAUSE_BOUNDARY = "TAG_CLAUSE_BOUNDARY"
TAG_ADVERSATIVE_CONJ = "TAG_ADVERSATIVE_CONJ"
TAG_SUBORDINATE_CONJ = "TAG_SUBORDINATE_CONJ"

# Extended Actor and Action Tag Taxonomy (Step 3.1)
TAG_ACTOR_CANDIDATE = "TAG_ACTOR_CANDIDATE"
TAG_ACTOR_EMPLOYER = "TAG_ACTOR_EMPLOYER"
TAG_ACTOR_RECRUITER = "TAG_ACTOR_RECRUITER"
TAG_ACTOR_AGENCY = "TAG_ACTOR_AGENCY"
TAG_ACTOR_GOVERNMENT = "TAG_ACTOR_GOVERNMENT"
TAG_ACTOR_SCAMMER = "TAG_ACTOR_SCAMMER"
TAG_ACTOR_THIRD_PARTY = "TAG_ACTOR_THIRD_PARTY"
TAG_VERB_REPORTING = "TAG_VERB_REPORTING"
TAG_VERB_DENIAL = "TAG_VERB_DENIAL"
TAG_ACTION_NON_FINANCIAL = "TAG_ACTION_NON_FINANCIAL"
TAG_CONDITIONAL = "TAG_CONDITIONAL"


class ActorRole(str, Enum):
    CANDIDATE = "CANDIDATE"          # Job applicant / job seeker
    EMPLOYER = "EMPLOYER"            # Official hiring company / HR / organization
    RECRUITER = "RECRUITER"          # Individual recruiter / talent representative
    AGENCY = "AGENCY"                # Third-party staffing firm / placement agency
    GOVERNMENT = "GOVERNMENT"        # Public examination body / licensing authority
    SCAMMER = "SCAMMER"              # Fraudster / imposter
    THIRD_PARTY = "THIRD_PARTY"      # Unaffiliated third party / external individual
    UNKNOWN = "UNKNOWN"              # Unspecified / unattributed entity (conservative default)


class Token(NamedTuple):
    index: int
    text: str
    normalized_text: str
    local_start: int
    local_end: int
    raw_start: int
    raw_end: int
    tags: set[str]
    is_hard_boundary: bool
    is_subclause_boundary: bool


class SubClause(NamedTuple):
    tokens: list[Token]
    start_pos: int
    end_pos: int
    has_adversative: bool

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]

    def index(self, value, *args):
        return self.tokens.index(value, *args)

    def count(self, value):
        return self.tokens.count(value)

    def __contains__(self, item):
        return item in self.tokens


class EpistemicFrame(str, Enum):
    DIRECT_INSTRUCTION = "DIRECT_INSTRUCTION"
    SCAM_WARNING = "SCAM_WARNING"
    REPORTED_CLAIM = "REPORTED_CLAIM"
    DENIAL_REPUDIATION = "DENIAL_REPUDIATION"
    DESCRIPTIVE_FACT = "DESCRIPTIVE_FACT"
    HYPOTHETICAL = "HYPOTHETICAL"
    UNKNOWN = "UNKNOWN"


class ModalityType(str, Enum):
    MANDATORY = "MANDATORY"
    OPTIONAL = "OPTIONAL"
    PROHIBITED = "PROHIBITED"
    NOT_REQUIRED = "NOT_REQUIRED"
    CONDITIONAL = "CONDITIONAL"
    HYPOTHETICAL = "HYPOTHETICAL"
    NEGATED_ACTION = "NEGATED_ACTION"
    DECLARATIVE = "DECLARATIVE"
    UNKNOWN = "UNKNOWN"


class PredicateAssociation(NamedTuple):
    governing_verb: Token
    anchor_noun: Token | None
    instruction_source: ActorRole
    action_performer: ActorRole
    collection_actor: ActorRole
    payment_recipient: ActorRole
    affected_patient: ActorRole
    information_recipient: ActorRole
    reporting_source: ActorRole
    claim_subject: ActorRole
    epistemic_frame: EpistemicFrame
    modality: ModalityType
    participating_tokens: list[Token]
    raw_start: int
    raw_end: int
    obligation_token: Token | None = None
    negation_token: Token | None = None
    descriptive_token: Token | None = None
    amount_token: Token | None = None
    subject_token: Token | None = None
    recipient_token: Token | None = None
    precond_token: Token | None = None
    self_auth_token: Token | None = None
    is_negated: bool = False
    is_descriptive: bool = False
    is_historical: bool = False
    is_optional: bool = False
    has_self_auth: bool = False
    association_type: str = "FEE_DEMAND"


class SemanticAssociation(NamedTuple):
    association_type: str  # "FEE_DEMAND", "CRED_DISCLOSE", "REV_AUTH"
    anchor_noun: Token
    governing_verb: Token
    obligation_token: Token | None
    negation_token: Token | None
    descriptive_token: Token | None
    amount_token: Token | None
    subject_token: Token | None
    recipient_token: Token | None
    precond_token: Token | None
    self_auth_token: Token | None
    participating_tokens: list[Token]
    is_negated: bool
    is_descriptive: bool
    is_historical: bool
    is_optional: bool
    has_self_auth: bool


class QualificationDecision(NamedTuple):
    is_qualified: bool
    pattern_id: str | None
    category: ScamCategory | None
    severity: FindingSeverity | None
    eligibility: CandidateEligibility | None
    status: FindingStatus | None
    explanation: str | None
    semantic_target: str | None = None
    action_type: str | None = None
    rejection_reason: RejectionReason | None = None
    raw_start: int = 0
    raw_end: int = 0


class SemanticMatchResult(NamedTuple):
    pattern_id: str
    category: ScamCategory
    severity: FindingSeverity
    local_start: int
    local_end: int
    raw_start: int
    raw_end: int
    explanation: str
    semantic_target: str
    action_type: str
    eligibility: CandidateEligibility
    status: FindingStatus
    participating_token_indices: list[int]


# Alias for compatibility with V5.2 design spec naming
HinglishMatchResult = SemanticMatchResult


MASTER_TOKEN_RE = re.compile(
    r"(?P<HARD_BOUNDARY>[.!?;:\n]+)"
    r"|(?P<ADVERSATIVE>\b(?:but|however|although|whereas|instead|yet|while|lekin|magar|parantu|kintu)\b)"
    r"|(?P<SUBORDINATE>\b(?:how|that|because|where|when|if|kyunki|ki|agar)\b)"
    r"|(?P<COORDINATING_CLAUSE>\b(?:and|aur)\b(?=\s+(?:(?:(?:the|our)\s+)?(?:candidates?|applicants?|job\s*seekers?|you|we|our\s+company|the\s+company|agency|consultancy|aap)\s+(?:must|shall|need\s+to|should|have\s+to|are\s+required\s+to|will|collects?|charges?|ko)|(?:to\s+)?(?:pay|jama|dena|de|transfer|deposit|remit|charge|collect|demand|submit|share|enter|provide|disclose|approve|accept|learn))\b))"
    r"|(?P<AMOUNT>(?:(?:rs\.?|inr|₹|\$)\s*\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?\s*(?:rupaye|rupees|rs|inr)))"
    r"|(?P<COMMA>,)"
    r"|(?P<PRECOND>\b(?:joining|selection|interview)\s+se\s+pehle\b)"
    r"|(?P<SUBJECT_CANDIDATE>\b(?:candidates?|applicants?|job\s*seekers?|aap)\s*(?:ko)?\b)"
    r"|(?P<ACTOR_SCAMMER>\b(?:scammers?|fraudsters?|impost[eo]rs?|fake\s+agents?)\b)"
    r"|(?P<ACTOR_EMPLOYER>\b(?:(?:our|the)\s+company|(?:our|the)\s+firm|company|employer|organization)\b)"
    r"|(?P<ACTOR_AGENCY>\b(?:(?:the\s+)?(?:agency|consultancy)|(?:third[- ]party|placement|staffing|recruitment)\s+agency)\b)"
    r"|(?P<ACTOR_RECRUITER>\b(?:recruiters?|hr|hiring\s+managers?|talent\s+acquisition)\b)"
    r"|(?P<ACTOR_GOVERNMENT>\b(?:government|govt|examination\s+body|licensing\s+authority|public\s+authority)\b)"
    r"|(?P<ACTOR_THIRD_PARTY>\b(?:third[- ]part(?:y|ies)|external\s+vendors?)\b)"
    r"|(?P<RECIPIENT_AGENT>\b(?:(?:to\s+(?:our\s+)?)?(?:hr|recruiter|fake\s+agent|agent)(?:\s+ko)?)\b)"
    r"|(?P<PROHIBITIVE_VERB>\b(?:mat\s+(?:do|dena|bhejna|karo)|never\s+share|do\s+not\s+(?:pay|share))\b)"
    r"|(?P<PROHIBITIVE_MARKER>\b(?:do\s+not|don't|never|mat|koi\s+bhi|nahi|bilkul\s+free)\b)"
    r"|(?P<DESCRIPTIVE>\b(?:is\s+used\s+to|used\s+to|available\s+online|hone\s+ka\s+wait\s+karein|wait\s+karein)\b)"
    r"|(?P<SELF_AUTH>\b(?:in\s+your\s+banking\s+app|your\s+own\s+upi\s+payment|your\s+own\s+payment)\b)"
    r"|(?P<REV_NOUN>\b(?:upi\s+collect\s+request|collect\s+request|payment\s+request)\b)"
    r"|(?P<CRED_NOUN>\b(?:upi\s*pin|bank\s*otp|sms\s*otp|net\s*banking\s*password|atm\s*pin|card\s*cvv)\b)"
    r"|(?P<FEE_NOUN>\b(?:security\s+deposit|gate\s+pass|registration\s+(?:fees?|charges?)|recruitment\s+(?:fees?|charges?)|placement\s+(?:fees?|charges?)|application\s+(?:fees?|charges?)|onboarding\s+(?:fees?|charges?)|documentation\s+(?:fees?|charges?)|processing\s+(?:fees?|charges?)|examination\s+(?:fees?|charges?)|exam\s+(?:fees?|charges?)|government\s+examination\s+fees?|uniform\s+(?:charges?|fees?)|registration|recruitment|application|शुल्क|shulk|kharcha)\b)"
    r"|(?P<PAY_VERB_COMPOUND>\b(?:jama\s+(?:karna\s+(?:padega|hoga)|karo|kijiye|karein)|pay\s+karna\s+(?:padega|hoga)|dena\s+(?:padega|hoga)|de\s+do|dijiye)\b)"
    r"|(?P<CRED_VERB_COMPOUND>\b(?:share\s+(?:karna\s+(?:padega|hoga)|karo|kijiye|karein)|enter\s+(?:karna\s+(?:padega|hoga)|karo|kijiye|karein))\b)"
    r"|(?P<REV_VERB_COMPOUND>\b(?:approve\s+(?:karna\s+(?:padega|hoga)|karo|kijiye)|accept\s+(?:karna\s+(?:padega|hoga)|karo|kijiye))\b)"
    r"|(?P<VERB_REPORTING>\b(?:said|says|claims?|claimed|stated?|states|alleged?|alleges|reported?|reports)\b)"
    r"|(?P<VERB_DENIAL>\b(?:denies|denied|refutes?|refuted|rejects?|rejected|repudiates?|repudiated|clarif(?:ies|ied))\b)"
    r"|(?P<ACTION_NON_FINANCIAL>\b(?:register(?:ing)?|registrations?|submit(?:ting)?|submits?|upload(?:ing)?|uploads?|apply|applying)\b)"
    r"|(?P<CONDITIONAL_MARKER>\b(?:unless|except|subject\s+to|provided\s+that)\b)"
    r"|(?P<HISTORICAL>\b(?:previously|already|had\s+paid|pehle\s+jama\s+kiya\s+tha)\b)"
    r"|(?P<OPTIONAL>\b(?:may\s+vary|can\s+pay|optional|voluntarily|voluntary)\b)"
    r"|(?P<NON_FINANCIAL_OBJ>\b(?:attention|respects?|visit)\b)"
    r"|(?P<UNRELATED_FIN_OBJ>\b(?:salaries|salary|taxes?)\b)"
    r"|(?P<WORD>[\w\u0900-\u097F]+)",
    re.IGNORECASE,
)


def tokenize_and_tag(
    segment: str | TextSegment,
    source_intervals: list[tuple[int, int]] | None = None,
    max_tokens: int = 200,
) -> list[Token]:
    """
    Stage 1: Tokenization and Lexical Tagging with Raw Source Coordinates.
    Extracts multi-word phrases, amounts, punctuation boundaries, and vocabulary tags.
    Preserves exact character coordinates mapping back to raw content.
    Caps output at max_tokens (default: 200).
    """
    try:
        if isinstance(segment, TextSegment):
            segment_text = segment.text
            if source_intervals is None:
                source_intervals = segment.source_intervals
        else:
            segment_text = str(segment)

        if source_intervals is None:
            source_intervals = [(i, i + 1) for i in range(len(segment_text))]

        tokens: list[Token] = []
        idx = 0
        for m in MASTER_TOKEN_RE.finditer(segment_text):
            if len(tokens) >= max_tokens:
                logger.warning(
                    "Token candidate cap reached (%d tokens); truncating further tokens for segment.",
                    max_tokens,
                )
                break

            raw_text = m.group(0)
            norm_text = raw_text.lower().strip()
            start, end = m.start(), m.end()

            # Bounds safety: clamp to valid source intervals
            raw_s = source_intervals[start][0] if start < len(source_intervals) else start
            raw_e = source_intervals[end - 1][1] if (end - 1) < len(source_intervals) else end

            tags: set[str] = set()
            is_hard = False
            is_sub = False

            kind = m.lastgroup
            if kind == "HARD_BOUNDARY":
                tags.add(TAG_CLAUSE_BOUNDARY)
                is_hard = True
            elif kind == "ADVERSATIVE":
                tags.add(TAG_ADVERSATIVE_CONJ)
                is_sub = True
            elif kind == "SUBORDINATE":
                tags.add(TAG_SUBORDINATE_CONJ)
                is_sub = True
            elif kind == "COORDINATING_CLAUSE":
                tags.add(TAG_CLAUSE_BOUNDARY)
                is_sub = True
            elif kind == "COMMA":
                tags.add(TAG_CLAUSE_BOUNDARY)
                is_sub = True
            elif kind == "AMOUNT":
                tags.add(TAG_AMOUNT)
            elif kind == "PRECOND":
                tags.add(TAG_PRECOND)
            elif kind == "SUBJECT_CANDIDATE":
                tags.add(TAG_SUBJECT_CANDIDATE)
                tags.add(TAG_ACTOR_CANDIDATE)
            elif kind == "ACTOR_SCAMMER":
                tags.add(TAG_ACTOR_SCAMMER)
                tags.add(TAG_RECIPIENT_AGENT)
            elif kind == "ACTOR_EMPLOYER":
                tags.add(TAG_ACTOR_EMPLOYER)
            elif kind == "ACTOR_AGENCY":
                tags.add(TAG_ACTOR_AGENCY)
                tags.add(TAG_RECIPIENT_AGENT)
            elif kind == "ACTOR_RECRUITER":
                tags.add(TAG_ACTOR_RECRUITER)
                tags.add(TAG_RECIPIENT_AGENT)
            elif kind == "ACTOR_GOVERNMENT":
                tags.add(TAG_ACTOR_GOVERNMENT)
            elif kind == "ACTOR_THIRD_PARTY":
                tags.add(TAG_ACTOR_THIRD_PARTY)
            elif kind == "RECIPIENT_AGENT":
                tags.add(TAG_RECIPIENT_AGENT)
                if "fake" in norm_text or "scam" in norm_text:
                    tags.add(TAG_ACTOR_SCAMMER)
                elif "agency" in norm_text or "consultan" in norm_text:
                    tags.add(TAG_ACTOR_AGENCY)
                elif "company" in norm_text or "firm" in norm_text:
                    tags.add(TAG_ACTOR_EMPLOYER)
                elif "recruiter" in norm_text or "hr" in norm_text:
                    tags.add(TAG_ACTOR_RECRUITER)
            elif kind == "VERB_REPORTING":
                tags.add(TAG_VERB_REPORTING)
            elif kind == "VERB_DENIAL":
                tags.add(TAG_VERB_DENIAL)
            elif kind == "PROHIBITIVE_VERB":
                tags.add(TAG_PROHIBITIVE)
                if "pay" in norm_text or "do" in norm_text or "dena" in norm_text:
                    tags.add(TAG_PAY_VERB)
                if "share" in norm_text or "bhejna" in norm_text:
                    tags.add(TAG_CRED_VERB)
            elif kind == "PROHIBITIVE_MARKER":
                tags.add(TAG_PROHIBITIVE)
            elif kind == "DESCRIPTIVE":
                tags.add(TAG_DESCRIPTIVE)
            elif kind == "SELF_AUTH":
                tags.add(TAG_SELF_AUTH_MARKER)
            elif kind == "FEE_NOUN":
                tags.add(TAG_FEE_NOUN)
            elif kind == "CRED_NOUN":
                tags.add(TAG_CRED_NOUN)
            elif kind == "REV_NOUN":
                tags.add(TAG_REV_NOUN)
            elif kind == "PAY_VERB_COMPOUND":
                tags.add(TAG_PAY_VERB)
                tags.add(TAG_OBLIGATION_MANDATORY)
            elif kind == "CRED_VERB_COMPOUND":
                tags.add(TAG_CRED_VERB)
                tags.add(TAG_OBLIGATION_MANDATORY)
            elif kind == "REV_VERB_COMPOUND":
                tags.add(TAG_REV_VERB)
                tags.add(TAG_OBLIGATION_MANDATORY)
            elif kind == "HISTORICAL":
                tags.add(TAG_HISTORICAL_MARKER)
            elif kind == "OPTIONAL":
                tags.add(TAG_OBLIGATION_OPTIONAL)
            elif kind == "ACTION_NON_FINANCIAL":
                tags.add(TAG_ACTION_NON_FINANCIAL)
            elif kind == "CONDITIONAL_MARKER":
                tags.add(TAG_CONDITIONAL)
            elif kind == "NON_FINANCIAL_OBJ":
                tags.add(TAG_NON_FINANCIAL_OBJ)
            elif kind == "UNRELATED_FIN_OBJ":
                tags.add(TAG_UNRELATED_FIN_OBJ)
            elif kind == "WORD":
                if norm_text in {
                    "pay", "paying", "jama", "dena", "de", "dijiye", "bhejo",
                    "transfer", "transferring", "deposit", "depositing", "remit", "remitting",
                    "charge", "charges", "charging", "collect", "collects", "collecting",
                    "demand", "demands", "demanding", "require", "requires"
                }:
                    tags.add(TAG_PAY_VERB)
                if norm_text in {"share", "enter", "batao", "bhejo", "send", "de", "dijiye", "provide", "disclose", "submit"}:
                    tags.add(TAG_CRED_VERB)
                if norm_text in {"register", "submitting", "submit", "apply", "upload"}:
                    tags.add(TAG_ACTION_NON_FINANCIAL)
                if norm_text in {"unless", "except"}:
                    tags.add(TAG_CONDITIONAL)
                if norm_text in {"approve", "accept"}:
                    tags.add(TAG_REV_VERB)
                if norm_text in {"must", "required", "mandatory", "shall", "have to", "padega", "hoga", "karo", "kijiye", "compulsory", "need"}:
                    tags.add(TAG_OBLIGATION_MANDATORY)
                if norm_text in {"may", "can", "optional", "voluntarily", "voluntary"}:
                    tags.add(TAG_OBLIGATION_OPTIONAL)
                if norm_text in {"previously", "already", "paid"}:
                    tags.add(TAG_HISTORICAL_MARKER)
                if norm_text in {"not", "never", "mat", "nahi", "no", "zero", "without"}:
                    tags.add(TAG_PROHIBITIVE)
                if norm_text in {"fees", "fee", "shulk", "शुल्क", "deposit", "deposits", "payment", "payments"}:
                    tags.add(TAG_FEE_NOUN)
                if norm_text in {"apna", "apne", "apni", "your"}:
                    tags.add(TAG_POSSESSIVE)
                if norm_text in {"scammer", "scammers", "fraudster", "fraudsters", "imposter", "imposters", "impostor", "impostors"}:
                    tags.add(TAG_ACTOR_SCAMMER)
                    tags.add(TAG_RECIPIENT_AGENT)
                if norm_text in {"company", "employer", "organization", "firm"}:
                    tags.add(TAG_ACTOR_EMPLOYER)
                if norm_text in {"agency", "consultancy"}:
                    tags.add(TAG_ACTOR_AGENCY)
                    tags.add(TAG_RECIPIENT_AGENT)
                if norm_text in {"recruiter", "recruiters", "hr"}:
                    tags.add(TAG_ACTOR_RECRUITER)
                    tags.add(TAG_RECIPIENT_AGENT)
                if norm_text in {"candidate", "candidates", "applicant", "applicants", "jobseeker", "jobseekers"}:
                    tags.add(TAG_ACTOR_CANDIDATE)
                    tags.add(TAG_SUBJECT_CANDIDATE)
                if norm_text in {"government", "govt"}:
                    tags.add(TAG_ACTOR_GOVERNMENT)
                if norm_text in {"thirdparty", "vendor", "vendors"}:
                    tags.add(TAG_ACTOR_THIRD_PARTY)
                if norm_text in {"said", "says", "claims", "claim", "claimed", "states", "stated", "alleges", "alleged", "reports", "reported"}:
                    tags.add(TAG_VERB_REPORTING)
                if norm_text in {"denies", "denied", "refutes", "refute", "refuted", "rejects", "rejected", "repudiates", "repudiated"}:
                    tags.add(TAG_VERB_DENIAL)

            tokens.append(
                Token(
                    index=idx,
                    text=raw_text,
                    normalized_text=norm_text,
                    local_start=start,
                    local_end=end,
                    raw_start=raw_s,
                    raw_end=raw_e,
                    tags=tags,
                    is_hard_boundary=is_hard,
                    is_subclause_boundary=is_sub,
                )
            )
            idx += 1
        return tokens
    except Exception as e:
        logger.warning("Tokenization failed fail-closed on segment: %s", e)
        return []


def partition_subclauses(tokens: list[Token]) -> list[SubClause]:
    """
    Stage 2: Sub-Clause Partitioning & Sentence Scoping.
    Partitions tokens into isolated sub-clause windows across hard boundaries
    (.!?;:\n) and clause-boundary conjunctions (adversative, subordinate, coordinating),
    preventing role and modifier bleeding across independent clauses.
    """
    if not tokens:
        return []

    try:
        subclauses: list[SubClause] = []
        current: list[Token] = []
        current_has_adv = False

        for t in tokens:
            if t.is_hard_boundary:
                if current:
                    subclauses.append(
                        SubClause(
                            tokens=current,
                            start_pos=current[0].raw_start,
                            end_pos=current[-1].raw_end,
                            has_adversative=current_has_adv or any(TAG_ADVERSATIVE_CONJ in tok.tags for tok in current),
                        )
                    )
                    current = []
                current_has_adv = False
            elif t.is_subclause_boundary:
                if current:
                    subclauses.append(
                        SubClause(
                            tokens=current,
                            start_pos=current[0].raw_start,
                            end_pos=current[-1].raw_end,
                            has_adversative=current_has_adv or any(TAG_ADVERSATIVE_CONJ in tok.tags for tok in current),
                        )
                    )
                    current = []
                if TAG_ADVERSATIVE_CONJ in t.tags:
                    current_has_adv = True
                else:
                    current_has_adv = False
            else:
                current.append(t)

        if current:
            subclauses.append(
                SubClause(
                    tokens=current,
                    start_pos=current[0].raw_start,
                    end_pos=current[-1].raw_end,
                    has_adversative=current_has_adv or any(TAG_ADVERSATIVE_CONJ in tok.tags for tok in current),
                )
            )

        return subclauses
    except Exception as e:
        logger.warning("Sub-clause partitioning failed fail-closed: %s", e)
        return []


RECRUITMENT_PURPOSE_PP_RE = re.compile(
    r"\bfor\s+(?:the\s+)?(?:application(?:\s+processing)?|registration|interviews?|placement|onboarding|recruitment|hiring|job(?:\s+offer)?|document(?:ation)?\s+verification|verification|training|assessments?|background\s+check)\b",
    re.IGNORECASE,
)
RECRUITMENT_INFINITIVE_PURPOSE_RE = re.compile(
    r"\bto\s+(?:attend|participate\s+in|appear\s+for|join|receive|get)\s+(?:the\s+)?(?:interviews?|job|offer(?:\s*letter)?|placement|selection|appointment|training)\b",
    re.IGNORECASE,
)
RECRUITMENT_PREREQUISITE_RE = re.compile(
    r"\b(?:before|prior\s+to|ahead\s+of)\s+(?:receiving|getting)?\s*(?:the\s+)?(?:offer(?:\s*letter)?|interviews?|placement|selection|appointment|hiring)\b"
    r"|\b(?:as\s+a\s+condition\s+of|condition\s+for|prerequisite\s+for)\s+(?:placement|employment|hiring|selection|joining|the\s+job)\b",
    re.IGNORECASE,
)
RECRUITMENT_NOMINAL_MOD_RE = re.compile(
    r"\b(?:onboarding|recruitment|placement|application|interview|registration|processing|verification|uniform|training|documentation)\s+(?:fees?|charges?|deposit|cost|amount)\b"
    r"|\b(?:employer['’]s|company['’]s)\s+(?:designated\s+)?agency\b"
    r"|\bagency\s+designated\s+by\s+(?:the\s+)?(?:company|employer)\b"
    r"|\bto\s+receive\s+(?:their|the|an)?\s*(?:offer\s*letter|employment\s*contract|joining\s*letter)\b",
    re.IGNORECASE,
)
COMMERCIAL_EXCLUSION_RE = re.compile(
    r"\bfor\s+(?:resume\s+coaching|career\s+counseling|mock\s+interviews?|course\s+materials?|study\s+guides?|certifications?|classes|coaching)\b",
    re.IGNORECASE,
)


def evaluate_recruitment_context_gating(
    subclause: Any,
    doc_tokens: list[Token] | None = None,
) -> bool:
    """
    Evaluates whether an elided payment directive has a demonstrable, deterministic
    syntactic attachment to the recruitment process (Cases 5B and 9B).
    Enforces fail-closed evaluation: returns False if attachment cannot be proven deterministically.
    """
    if isinstance(subclause, str):
        text = subclause
    elif hasattr(subclause, "tokens"):
        text = " ".join(t.text for t in subclause.tokens)
    elif hasattr(subclause, "participating_tokens"):
        text = " ".join(t.text for t in subclause.participating_tokens)
    elif isinstance(subclause, (list, tuple)):
        text = " ".join(t.text for t in subclause if hasattr(t, "text"))
    else:
        text = str(subclause)

    if COMMERCIAL_EXCLUSION_RE.search(text):
        return False
    if RECRUITMENT_PURPOSE_PP_RE.search(text):
        return True
    if RECRUITMENT_INFINITIVE_PURPOSE_RE.search(text):
        return True
    if RECRUITMENT_PREREQUISITE_RE.search(text):
        return True
    if RECRUITMENT_NOMINAL_MOD_RE.search(text):
        return True
    return False


def identify_epistemic_frame(subclause: SubClause) -> EpistemicFrame:
    """
    Classifies the epistemic framing / stance of a sub-clause:
      - DENIAL_REPUDIATION: Denies or refutes an allegation or policy violation.
      - SCAM_WARNING: Direct warning against fraudulent offers or prohibited sharing.
      - REPORTED_CLAIM: Third-party hearsay assertion (e.g. said, claims).
      - DIRECT_INSTRUCTION: Mandatory candidate-directed action.
      - DESCRIPTIVE_FACT: Indicative descriptive statement or historical report.
      - UNKNOWN: Fallback fail-closed default.
    """
    words = [t.normalized_text for t in subclause]
    text_lower = " ".join(t.text for t in subclause).lower()

    if any(TAG_VERB_DENIAL in t.tags for t in subclause):
        return EpistemicFrame.DENIAL_REPUDIATION

    if any(TAG_PROHIBITIVE in t.tags for t in subclause):
        if (
            any(w in words for w in {"beware", "warning", "alert", "promises", "guaranteed"})
            or "anyone who" in text_lower
            or "guaranteed job" in text_lower
        ):
            return EpistemicFrame.SCAM_WARNING

    if any(TAG_VERB_REPORTING in t.tags for t in subclause):
        return EpistemicFrame.REPORTED_CLAIM

    if any(TAG_OBLIGATION_MANDATORY in t.tags for t in subclause) or any(TAG_PAY_VERB in t.tags or TAG_CRED_VERB in t.tags for t in subclause):
        if any(TAG_ACTOR_SCAMMER in t.tags for t in subclause) and not any(TAG_PROHIBITIVE in t.tags for t in subclause):
            return EpistemicFrame.DESCRIPTIVE_FACT
        return EpistemicFrame.DIRECT_INSTRUCTION

    if any(TAG_DESCRIPTIVE in t.tags or TAG_HISTORICAL_MARKER in t.tags for t in subclause):
        return EpistemicFrame.DESCRIPTIVE_FACT

    return EpistemicFrame.UNKNOWN


def identify_modality(subclause: SubClause, gov_verb: Token | None) -> ModalityType:
    """Classifies predicate modality and obligation polarity."""
    words = [t.normalized_text for t in subclause]
    text_lower = " ".join(t.text for t in subclause).lower()

    if any(TAG_PROHIBITIVE in t.tags for t in subclause):
        return ModalityType.PROHIBITED
    if any(p in text_lower for p in ["does not charge", "does not collect", "doesn't charge", "doesn't collect", "kabhi nahi"]):
        return ModalityType.NEGATED_ACTION
    if "not required" in text_lower or "no obligation" in text_lower or "does not require" in text_lower:
        return ModalityType.NOT_REQUIRED
    if "if" in words or "unless" in words or any(TAG_CONDITIONAL in t.tags for t in subclause):
        return ModalityType.CONDITIONAL
    if any(TAG_OBLIGATION_MANDATORY in t.tags for t in subclause):
        return ModalityType.MANDATORY
    if any(TAG_OBLIGATION_OPTIONAL in t.tags for t in subclause) or "voluntarily" in words:
        return ModalityType.OPTIONAL
    if any(TAG_DESCRIPTIVE in t.tags or TAG_HISTORICAL_MARKER in t.tags for t in subclause):
        return ModalityType.DECLARATIVE
    return ModalityType.DECLARATIVE if gov_verb else ModalityType.UNKNOWN


def associate_tokens_scoped(
    tokens: list[Token],
    max_window: int = 10,
) -> list[PredicateAssociation]:
    """
    Stage 2: Bounded Semantic Association with Local Governor and Action-Scoped Modifiers.
    Implements nearest-valid-governor selection, object-binding checks, competing-object rejection,
    and coordinated-predicate role inheritance (F-01).
    """
    associations: list[PredicateAssociation] = []
    subclauses = partition_subclauses(tokens)

    inherited_instruction_source = ActorRole.UNKNOWN
    inherited_action_performer = ActorRole.UNKNOWN
    inherited_reporting_source = ActorRole.UNKNOWN

    for subclause in subclauses:
        sc_len = len(subclause)

        if subclause.has_adversative:
            inherited_instruction_source = ActorRole.UNKNOWN
            inherited_action_performer = ActorRole.UNKNOWN
            inherited_reporting_source = ActorRole.UNKNOWN

        frame = identify_epistemic_frame(subclause)

        sc_instruction_source = ActorRole.UNKNOWN
        sc_action_performer = ActorRole.UNKNOWN
        sc_collection_actor = ActorRole.UNKNOWN
        sc_payment_recipient = ActorRole.UNKNOWN
        sc_reporting_source = ActorRole.UNKNOWN
        sc_claim_subject = ActorRole.UNKNOWN
        sc_affected_patient = ActorRole.UNKNOWN
        sc_info_recipient = ActorRole.UNKNOWN

        has_employer = any(TAG_ACTOR_EMPLOYER in t.tags for t in subclause)
        has_scammer = any(TAG_ACTOR_SCAMMER in t.tags for t in subclause)
        has_agency = any(TAG_ACTOR_AGENCY in t.tags for t in subclause)
        has_recruiter = any(TAG_ACTOR_RECRUITER in t.tags for t in subclause)
        has_candidate = any(TAG_ACTOR_CANDIDATE in t.tags or TAG_SUBJECT_CANDIDATE in t.tags for t in subclause)
        has_gov = any(TAG_ACTOR_GOVERNMENT in t.tags for t in subclause)

        words = [t.normalized_text for t in subclause]
        text_lower = " ".join(t.text for t in subclause).lower()

        if frame == EpistemicFrame.REPORTED_CLAIM:
            if has_recruiter:
                sc_reporting_source = ActorRole.RECRUITER
                inherited_reporting_source = ActorRole.RECRUITER
            elif has_agency:
                sc_reporting_source = ActorRole.AGENCY
                inherited_reporting_source = ActorRole.AGENCY
            elif has_employer:
                sc_reporting_source = ActorRole.EMPLOYER
                inherited_reporting_source = ActorRole.EMPLOYER
        elif inherited_reporting_source != ActorRole.UNKNOWN:
            sc_reporting_source = inherited_reporting_source
            frame = EpistemicFrame.REPORTED_CLAIM

        if frame == EpistemicFrame.REPORTED_CLAIM and has_employer and sc_reporting_source != ActorRole.EMPLOYER:
            sc_claim_subject = ActorRole.EMPLOYER

        if frame == EpistemicFrame.DENIAL_REPUDIATION:
            if has_employer:
                sc_claim_subject = ActorRole.EMPLOYER
            elif has_agency:
                sc_claim_subject = ActorRole.AGENCY

        # Stated collection actor
        if any(w in words for w in ["collects", "collect", "charges", "charge"]):
            is_negated_coll = any(w in text_lower for w in ["does not", "doesn't", "no fee", "never", "not", "without"])
            if not is_negated_coll:
                if has_agency:
                    sc_collection_actor = ActorRole.AGENCY
                elif has_employer:
                    sc_collection_actor = ActorRole.EMPLOYER

        # Payee
        if "to the agency" in text_lower or "pay the agency" in text_lower or "pay agency" in text_lower:
            sc_payment_recipient = ActorRole.AGENCY
        elif "to our company" in text_lower or "pay our company" in text_lower:
            sc_payment_recipient = ActorRole.EMPLOYER

        # Candidate action performer
        if has_candidate:
            sc_action_performer = ActorRole.CANDIDATE
        elif inherited_action_performer != ActorRole.UNKNOWN:
            sc_action_performer = inherited_action_performer

        # Instruction source
        if frame not in (EpistemicFrame.DENIAL_REPUDIATION, EpistemicFrame.REPORTED_CLAIM):
            if any(w in words for w in ["asks", "ask", "directs", "orders", "requires", "require", "instructs"]):
                if has_employer:
                    sc_instruction_source = ActorRole.EMPLOYER
                elif has_scammer:
                    sc_instruction_source = ActorRole.SCAMMER
                elif has_agency:
                    sc_instruction_source = ActorRole.AGENCY
                elif has_recruiter:
                    sc_instruction_source = ActorRole.RECRUITER
            elif inherited_instruction_source != ActorRole.UNKNOWN:
                sc_instruction_source = inherited_instruction_source
            elif has_employer and any(TAG_PAY_VERB in t.tags for t in subclause):
                sc_instruction_source = ActorRole.EMPLOYER
            elif has_scammer and any(TAG_PAY_VERB in t.tags for t in subclause):
                sc_instruction_source = ActorRole.SCAMMER

        found_anchor = False

        for i, t in enumerate(subclause):
            # A. Fee Noun Anchor (TAG_FEE_NOUN)
            if TAG_FEE_NOUN in t.tags:
                found_anchor = True
                gov_verb = None
                competing_obj = None

                w_start = max(0, i - max_window)
                w_end = min(sc_len, i + max_window + 1)

                candidate_verbs = []
                for j in range(w_start, w_end):
                    if j == i:
                        continue
                    tok = subclause[j]
                    if TAG_PAY_VERB in tok.tags:
                        candidate_verbs.append((abs(i - j), j, tok))
                    if TAG_NON_FINANCIAL_OBJ in tok.tags or TAG_UNRELATED_FIN_OBJ in tok.tags:
                        competing_obj = tok

                if candidate_verbs:
                    candidate_verbs.sort()
                    dist, verb_idx, nearest_verb = candidate_verbs[0]

                    is_bound_to_competing = False
                    if competing_obj:
                        comp_dist = abs(verb_idx - subclause.index(competing_obj))
                        if comp_dist < dist:
                            is_bound_to_competing = True

                    if not is_bound_to_competing:
                        gov_verb = nearest_verb
                else:
                    opt_modal = [tok for tok in subclause if TAG_OBLIGATION_OPTIONAL in tok.tags]
                    if opt_modal:
                        gov_verb = opt_modal[0]

                if gov_verb:
                    ob_tok = next((tok for tok in subclause if TAG_OBLIGATION_MANDATORY in tok.tags), None)
                    opt_tok = next((tok for tok in subclause if TAG_OBLIGATION_OPTIONAL in tok.tags), None)
                    hist_tok = next((tok for tok in subclause if TAG_HISTORICAL_MARKER in tok.tags), None)
                    neg_tok = next((tok for tok in subclause if TAG_PROHIBITIVE in tok.tags), None)
                    desc_tok = next((tok for tok in subclause if TAG_DESCRIPTIVE in tok.tags), None)
                    amt_tok = next((tok for tok in subclause if TAG_AMOUNT in tok.tags), None)
                    subj_tok = next((tok for tok in subclause if TAG_SUBJECT_CANDIDATE in tok.tags), None)
                    prec_tok = next((tok for tok in subclause if TAG_PRECOND in tok.tags), None)

                    is_neg = False
                    if neg_tok:
                        neg_idx = subclause.index(neg_tok)
                        v_idx = subclause.index(gov_verb)
                        if abs(neg_idx - v_idx) <= 4 or neg_idx < v_idx:
                            is_neg = True

                    core = [t, gov_verb]
                    if ob_tok:
                        core.append(ob_tok)
                    if amt_tok:
                        core.append(amt_tok)
                    if subj_tok:
                        core.append(subj_tok)
                    if prec_tok:
                        core.append(prec_tok)
                    if neg_tok and is_neg:
                        core.append(neg_tok)
                    core.sort(key=lambda x: x.index)

                    modality = identify_modality(subclause, gov_verb)

                    associations.append(
                        PredicateAssociation(
                            governing_verb=gov_verb,
                            anchor_noun=t,
                            instruction_source=sc_instruction_source,
                            action_performer=sc_action_performer,
                            collection_actor=sc_collection_actor,
                            payment_recipient=sc_payment_recipient,
                            affected_patient=sc_affected_patient,
                            information_recipient=sc_info_recipient,
                            reporting_source=sc_reporting_source,
                            claim_subject=sc_claim_subject,
                            epistemic_frame=frame,
                            modality=modality,
                            participating_tokens=core,
                            raw_start=min(tok.raw_start for tok in core),
                            raw_end=max(tok.raw_end for tok in core),
                            obligation_token=ob_tok,
                            negation_token=neg_tok if is_neg else None,
                            descriptive_token=desc_tok,
                            amount_token=amt_tok,
                            subject_token=subj_tok,
                            recipient_token=None,
                            precond_token=prec_tok,
                            self_auth_token=None,
                            is_negated=is_neg,
                            is_descriptive=desc_tok is not None,
                            is_historical=hist_tok is not None,
                            is_optional=opt_tok is not None,
                            has_self_auth=False,
                            association_type="FEE_DEMAND",
                        )
                    )

            # B. Credential Noun Anchor (TAG_CRED_NOUN)
            elif TAG_CRED_NOUN in t.tags:
                found_anchor = True
                gov_verb = None
                w_start = max(0, i - max_window)
                w_end = min(sc_len, i + max_window + 1)

                candidate_verbs = []
                for j in range(w_start, w_end):
                    if j == i:
                        continue
                    tok = subclause[j]
                    if TAG_CRED_VERB in tok.tags:
                        candidate_verbs.append((abs(i - j), j, tok))

                if candidate_verbs:
                    candidate_verbs.sort()
                    gov_verb = candidate_verbs[0][2]

                if gov_verb:
                    neg_tok = next((tok for tok in subclause if TAG_PROHIBITIVE in tok.tags), None)
                    desc_tok = next((tok for tok in subclause if TAG_DESCRIPTIVE in tok.tags), None)
                    recip_tok = next((tok for tok in subclause if TAG_RECIPIENT_AGENT in tok.tags), None)
                    self_auth_tok = next((tok for tok in subclause if TAG_SELF_AUTH_MARKER in tok.tags), None)
                    pos_tok = next((tok for tok in subclause if TAG_POSSESSIVE in tok.tags), None)

                    is_neg = False
                    if neg_tok:
                        neg_idx = subclause.index(neg_tok)
                        v_idx = subclause.index(gov_verb)
                        if abs(neg_idx - v_idx) <= 4 or neg_idx < v_idx:
                            is_neg = True

                    core = [t, gov_verb]
                    if recip_tok:
                        core.append(recip_tok)
                    if pos_tok:
                        core.append(pos_tok)
                    if neg_tok and is_neg:
                        core.append(neg_tok)
                    core.sort(key=lambda x: x.index)

                    modality = identify_modality(subclause, gov_verb)

                    associations.append(
                        PredicateAssociation(
                            governing_verb=gov_verb,
                            anchor_noun=t,
                            instruction_source=sc_instruction_source,
                            action_performer=sc_action_performer,
                            collection_actor=sc_collection_actor,
                            payment_recipient=sc_payment_recipient,
                            affected_patient=sc_affected_patient,
                            information_recipient=sc_info_recipient,
                            reporting_source=sc_reporting_source,
                            claim_subject=sc_claim_subject,
                            epistemic_frame=frame,
                            modality=modality,
                            participating_tokens=core,
                            raw_start=min(tok.raw_start for tok in core),
                            raw_end=max(tok.raw_end for tok in core),
                            obligation_token=None,
                            negation_token=neg_tok if is_neg else None,
                            descriptive_token=desc_tok,
                            amount_token=None,
                            subject_token=None,
                            recipient_token=recip_tok,
                            precond_token=None,
                            self_auth_token=self_auth_tok,
                            is_negated=is_neg,
                            is_descriptive=desc_tok is not None,
                            is_historical=False,
                            is_optional=False,
                            has_self_auth=self_auth_tok is not None,
                            association_type="CRED_DISCLOSE",
                        )
                    )

            # C. Reverse Auth Noun Anchor (TAG_REV_NOUN)
            elif TAG_REV_NOUN in t.tags:
                found_anchor = True
                gov_verb = None
                w_start = max(0, i - max_window)
                w_end = min(sc_len, i + max_window + 1)

                candidate_verbs = []
                for j in range(w_start, w_end):
                    if j == i:
                        continue
                    tok = subclause[j]
                    if TAG_REV_VERB in tok.tags:
                        candidate_verbs.append((abs(i - j), j, tok))

                if candidate_verbs:
                    candidate_verbs.sort()
                    gov_verb = candidate_verbs[0][2]

                if gov_verb:
                    neg_tok = next((tok for tok in subclause if TAG_PROHIBITIVE in tok.tags), None)
                    desc_tok = next((tok for tok in subclause if TAG_DESCRIPTIVE in tok.tags), None)

                    is_neg = False
                    if neg_tok:
                        neg_idx = subclause.index(neg_tok)
                        v_idx = subclause.index(gov_verb)
                        if abs(neg_idx - v_idx) <= 4 or neg_idx < v_idx:
                            is_neg = True

                    core = [t, gov_verb]
                    if neg_tok and is_neg:
                        core.append(neg_tok)
                    core.sort(key=lambda x: x.index)

                    modality = identify_modality(subclause, gov_verb)

                    associations.append(
                        PredicateAssociation(
                            governing_verb=gov_verb,
                            anchor_noun=t,
                            instruction_source=sc_instruction_source,
                            action_performer=sc_action_performer,
                            collection_actor=sc_collection_actor,
                            payment_recipient=sc_payment_recipient,
                            affected_patient=sc_affected_patient,
                            information_recipient=sc_info_recipient,
                            reporting_source=sc_reporting_source,
                            claim_subject=sc_claim_subject,
                            epistemic_frame=frame,
                            modality=modality,
                            participating_tokens=core,
                            raw_start=min(tok.raw_start for tok in core),
                            raw_end=max(tok.raw_end for tok in core),
                            obligation_token=None,
                            negation_token=neg_tok if is_neg else None,
                            descriptive_token=desc_tok,
                            amount_token=None,
                            subject_token=None,
                            recipient_token=None,
                            precond_token=None,
                            self_auth_token=None,
                            is_negated=is_neg,
                            is_descriptive=desc_tok is not None,
                            is_historical=False,
                            is_optional=False,
                            has_self_auth=False,
                            association_type="REV_AUTH",
                        )
                    )

        # Capture denial or reporting verbs even if fee noun is present in the clause
        for dv in [tok for tok in subclause if TAG_VERB_DENIAL in tok.tags]:
            if not any(a.governing_verb == dv for a in associations):
                associations.append(
                    PredicateAssociation(
                        governing_verb=dv,
                        anchor_noun=next((tok for tok in subclause if TAG_FEE_NOUN in tok.tags), None),
                        instruction_source=ActorRole.UNKNOWN,
                        action_performer=ActorRole.UNKNOWN,
                        collection_actor=ActorRole.UNKNOWN,
                        payment_recipient=ActorRole.UNKNOWN,
                        affected_patient=ActorRole.UNKNOWN,
                        information_recipient=ActorRole.UNKNOWN,
                        reporting_source=ActorRole.UNKNOWN,
                        claim_subject=sc_claim_subject if sc_claim_subject != ActorRole.UNKNOWN else (ActorRole.EMPLOYER if has_employer else ActorRole.UNKNOWN),
                        epistemic_frame=EpistemicFrame.DENIAL_REPUDIATION,
                        modality=ModalityType.NEGATED_ACTION,
                        participating_tokens=list(subclause.tokens),
                        raw_start=subclause.start_pos,
                        raw_end=subclause.end_pos,
                        is_negated=True,
                        association_type="DENIAL_REPUDIATION",
                    )
                )

        for rv in [tok for tok in subclause if TAG_VERB_REPORTING in tok.tags]:
            if not any(a.governing_verb == rv for a in associations):
                associations.append(
                    PredicateAssociation(
                        governing_verb=rv,
                        anchor_noun=None,
                        instruction_source=ActorRole.UNKNOWN,
                        action_performer=ActorRole.UNKNOWN,
                        collection_actor=ActorRole.UNKNOWN,
                        payment_recipient=ActorRole.UNKNOWN,
                        affected_patient=ActorRole.UNKNOWN,
                        information_recipient=ActorRole.UNKNOWN,
                        reporting_source=sc_reporting_source,
                        claim_subject=sc_claim_subject,
                        epistemic_frame=EpistemicFrame.REPORTED_CLAIM,
                        modality=ModalityType.DECLARATIVE,
                        participating_tokens=list(subclause.tokens),
                        raw_start=subclause.start_pos,
                        raw_end=subclause.end_pos,
                        association_type="REPORTED_CLAIM",
                    )
                )

        # Standalone predicate verb (e.g. submit documents, pay without fee noun, register)
        if not found_anchor:
            verbs = [t for t in subclause if any(tag in t.tags for tag in (TAG_PAY_VERB, TAG_VERB_REPORTING, TAG_VERB_DENIAL, TAG_CRED_VERB, TAG_ACTION_NON_FINANCIAL))]
            if verbs and not any(a.governing_verb in verbs for a in associations):
                gov_verb = verbs[0]
                is_eligible_standalone = (
                    TAG_ACTION_NON_FINANCIAL in gov_verb.tags
                    or TAG_VERB_REPORTING in gov_verb.tags
                    or TAG_VERB_DENIAL in gov_verb.tags
                    or TAG_CRED_VERB in gov_verb.tags
                    or sc_payment_recipient in (ActorRole.AGENCY, ActorRole.THIRD_PARTY)
                    or sc_reporting_source != ActorRole.UNKNOWN
                    or frame == EpistemicFrame.SCAM_WARNING
                    or (has_employer and any(w in words for w in {"charge", "charges", "require", "requires"}))
                )
                if is_eligible_standalone:
                    modality = identify_modality(subclause, gov_verb)
                    core = list(subclause.tokens)
                    assoc_type = "NON_FINANCIAL_ACTION" if TAG_ACTION_NON_FINANCIAL in gov_verb.tags else "PREDICATE_VERB"
                    associations.append(
                        PredicateAssociation(
                            governing_verb=gov_verb,
                            anchor_noun=None,
                            instruction_source=sc_instruction_source,
                            action_performer=sc_action_performer,
                            collection_actor=sc_collection_actor,
                            payment_recipient=sc_payment_recipient,
                            affected_patient=sc_affected_patient,
                            information_recipient=sc_info_recipient,
                            reporting_source=sc_reporting_source,
                            claim_subject=sc_claim_subject,
                            epistemic_frame=frame,
                            modality=modality,
                            participating_tokens=core,
                            raw_start=subclause.start_pos,
                            raw_end=subclause.end_pos,
                            is_negated=modality in (ModalityType.PROHIBITED, ModalityType.NEGATED_ACTION),
                            is_descriptive=frame == EpistemicFrame.DESCRIPTIVE_FACT,
                            is_optional=modality == ModalityType.OPTIONAL,
                            association_type=assoc_type,
                        )
                    )

        if sc_instruction_source != ActorRole.UNKNOWN:
            inherited_instruction_source = sc_instruction_source
        if sc_action_performer != ActorRole.UNKNOWN:
            inherited_action_performer = sc_action_performer
        if sc_reporting_source != ActorRole.UNKNOWN:
            inherited_reporting_source = sc_reporting_source

    return associations


# Alias for explicit semantic role attribution
associate_predicate_roles = associate_tokens_scoped


def qualify_semantic_association(
    assoc: SemanticAssociation | PredicateAssociation,
    has_recruitment_context: bool = False,
) -> QualificationDecision:
    """
    Stage 3: Pure Semantic Qualification (Decoupled from Emission).
    Evaluates self-authentication, descriptive framing, historical reports,
    discretionary fees, category-preserving advisories, reported claims,
    and active demands across all 20 acceptance matrix cases.
    """
    if hasattr(assoc, "raw_start") and hasattr(assoc, "raw_end"):
        raw_start = assoc.raw_start
        raw_end = assoc.raw_end
    elif getattr(assoc, "participating_tokens", None):
        raw_start = min(t.raw_start for t in assoc.participating_tokens)
        raw_end = max(t.raw_end for t in assoc.participating_tokens)
    else:
        raw_start = 0
        raw_end = 0

    tokens = getattr(assoc, "participating_tokens", [])
    token_texts = {t.normalized_text for t in tokens}

    # 1. Self-Initiated Banking Authentication
    if getattr(assoc, "has_self_auth", False):
        return QualificationDecision(
            is_qualified=False,
            pattern_id=None,
            category=None,
            severity=None,
            eligibility=CandidateEligibility.REJECTED,
            status=None,
            explanation=None,
            semantic_target=None,
            action_type=None,
            rejection_reason=RejectionReason.SELF_AUTH_NOT_DISCLOSURE,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    # 2. Historical & Descriptive Language
    if getattr(assoc, "is_historical", False):
        return QualificationDecision(
            is_qualified=False,
            pattern_id=None,
            category=None,
            severity=None,
            eligibility=CandidateEligibility.REJECTED,
            status=None,
            explanation=None,
            semantic_target=None,
            action_type=None,
            rejection_reason=RejectionReason.HISTORICAL_REPORT,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    if getattr(assoc, "is_descriptive", False) and getattr(assoc, "collection_actor", None) != ActorRole.AGENCY and getattr(assoc, "instruction_source", None) != ActorRole.SCAMMER:
        return QualificationDecision(
            is_qualified=False,
            pattern_id=None,
            category=None,
            severity=None,
            eligibility=CandidateEligibility.REJECTED,
            status=None,
            explanation=None,
            semantic_target=None,
            action_type=None,
            rejection_reason=RejectionReason.DESCRIPTIVE_NOT_DIRECTIVE,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    # 3. Non-Financial Actions (Cases 6.1, 7.1)
    gov_norm = getattr(getattr(assoc, "governing_verb", None), "normalized_text", "")
    assoc_type = getattr(assoc, "association_type", "")
    if (
        assoc_type == "NON_FINANCIAL_ACTION"
        or (gov_norm in {"submit", "register", "upload", "apply"} and not any(TAG_FEE_NOUN in t.tags for t in tokens))
    ):
        return QualificationDecision(
            is_qualified=False,
            pattern_id=None,
            category=ScamCategory.UNKNOWN,
            severity=FindingSeverity.INFORMATIONAL,
            eligibility=CandidateEligibility.REJECTED,
            status=FindingStatus.EXPLICIT_CLAIM,
            explanation="Non-financial candidate action (e.g. document submission or registration).",
            semantic_target="target:non_financial",
            action_type="action:non_financial",
            rejection_reason=RejectionReason.NON_FINANCIAL_ACTION,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    # 3b. Dedicated Credential & Reverse Auth Advisories / Actions
    modality = getattr(assoc, "modality", ModalityType.UNKNOWN)
    is_neg = (
        getattr(assoc, "is_negated", False)
        or modality in (ModalityType.PROHIBITED, ModalityType.NEGATED_ACTION, ModalityType.NOT_REQUIRED)
    )
    if assoc_type == "CRED_DISCLOSE":
        if is_neg:
            return QualificationDecision(
                is_qualified=True,
                pattern_id="ADVISORY_CREDENTIAL_SAFETY",
                category=ScamCategory.CREDENTIAL_HARVESTING,
                severity=FindingSeverity.INFORMATIONAL,
                eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
                status=FindingStatus.EXPLICIT_CLAIM,
                explanation="Advisory guidance warning candidates not to disclose credentials or authentication secrets.",
                semantic_target="target:cred_safety",
                action_type="advisory:caution",
                rejection_reason=None,
                raw_start=raw_start,
                raw_end=raw_end,
            )
        else:
            return QualificationDecision(
                is_qualified=True,
                pattern_id="CRED_HARVEST_AUTH_SECRETS",
                category=ScamCategory.CREDENTIAL_HARVESTING,
                severity=FindingSeverity.CRITICAL,
                eligibility=CandidateEligibility.SCORABLE_ACTIVE,
                status=FindingStatus.EXPLICIT_CLAIM,
                explanation="Directive requesting banking authentication secret or OTP disclosure.",
                semantic_target="target:cred_upi_pin",
                action_type="harvest:disclose",
                rejection_reason=None,
                raw_start=raw_start,
                raw_end=raw_end,
            )

    if assoc_type == "REV_AUTH":
        if is_neg:
            return QualificationDecision(
                is_qualified=True,
                pattern_id="ADVISORY_REVERSE_AUTH_SAFETY",
                category=ScamCategory.ADVANCE_FEE,
                severity=FindingSeverity.INFORMATIONAL,
                eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
                status=FindingStatus.EXPLICIT_CLAIM,
                explanation="Advisory guidance warning candidates against approving collect requests or debit authorizations.",
                semantic_target="target:rev_safety",
                action_type="advisory:caution",
                rejection_reason=None,
                raw_start=raw_start,
                raw_end=raw_end,
            )
        else:
            return QualificationDecision(
                is_qualified=True,
                pattern_id="ADV_FEE_REVERSE_AUTH",
                category=ScamCategory.ADVANCE_FEE,
                severity=FindingSeverity.CRITICAL,
                eligibility=CandidateEligibility.SCORABLE_ACTIVE,
                status=FindingStatus.EXPLICIT_CLAIM,
                explanation="Deceptive reverse collect request instructing candidate to approve UPI debit to receive funds.",
                semantic_target="target:rev_upi_collect",
                action_type="fraud:overpay",
                rejection_reason=None,
                raw_start=raw_start,
                raw_end=raw_end,
            )

    # 4. Epistemic: Scam Warnings (Case 10)
    frame = getattr(assoc, "epistemic_frame", None)
    if frame == EpistemicFrame.SCAM_WARNING:
        return QualificationDecision(
            is_qualified=True,
            pattern_id="ADVISORY_SCAM_WARNING",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.INFORMATIONAL,
            eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            explanation="Scam warning advisory against fraudulent recruitment charges or guaranteed job promises.",
            semantic_target="target:warning",
            action_type="advisory:warning",
            rejection_reason=None,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    # 5. Epistemic: Denial & Repudiation (Cases 9.2, 12.2)
    if frame == EpistemicFrame.DENIAL_REPUDIATION:
        return QualificationDecision(
            is_qualified=True,
            pattern_id="DENIAL_COMPANY_REPUDIATION",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.INFORMATIONAL,
            eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            explanation="Employer repudiation or denial of fee authorization or third-party collection.",
            semantic_target="target:denial",
            action_type="disclaimer:repudiation",
            rejection_reason=None,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    # 6. Scammer Instruction Source (Case 4)
    instr_source = getattr(assoc, "instruction_source", ActorRole.UNKNOWN)
    if instr_source == ActorRole.SCAMMER:
        return QualificationDecision(
            is_qualified=True,
            pattern_id="ADVISORY_SCAMMER_ATTRIBUTION",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.INFORMATIONAL,
            eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            explanation="External fraudster directive attributed to scammers; non-scorable advisory.",
            semantic_target="target:scammer_directive",
            action_type="advisory:scammer",
            rejection_reason=None,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    # 7. Discretionary / Optional / Statutory Fees (Cases 7.2, 8.2, 11)
    modality = getattr(assoc, "modality", ModalityType.UNKNOWN)
    is_opt = getattr(assoc, "is_optional", False) or modality == ModalityType.OPTIONAL
    is_conditional = modality == ModalityType.CONDITIONAL or "unless" in token_texts or "except" in token_texts

    # Case 8.2: Government examination fee / statutory exception
    is_govt = (
        getattr(assoc, "collection_actor", None) == ActorRole.GOVERNMENT
        or getattr(assoc, "payment_recipient", None) == ActorRole.GOVERNMENT
        or any(tok in token_texts for tok in {"government", "govt", "examination", "exam", "upsc", "ssc"})
    )
    if is_opt and is_govt:
        return QualificationDecision(
            is_qualified=True,
            pattern_id="ADVISORY_GOVERNMENT_EXAM_FEE",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.INFORMATIONAL,
            eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
            status=FindingStatus.AMBIGUOUS_SIGNAL,
            explanation="Voluntary or statutory government examination fee notice; non-scorable advisory.",
            semantic_target="target:government_fee",
            action_type="advisory:statutory_fee",
            rejection_reason=None,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    # Case 11: Conditional equipment / optional fee exception
    if is_conditional or (modality == ModalityType.NOT_REQUIRED and "unless" in token_texts):
        return QualificationDecision(
            is_qualified=False,
            pattern_id=None,
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.LOW,
            eligibility=CandidateEligibility.REJECTED,
            status=FindingStatus.AMBIGUOUS_SIGNAL,
            explanation="Discretionary equipment exception; rejected as non-mandatory.",
            semantic_target="target:equipment_exception",
            action_type="rejection:conditional_discretionary",
            rejection_reason=RejectionReason.DISCRETIONARY_NOT_MANDATORY,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    # Case 7.2: Generic discretionary / optional fee
    if is_opt:
        return QualificationDecision(
            is_qualified=False,
            pattern_id=None,
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.LOW,
            eligibility=CandidateEligibility.REJECTED,
            status=FindingStatus.AMBIGUOUS_SIGNAL,
            explanation="Discretionary or voluntary fee payment (not mandatory).",
            semantic_target="target:fee_optional",
            action_type="rejection:discretionary",
            rejection_reason=RejectionReason.DISCRETIONARY_NOT_MANDATORY,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    # 8. Negated / Prohibitive / Disclaimers (Cases 2, 3.1, 8.1)
    is_neg = (
        getattr(assoc, "is_negated", False)
        or modality in (ModalityType.PROHIBITED, ModalityType.NEGATED_ACTION, ModalityType.NOT_REQUIRED)
    )
    if is_neg:
        has_employer = (
            getattr(assoc, "instruction_source", None) == ActorRole.EMPLOYER
            or getattr(assoc, "claim_subject", None) == ActorRole.EMPLOYER
            or any(t in token_texts for t in {"company", "employer", "firm", "our company", "the company"})
        )
        if has_employer or modality == ModalityType.NOT_REQUIRED:
            return QualificationDecision(
                is_qualified=True,
                pattern_id="DISCLAIMER_COMPANY_NO_FEE",
                category=ScamCategory.ADVANCE_FEE,
                severity=FindingSeverity.INFORMATIONAL,
                eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
                status=FindingStatus.EXPLICIT_CLAIM,
                explanation="Employer disclaimer stating that company does not charge candidate fees.",
                semantic_target="target:company_disclaimer",
                action_type="disclaimer:no_fee",
                rejection_reason=None,
                raw_start=raw_start,
                raw_end=raw_end,
            )
        if assoc_type == "FEE_DEMAND" or any(TAG_FEE_NOUN in t.tags for t in tokens):
            return QualificationDecision(
                is_qualified=True,
                pattern_id="ADVISORY_GENERAL_NO_FEE",
                category=ScamCategory.ADVANCE_FEE,
                severity=FindingSeverity.INFORMATIONAL,
                eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
                status=FindingStatus.EXPLICIT_CLAIM,
                explanation="General guidance or cautionary directive against paying upfront recruitment charges.",
                semantic_target="target:fee_general",
                action_type="advisory:caution",
                rejection_reason=None,
                raw_start=raw_start,
                raw_end=raw_end,
            )
        elif assoc_type == "CRED_DISCLOSE":
            return QualificationDecision(
                is_qualified=True,
                pattern_id="ADVISORY_CREDENTIAL_SAFETY",
                category=ScamCategory.CREDENTIAL_HARVESTING,
                severity=FindingSeverity.INFORMATIONAL,
                eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
                status=FindingStatus.EXPLICIT_CLAIM,
                explanation="Advisory guidance warning candidates not to disclose credentials or authentication secrets.",
                semantic_target="target:cred_safety",
                action_type="advisory:caution",
                rejection_reason=None,
                raw_start=raw_start,
                raw_end=raw_end,
            )
        elif assoc_type == "REV_AUTH":
            return QualificationDecision(
                is_qualified=True,
                pattern_id="ADVISORY_REVERSE_AUTH_SAFETY",
                category=ScamCategory.ADVANCE_FEE,
                severity=FindingSeverity.INFORMATIONAL,
                eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
                status=FindingStatus.EXPLICIT_CLAIM,
                explanation="Advisory guidance warning candidates against approving collect requests or debit authorizations.",
                semantic_target="target:rev_safety",
                action_type="advisory:caution",
                rejection_reason=None,
                raw_start=raw_start,
                raw_end=raw_end,
            )

    # 9. Reported Claims & Hearsay (Cases 9A, 9B, 12.1)
    rep_source = getattr(assoc, "reporting_source", ActorRole.UNKNOWN)
    if frame == EpistemicFrame.REPORTED_CLAIM or rep_source != ActorRole.UNKNOWN:
        claim_subj = getattr(assoc, "claim_subject", ActorRole.UNKNOWN)
        if claim_subj == ActorRole.EMPLOYER or getattr(assoc, "collection_actor", None) == ActorRole.EMPLOYER:
            return QualificationDecision(
                is_qualified=True,
                pattern_id="ADVISORY_ALLEGED_EMPLOYER_FEE",
                category=ScamCategory.ADVANCE_FEE,
                severity=FindingSeverity.INFORMATIONAL,
                eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
                status=FindingStatus.AMBIGUOUS_SIGNAL,
                explanation="Third-party claim alleging employer fee collection.",
                semantic_target="target:alleged_fee",
                action_type="advisory:allegation",
                rejection_reason=None,
                raw_start=raw_start,
                raw_end=raw_end,
            )
        if has_recruitment_context:
            return QualificationDecision(
                is_qualified=True,
                pattern_id="ADV_FEE_REPORTED_RECRUITMENT",
                category=ScamCategory.ADVANCE_FEE,
                severity=FindingSeverity.LOW,
                eligibility=CandidateEligibility.SCORABLE_AMBIGUOUS,
                status=FindingStatus.AMBIGUOUS_SIGNAL,
                explanation="Reported fee instruction demonstrably connected to recruitment process.",
                semantic_target="target:reported_recruitment_fee",
                action_type="demand:reported_fee",
                rejection_reason=None,
                raw_start=raw_start,
                raw_end=raw_end,
            )
        else:
            return QualificationDecision(
                is_qualified=False,
                pattern_id=None,
                category=ScamCategory.ADVANCE_FEE,
                severity=FindingSeverity.LOW,
                eligibility=CandidateEligibility.REJECTED,
                status=FindingStatus.AMBIGUOUS_SIGNAL,
                explanation="Reported third-party claim lacking demonstrable recruitment context.",
                semantic_target="target:reported_unverified",
                action_type="rejection:no_context",
                rejection_reason=RejectionReason.NO_RECRUITMENT_CONTEXT,
                raw_start=raw_start,
                raw_end=raw_end,
            )

    # 10. Descriptive Fact: Agency Collection (Case 3.2)
    coll_actor = getattr(assoc, "collection_actor", ActorRole.UNKNOWN)
    if coll_actor == ActorRole.AGENCY and frame == EpistemicFrame.DESCRIPTIVE_FACT:
        return QualificationDecision(
            is_qualified=True,
            pattern_id="ADVISORY_AGENCY_FEE",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.INFORMATIONAL,
            eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
            status=FindingStatus.AMBIGUOUS_SIGNAL,
            explanation="Descriptive report of agency fee collection.",
            semantic_target="target:agency_fee",
            action_type="advisory:agency_collection",
            rejection_reason=None,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    if getattr(assoc, "is_descriptive", False) or frame == EpistemicFrame.DESCRIPTIVE_FACT:
        return QualificationDecision(
            is_qualified=False,
            pattern_id=None,
            category=None,
            severity=None,
            eligibility=CandidateEligibility.REJECTED,
            status=None,
            explanation=None,
            semantic_target=None,
            action_type=None,
            rejection_reason=RejectionReason.DESCRIPTIVE_NOT_DIRECTIVE,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    # 11. External Payee / Agency Directives with Elided Fee (Cases 5A, 5B)
    recip = getattr(assoc, "payment_recipient", ActorRole.UNKNOWN)
    if recip in (ActorRole.AGENCY, ActorRole.THIRD_PARTY):
        if has_recruitment_context:
            return QualificationDecision(
                is_qualified=True,
                pattern_id="ADV_FEE_AGENCY_RECRUITMENT",
                category=ScamCategory.ADVANCE_FEE,
                severity=FindingSeverity.LOW,
                eligibility=CandidateEligibility.SCORABLE_AMBIGUOUS,
                status=FindingStatus.AMBIGUOUS_SIGNAL,
                explanation="Payment directive to agency demonstrably connected to recruitment process.",
                semantic_target="target:agency_recruitment_fee",
                action_type="demand:agency_fee",
                rejection_reason=None,
                raw_start=raw_start,
                raw_end=raw_end,
            )
        else:
            return QualificationDecision(
                is_qualified=False,
                pattern_id=None,
                category=ScamCategory.ADVANCE_FEE,
                severity=FindingSeverity.LOW,
                eligibility=CandidateEligibility.REJECTED,
                status=FindingStatus.AMBIGUOUS_SIGNAL,
                explanation="Payment directive to agency lacking verified recruitment context.",
                semantic_target="target:agency_isolated",
                action_type="rejection:no_context",
                rejection_reason=RejectionReason.NO_RECRUITMENT_CONTEXT,
                raw_start=raw_start,
                raw_end=raw_end,
            )

    # 12. Active Scam Demands (Cases 1, 6.2, Credential Disclosure, Reverse Auth)
    if assoc_type == "FEE_DEMAND" or any(TAG_FEE_NOUN in t.tags for t in tokens):
        return QualificationDecision(
            is_qualified=True,
            pattern_id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.CRITICAL,
            eligibility=CandidateEligibility.SCORABLE_ACTIVE,
            status=FindingStatus.EXPLICIT_CLAIM,
            explanation="Mandatory advance fee payment directive in recruitment context.",
            semantic_target="target:fee_registration",
            action_type="demand:pay",
            rejection_reason=None,
            raw_start=raw_start,
            raw_end=raw_end,
        )
    elif assoc_type == "CRED_DISCLOSE":
        return QualificationDecision(
            is_qualified=True,
            pattern_id="CRED_HARVEST_AUTH_SECRETS",
            category=ScamCategory.CREDENTIAL_HARVESTING,
            severity=FindingSeverity.CRITICAL,
            eligibility=CandidateEligibility.SCORABLE_ACTIVE,
            status=FindingStatus.EXPLICIT_CLAIM,
            explanation="Directive requesting banking authentication secret or OTP disclosure.",
            semantic_target="target:cred_upi_pin",
            action_type="harvest:disclose",
            rejection_reason=None,
            raw_start=raw_start,
            raw_end=raw_end,
        )
    elif assoc_type == "REV_AUTH":
        return QualificationDecision(
            is_qualified=True,
            pattern_id="ADV_FEE_REVERSE_AUTH",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.CRITICAL,
            eligibility=CandidateEligibility.SCORABLE_ACTIVE,
            status=FindingStatus.EXPLICIT_CLAIM,
            explanation="Deceptive reverse collect request instructing candidate to approve UPI debit to receive funds.",
            semantic_target="target:rev_upi_collect",
            action_type="fraud:overpay",
            rejection_reason=None,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    return QualificationDecision(
        is_qualified=False,
        pattern_id=None,
        category=None,
        severity=None,
        eligibility=CandidateEligibility.REJECTED,
        status=None,
        explanation=None,
        semantic_target=None,
        action_type=None,
        rejection_reason=RejectionReason.NO_GOVERNING_VERB,
        raw_start=raw_start,
        raw_end=raw_end,
    )


def emit_candidate(
    decision: QualificationDecision,
    assoc: SemanticAssociation,
) -> SemanticMatchResult:
    """
    Stage 4: Candidate Emission with Tightly Bounded Span.
    Derives candidate span strictly from participating semantic core tokens.
    """
    min_loc_s = min(t.local_start for t in assoc.participating_tokens)
    max_loc_e = max(t.local_end for t in assoc.participating_tokens)
    min_raw_s = min(t.raw_start for t in assoc.participating_tokens)
    max_raw_e = max(t.raw_end for t in assoc.participating_tokens)

    return SemanticMatchResult(
        pattern_id=decision.pattern_id or "UNKNOWN_PATTERN",
        category=decision.category or ScamCategory.UNKNOWN,
        severity=decision.severity or FindingSeverity.INFORMATIONAL,
        local_start=min_loc_s,
        local_end=max_loc_e,
        raw_start=min_raw_s,
        raw_end=max_raw_e,
        explanation=decision.explanation or "Qualified semantic pattern match.",
        semantic_target=decision.semantic_target or "target:unknown",
        action_type=decision.action_type or "demand:unspecified",
        eligibility=decision.eligibility or CandidateEligibility.SCORABLE_ACTIVE,
        status=decision.status or FindingStatus.EXPLICIT_CLAIM,
        participating_token_indices=[t.index for t in assoc.participating_tokens],
    )


def match_semantic_actions(
    text: str,
    source_intervals: list[tuple[int, int]] | None = None,
    pattern_candidate_counts: dict[str, int] | None = None,
    total_candidates_tracker: list[int] | None = None,
    max_per_pattern: int = 50,
    max_total: int = 200,
) -> list[SemanticMatchResult]:
    """
    Executes the multi-stage semantic qualification pipeline:
      Stage 1: Tokenization and lexical tagging.
      Stage 2: Bounded semantic association with sub-clause isolation.
      Stage 3: Pure semantic qualification and category-preserving advisory handling.
      Stage 4: Candidate emission with tightly bounded spans.
    """
    if pattern_candidate_counts is None:
        pattern_candidate_counts = {}
    if total_candidates_tracker is None:
        total_candidates_tracker = [0]

    if total_candidates_tracker[0] >= max_total:
        return []

    tokens = tokenize_and_tag(text, source_intervals)
    if not tokens:
        return []

    associations = associate_tokens_scoped(tokens, max_window=10)
    if not associations:
        return []

    emitted: list[SemanticMatchResult] = []
    for assoc in associations:
        if total_candidates_tracker[0] >= max_total:
            break

        decision = qualify_semantic_association(assoc)
        if not decision.is_qualified:
            continue

        p_id = decision.pattern_id or "UNKNOWN"
        if pattern_candidate_counts.get(p_id, 0) >= max_per_pattern:
            continue

        candidate = emit_candidate(decision, assoc)
        emitted.append(candidate)

        pattern_candidate_counts[p_id] = pattern_candidate_counts.get(p_id, 0) + 1
        total_candidates_tracker[0] += 1

    return emitted


def match_hinglish_in_segment(
    segment: Any,
    pattern_candidate_counts: dict[str, int] | None = None,
    total_candidates_tracker: list[int] | None = None,
    max_per_pattern: int = 50,
    max_total: int = 200,
) -> list[SemanticMatchResult]:
    """Adapter invoking match_semantic_actions on a TextSegment instance."""
    segment_text = getattr(segment, "text", str(segment))
    source_intervals = getattr(segment, "source_intervals", None)
    return match_semantic_actions(
        text=segment_text,
        source_intervals=source_intervals,
        pattern_candidate_counts=pattern_candidate_counts,
        total_candidates_tracker=total_candidates_tracker,
        max_per_pattern=max_per_pattern,
        max_total=max_total,
    )


class SemanticEvidenceAdapter:
    """
    Step 3.5: Hardened Semantic Evidence Adapter.
    Transforms pure QualificationDecision tuples into Phase 2 compliant EvidenceItem models.
    Guarantees:
      - Populate all 4 required provenance fields (model_fields_set): category, source, severity, status.
      - Sets source = DetectionSource.DETERMINISTIC_RULE.
      - Deterministic finding_id: f"{pattern_id}_{raw_start}_{raw_end}".
      - Verbatim matched_text: raw_content[raw_start:raw_end].
      - Fail-closed on bounds error or Pydantic validation failure.
    """

    @staticmethod
    def adapt(decision: QualificationDecision, raw_content: str) -> EvidenceItem | None:
        try:
            if not isinstance(raw_content, str):
                logger.warning("SemanticEvidenceAdapter fail-closed: raw_content must be str")
                return None

            raw_start = decision.raw_start
            raw_end = decision.raw_end
            if not (0 <= raw_start <= raw_end <= len(raw_content)):
                logger.warning(
                    "SemanticEvidenceAdapter fail-closed: invalid bounds [%d, %d] for content len %d",
                    raw_start,
                    raw_end,
                    len(raw_content),
                )
                return None

            matched_text = raw_content[raw_start:raw_end]
            pattern_id = decision.pattern_id or "SEM_GENERIC_FINDING"
            finding_id = f"{pattern_id}_{raw_start}_{raw_end}"

            category = decision.category or ScamCategory.UNKNOWN
            source = DetectionSource.DETERMINISTIC_RULE
            severity = decision.severity or FindingSeverity.LOW
            status = decision.status or FindingStatus.AMBIGUOUS_SIGNAL
            eligibility = decision.eligibility or CandidateEligibility.REJECTED
            rejection_reason = decision.rejection_reason
            explanation = decision.explanation or "Qualified semantic pattern match."
            context_required = decision.semantic_target

            item = EvidenceItem(
                id=pattern_id,
                category=category,
                source=source,
                severity=severity,
                status=status,
                matched_text=matched_text,
                explanation=explanation,
                context_required=context_required,
                finding_id=finding_id,
                contradicts_finding_ids=[],
                start_pos=raw_start,
                end_pos=raw_end,
                eligibility=eligibility,
                rejection_reason=rejection_reason,
            )
            return item
        except Exception as e:
            logger.warning("SemanticEvidenceAdapter adapt failed fail-closed: %s", e)
            return None

    @staticmethod
    def adapt_batch(decisions: list[QualificationDecision], raw_content: str) -> list[EvidenceItem]:
        """Converts a batch of QualificationDecision objects into Phase 2 compliant EvidenceItem list."""
        items: list[EvidenceItem] = []
        for d in decisions:
            item = SemanticEvidenceAdapter.adapt(d, raw_content)
            if item is not None:
                items.append(item)
        return items
