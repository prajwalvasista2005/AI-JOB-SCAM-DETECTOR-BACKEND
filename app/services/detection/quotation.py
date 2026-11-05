from collections import defaultdict
import re
from typing import Any
import unicodedata

from app.services.detection.models import (
    CandidateEligibility,
    DetectionSource,
    EvidenceItem,
    FindingSeverity,
    FindingStatus,
    ScamCategory,
)
from app.services.detection.patterns import (
    COMPANY_FEE_DISCLAIMER_RE,
    DISCLAIMER_IMMUNE_IDS,
)


# Fine-grained clause boundaries: sentence punctuation and coordinating/adversative conjunctions
CLAUSE_BOUNDARY_RE = re.compile(
    r"[\.\!\?\;\n]+"
    r"|,\s*(?:and|but|however|yet|while|whereas|although)\b"
    r"|\b(?:but|whereas|however|yet|although|in\s+reality|on\s+the\s+contrary)\b"
    r"|\band\s+(?:(?:the|our)\s+)?(?:candidates?|applicants?|job\s*seekers?|you|we|our\s+company|the\s+company|agency|consultancy)\s+(?:must|shall|need\s+to|should|have\s+to|are\s+required\s+to|will|collects?|charges?)\b",
    re.IGNORECASE,
)

# Clause-level warning framing: evaluates whether the governing clause is reported speech/scam warning
CLAUSE_WARNING_FRAMING_RE = re.compile(
    r"\b(?:scam\s+alert|fraud\s+alert)\b"
    r"|\b(?:warning|caution)\s*:\s*(?:scam\s+alert|fraud\s+alert|beware|do\s+not\s+(?:pay|transfer|remit|fall\s+for)|scammers?|fraudsters?|fake|unauthorized)\b"
    r"|\bbeware\s+of\s+(?:scams?|scammers?|fraudsters?|fraud|fake|unauthorized)\b"
    r"|\b(?:scammers?|fraudsters?|fake\s+recruiters?|unauthorized\s+(?:agencies|agents?|persons?)|unscrupulous\s+elements?)\s+(?:are\s+)?(?:falsely\s+)?(?:claims?|claiming|demands?|demanding|asks?|asking|collects?|collecting|promising|solicits?|soliciting|who\s+say|who\s+ask|who\s+falsely\s+claim)\b"
    r"|\b(?:falsely\s+(?:claiming|claimed|claim)|fake\s+claims?|fraudulent\s+claims?|do\s+not\s+fall\s+for|do\s+not\s+pay)\b"
    r"|\b(?:report\s+(?:them|scams?|fraud|suspicious)|should\s+report|please\s+report)\b",
    re.IGNORECASE,
)

# Action-linked explicit employer attribution check for Tier 1
TIER1_EMPLOYER_ACTION_RE = re.compile(
    # Alternative 1: Recipient of payment verb (with optional intervening fee noun phrase or amount)
    r"\b(?:pay|deposit|transfer|remit|send)\s+"
    r"(?:(?:a|an|the|any)?\s*(?:non[-\s]refundable|mandatory|compulsory)?\s*(?:registration|recruitment|placement|application|onboarding|documentation|processing)?\s*(?:fees?|charges?|deposits?|payments?)\s+)?"
    r"(?:(?:rs\.?|inr|\$|£|€)?\s*\d+(?:,\d+)*\s+)?"
    r"(?:directly\s+)?(?:to\s+|with\s+|into\s+)?"
    r"(?:our\s+company|the\s+company|our\s+firm|the\s+firm|our\s+organization|our\s+hr|us)\b"
    r"|\b(?:pay|deposit|remit)\s+(?:us)\s+"
    r"(?:(?:a|an|the)?\s*(?:non[-\s]refundable|mandatory|compulsory)?\s*)?"
    r"(?:registration|recruitment|placement|application|onboarding|documentation|processing)?\s*(?:fees?|charges?)\b"
    # Alternative 2: Subject/agent charging or collecting fee
    r"|\b(?:our\s+company|the\s+company|our\s+firm|our\s+organization|our\s+hr|we)\s+"
    r"(?:collects?|charges?|requires?|demands?|asks?\s+for|takes?)\s+"
    r"(?:(?:a|an|the|any)?\s*(?:non[-\s]refundable|mandatory|compulsory|upfront)?\s*)?"
    r"(?:(?:rs\.?|inr|\$|£|€)?\s*\d+(?:,\d+)*\s+)?"
    r"(?:registration|recruitment|placement|application|onboarding|documentation|processing)?\s*(?:fees?|charges?|deposits?|payments?)\b"
    # Alternative 3: Employer explicitly instructing/directing candidate to pay fee
    r"|\b(?:our\s+company|the\s+company|our\s+firm|our\s+organization|our\s+hr|we)\s+"
    r"(?:asks?|requires?|instructs?|directs?|tells?|demands?)\s+"
    r"(?:(?:them|candidates?|applicants?|you)\s+)?"
    r"(?:to\s+)?(?:pay|deposit|transfer|remit|send)\s+"
    r"(?:(?:a|an|the|any)?\s*(?:non[-\s]refundable|mandatory|compulsory|upfront)?\s*)?"
    r"(?:(?:rs\.?|inr|\$|£|€)?\s*\d+(?:,\d+)*\s+)?"
    r"(?:registration|recruitment|placement|application|onboarding|documentation|processing)?\s*(?:fees?|charges?|deposits?|payments?)\b",
    re.IGNORECASE,
)

# Action-linked explicit third-party agency / intermediary attribution check for Tier 2
TIER2_AGENCY_ACTION_RE = re.compile(
    # Alternative 1: Payment verb with optional preposition OR ditransitive direct recipient
    r"\b(?:pay|deposit|transfer|remit|send)\s+"
    r"(?:(?:rs\.?|inr|\$|£|€)?\s*\d+(?:,\d+)*\s+)?(?:directly\s+)?"
    r"(?:(?:to|with|into)\s+)?"
    r"(?:(?!pay\b|to\b)[a-zA-Z0-9_\-]+\s+){0,4}(?:agency|consultancy|placement\s+agency|staffing\s+firm|consultant|intermediary|third[-\s]party)\b"
    r"(?:\s+(?:(?:a|an|the|any)?\s*(?:non[-\s]refundable|mandatory|compulsory)?\s*(?:registration|recruitment|placement|application|onboarding|documentation|processing)?\s*(?:fees?|charges?|deposits?|payments?)|(?:rs\.?|inr|\$|£|€)?\s*\d+(?:,\d+)*))?"
    # Alternative 2: Subject/agent charging or collecting fee
    r"|\b(?:(?:our|the|[a-zA-Z0-9_\-]+)\s+)?(?:agency|consultancy|placement\s+agency|staffing\s+firm|consultant|third[-\s]party)\s+"
    r"(?:collects?|charges?|demands?|requires?|takes?)\s+"
    r"(?:(?:a|an|the|any)?\s*(?:non[-\s]refundable|mandatory|compulsory|upfront)?\s*)?"
    r"(?:(?:rs\.?|inr|\$|£|€)?\s*\d+(?:,\d+)*\s+)?"
    r"(?:registration|recruitment|placement|application|onboarding|documentation|processing)?\s*(?:fees?|charges?|deposits?|payments?)\b",
    re.IGNORECASE,
)

# Backward-compatibility alias for tests referencing AGENCY_EXPLICIT_RE or EMPLOYER_EXPLICIT_RE
EMPLOYER_EXPLICIT_RE = TIER1_EMPLOYER_ACTION_RE
AGENCY_EXPLICIT_RE = TIER2_AGENCY_ACTION_RE

# Explicit employer repudiation of third-party agencies in disclaimer
DISCLAIMER_AGENCY_REPUDIATION_RE = re.compile(
    r"\b(?:no\s+(?:agenc(?:y|ies)|consultanc(?:y|ies)|consultants?|agents?|third\s*part(?:y|ies))\s+(?:is|are)\s+authorized"
    r"|(?:do\s+not|does\s+not|never)\s+authorize\s+(?:any\s+)?(?:agenc(?:y|ies)|consultanc(?:y|ies)|consultants?|third\s*parties|agents?)"
    r"|(?:do\s+not|does\s+not|never)\s+(?:employ|hire|partner\s+with)\s+(?:any\s+)?(?:agenc(?:y|ies)|consultanc(?:y|ies)|consultants?))\b",
    re.IGNORECASE,
)

# Negation words that precede the matched phrase within the same clause
PRECEDING_NEGATION_RE = re.compile(
    r"\b(?:no|never|without|not|zero|free\s+of)\b(?:\s+\w+){0,4}\s*$",
    re.IGNORECASE,
)

# Direct quantifier negations acting on the fee as subject (e.g. 'No registration fee is mandatory')
DIRECT_FEE_NEGATION_RE = re.compile(
    r"\b(?:no|without|zero|free\s+of)\b(?:\s+(?:such|any|advance|upfront|mandatory))?\s*$",
    re.IGNORECASE,
)

# Negation patterns that follow the matched phrase within the same clause,
# permitting optional intervening specifiers or amounts (e.g. "of Rs. 500", "for this role")
SUCCEEDING_NEGATION_RE = re.compile(
    r"^\s*(?:(?:\S+\s+){0,5})?(?:(?:is|are|will\s+be|shall\s+be)\s+)?(?:not|never)\s+(?:required|needed|charged|asked|applicable|mandatory|collected|necessary)\b"
    r"|^\s*(?:(?:\S+\s+){0,5})?(?:is|are|will\s+be)\s+(?:waived|free|optional|zero)\b"
    r"|^\s*(?:(?:\S+\s+){0,5})?(?:is|are)\s+(?:completely\s+free|not\s+applicable)\b",
    re.IGNORECASE,
)

# Phrases that explicitly assert an active payment requirement (override false preceding negation)
ASSERTIVE_PAYMENT_RE = re.compile(
    r"\b(?:non[-\s]refundable|mandatory|compulsory|must\s+pay|must\s+be\s+paid)\b",
    re.IGNORECASE,
)

# Common abbreviations with trailing periods that must not trigger sentence boundary splitting
ABBREVIATIONS_RE = re.compile(
    r"\b(Rs|INR|Inc|Corp|Ltd|Co|Pvt|No|approx|Mr|Mrs|Ms|Dr|e\.g|i\.e|vs)\.",
    re.IGNORECASE,
)
ABBREVIATIONS_PATTERN = ABBREVIATIONS_RE

# Sentence terminators: period (not followed by digit), exclamation, question, semicolon, newline
SENTENCE_TERMINATOR_RE = re.compile(r"\.(?!\d)|[\!\?\;\n]+")

COORDINATE_SPLIT_RE = re.compile(r"(?<!\d),(?!\d)\s*(?:and\s+)?|\band\s+", re.IGNORECASE)

# Expanded payment & collection action verbs for isolation stopping
PAYMENT_ACTION_STOP_RE = re.compile(
    r"\b(?:pay|deposit|transfer|remit|send|collect|charge|receive|demand|require|asks?\s+for)\b",
    re.IGNORECASE,
)

ADVERSATIVE_BOUNDARY_RE = re.compile(
    r"\b(?:but|however|whereas|yet|although)\b",
    re.IGNORECASE,
)

# Governed subject/modal candidate at the beginning of a predicate
SUBJECT_MODAL_RE = re.compile(
    r"^\s*((?:candidates?|applicants?|job\s*seekers?|employees?|students?|everyone|all\s+candidates?)\s+"
    r"(?:must|need\s+to|have\s+to|should|shall|are\s+required\s+to)?)\b",
    re.IGNORECASE,
)

# Detect if the current segment already introduces an explicit subject/actor
CURRENT_ACTOR_RE = re.compile(
    r"\b(?:our\s+company|the\s+company|our\s+firm|the\s+firm|our\s+organization|our\s+hr|we|employer|recruiter|"
    r"(?:[a-zA-Z0-9_\-]+\s+)?(?:agency|consultancy|placement\s+agency|staffing\s+firm)|"
    r"candidates?|applicants?|job\s*seekers?|everyone|all\s+candidates?)\b",
    re.IGNORECASE,
)

# Unrelated idioms containing "no" that do NOT negate a payment demand
UNRELATED_NEGATION_IDIOM_RE = re.compile(
    r"\b(?:no\s+doubt|no\s+mistake|no\s+exceptions?|make\s+no\s+error)\b",
    re.IGNORECASE,
)


def mask_abbreviations(text: str) -> str:
    """Masks periods in abbreviations with an underscore to preserve exact string offsets."""
    return ABBREVIATIONS_RE.sub(lambda m: m.group(0)[:-1] + "_", text)


class DocumentContext:
    """
    Per-analysis-request context holding precomputed text masking and span-level clause caches.
    Lifecycle:
        Instantiated at the start of analyze_job_posting(raw_content).
        Garbage-collected when analyze_job_posting returns.
    Guarantees:
        - Precomputes mask_abbreviations(raw_content) exactly once per request.
        - Memoizes (pre_clause, post_clause) by exact coordinates (start_pos, end_pos).
        - Hard ceiling of 500 cache entries; subsequent entries bypass caching safely.
        - Zero cross-request state leakage (no global/module-level storage).
    """

    MAX_CACHE_ENTRIES: int = 500

    def __init__(self, raw_content: str):
        self.raw_content: str = raw_content
        self.masked_content: str = mask_abbreviations(raw_content)
        self._clause_bounds_cache: dict[tuple[int, int], tuple[str, str]] = {}
        self._sentence_bounds_cache: dict[tuple[int, int], tuple[int, int]] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0


def _normalize_text(text: str) -> str:
    """Normalizes text by removing diacritics, lowering case, and collapsing whitespace."""
    normalized = unicodedata.normalize("NFKD", text)
    cleaned = "".join(c for c in normalized if not unicodedata.combining(c))
    return " ".join(cleaned.lower().split())


def extract_action_context(
    raw_content: str,
    start_pos: int,
    end_pos: int,
    doc_ctx: DocumentContext | None = None,
    return_sentence_bounds: bool = False,
) -> tuple[str, str] | tuple[int, int, str, str]:
    """
    Extracts preceding and succeeding clause context up to genuine sentence boundaries,
    coordinating conjunctions, and adversatives, safely preserving currency, legal entity,
    and title abbreviations like 'Rs.', 'Pvt. Ltd.', 'Inc.', etc. without character offset drift.
    """
    # Defensive coordinate validation: safely handle invalid/out-of-range coordinates
    is_valid_coords = (
        isinstance(start_pos, int)
        and isinstance(end_pos, int)
        and not isinstance(start_pos, bool)
        and not isinstance(end_pos, bool)
        and 0 <= start_pos < end_pos <= len(raw_content)
    )

    if not is_valid_coords:
        if return_sentence_bounds:
            return 0, 0, "", ""
        return "", ""

    cache_key = (start_pos, end_pos)
    if doc_ctx is not None:
        if cache_key in doc_ctx._clause_bounds_cache:
            doc_ctx.cache_hits += 1
            if return_sentence_bounds:
                s_start, s_end = doc_ctx._sentence_bounds_cache[cache_key]
                c_pre, c_post = doc_ctx._clause_bounds_cache[cache_key]
                return s_start, s_end, c_pre, c_post
            return doc_ctx._clause_bounds_cache[cache_key]
        doc_ctx.cache_misses += 1

    # 1. Mask abbreviations (using precomputed masked_content if doc_ctx available)
    masked = doc_ctx.masked_content if doc_ctx is not None else mask_abbreviations(raw_content)

    # 2. Delimit enclosing sentence bounds using masked string
    last_stop = 0
    for m in SENTENCE_TERMINATOR_RE.finditer(masked[:start_pos]):
        last_stop = m.end()
    sent_start = last_stop

    next_stop = len(raw_content)
    m = SENTENCE_TERMINATOR_RE.search(masked[end_pos:])
    if m:
        next_stop = end_pos + m.start()

    # 3. Slice ORIGINAL raw_content (preserving exact abbreviation text)
    pre_sentence = raw_content[sent_start:start_pos]
    post_sentence = raw_content[end_pos:next_stop]

    # 4. Split pre_sentence into coordinate segments
    coord_segments = COORDINATE_SPLIT_RE.split(pre_sentence)
    current_action_prefix = coord_segments[-1]

    # 5. Actor precedence: check if current segment introduces an explicit actor
    has_own_actor = bool(CURRENT_ACTOR_RE.search(current_action_prefix))
    inherited_subject_modal = ""

    if not has_own_actor:
        for seg in reversed(coord_segments[:-1]):
            # Stop Condition 1: Prior payment or collection action -> isolate completely
            if PAYMENT_ACTION_STOP_RE.search(seg):
                break
            # Stop Condition 2: Adversative clause boundary -> isolate completely
            if ADVERSATIVE_BOUNDARY_RE.search(seg):
                break
            # Target: Subject/modal root -> inherit only the root
            sub_m = SUBJECT_MODAL_RE.match(seg)
            if sub_m:
                inherited_subject_modal = sub_m.group(1).strip()
                break

    # 6. Form pre_clause
    if inherited_subject_modal:
        clean_prefix = current_action_prefix.strip()
        pre_clause = f"{inherited_subject_modal} {clean_prefix}".strip()
    else:
        pre_clause = current_action_prefix.strip()

    # 7. Form post_clause up to next coordinate split or sentence stop
    post_segments = COORDINATE_SPLIT_RE.split(post_sentence)
    post_clause = post_segments[0].strip() if post_segments else ""

    clause_res = (pre_clause, post_clause)

    if doc_ctx is not None:
        if len(doc_ctx._clause_bounds_cache) < doc_ctx.MAX_CACHE_ENTRIES:
            doc_ctx._clause_bounds_cache[cache_key] = clause_res
            doc_ctx._sentence_bounds_cache[cache_key] = (sent_start, next_stop)

    if return_sentence_bounds:
        return sent_start, next_stop, pre_clause, post_clause

    return clause_res


def _extract_clause_bounds(
    raw_content: str,
    start_pos: int,
    end_pos: int,
    doc_ctx: DocumentContext | None = None,
) -> tuple[str, str]:
    """Backward-compatible wrapper around extract_action_context."""
    return extract_action_context(raw_content, start_pos, end_pos, doc_ctx=doc_ctx)


def is_occurrence_negated(prefix_clause: str, suffix_clause: str, matched_text: str) -> bool:
    """
    Evaluates whether an individual occurrence of matched_text is negated
    by checking surrounding clause context on both sides.
    """
    # 1. Direct subject fee negation overrides predicate (e.g. 'No registration fee is mandatory')
    if DIRECT_FEE_NEGATION_RE.search(prefix_clause):
        return True

    # 2. Succeeding clause negation (e.g. 'Registration fee of Rs. 500 is not required', 'fee is waived',
    # 'fee is not required and is non-refundable')
    if SUCCEEDING_NEGATION_RE.search(suffix_clause):
        return True

    # 3. Explicit free / zero cost in the enclosing clause
    combined_context = f"{prefix_clause} {matched_text} {suffix_clause}"
    if re.search(
        r"\b(?:free\s+of\s+(?:cost|charge)|at\s+no\s+cost|zero\s+(?:cost|fee|charge)|without\s+any\s+(?:fee|charge|cost))\b",
        combined_context,
        re.IGNORECASE,
    ):
        return True

    # 4. Assertive payment override:
    # If the clause explicitly asserts that payment is non-refundable, mandatory, or must be paid,
    # and the fee itself is NOT negated by a preceding negation, do NOT negate.
    if ASSERTIVE_PAYMENT_RE.search(suffix_clause) or ASSERTIVE_PAYMENT_RE.search(matched_text):
        if UNRELATED_NEGATION_IDIOM_RE.search(prefix_clause) or not PRECEDING_NEGATION_RE.search(prefix_clause):
            return False

    # 5. General preceding clause negation (e.g. 'does not charge registration fees', 'never ask for fees')
    # Filter out unrelated idioms like 'make no mistake' or 'no doubt'
    if PRECEDING_NEGATION_RE.search(prefix_clause):
        if UNRELATED_NEGATION_IDIOM_RE.search(prefix_clause):
            return False
        return True

    return False


def validate_evidence_quotation(
    finding: EvidenceItem,
    raw_content: str,
    doc_ctx: DocumentContext | None = None,
) -> EvidenceItem:
    """
    Validates the quotation and context of an evidence finding against raw input content.
    Enforces:
      1. Caller-Owned Object Immutability: Operates on a deep copy of finding.
      2. Exact Verbatim Substring Matching: Fabricated, mismatched, or out-of-bounds quotes
         cannot qualify for EXPLICIT_CLAIM status, have matched_text cleared, and are capped
         at HIGH severity with CONTEXTUAL_INFERENCE status.
      3. Precise Character Offset Alignment: If start_pos and end_pos are present, verifies
         they exactly span the matched quote. If missing, discovers offsets directly on
         raw_content without .lower() length drift.
      4. Bidirectional Negation: Negation appearing before or after the quote suppresses
         the finding from risk scoring (reclassifying as INFORMATIONAL / AMBIGUOUS_SIGNAL).
      5. Provenance Preservation: Preserves source while accurately reflecting textual grounding.
    """
    res = finding.model_copy(deep=True)

    if not res.matched_text or not res.matched_text.strip():
        # Finding has no matched quote; ensure it is marked as CONTEXTUAL_INFERENCE
        # and cap severity to prevent unquoted findings from triggering CRITICAL
        if res.status == FindingStatus.EXPLICIT_CLAIM:
            res.status = FindingStatus.CONTEXTUAL_INFERENCE
        if res.severity == FindingSeverity.CRITICAL:
            res.severity = FindingSeverity.HIGH
        res.matched_text = None
        res.start_pos = None
        res.end_pos = None
        res.eligibility = CandidateEligibility.NON_SCORABLE_CONTEXTUAL
        return res

    quote = res.matched_text.strip()

    # Verify existing start_pos and end_pos if provided
    if res.start_pos is not None and res.end_pos is not None:
        if 0 <= res.start_pos < res.end_pos <= len(raw_content):
            slice_text = raw_content[res.start_pos:res.end_pos]
            if slice_text.strip().lower() == quote.lower():
                # Overwrite matched_text with exact raw slice
                res.matched_text = slice_text
                occurrences = [(res.start_pos, res.end_pos)]
            else:
                # Mismatched span content -> fail closed
                res.status = FindingStatus.CONTEXTUAL_INFERENCE
                res.matched_text = None
                res.start_pos = None
                res.end_pos = None
                res.eligibility = CandidateEligibility.NON_SCORABLE_CONTEXTUAL
                if res.severity == FindingSeverity.CRITICAL:
                    res.severity = FindingSeverity.HIGH
                return res
        else:
            # Out of bounds span -> fail closed
            res.status = FindingStatus.CONTEXTUAL_INFERENCE
            res.matched_text = None
            res.start_pos = None
            res.end_pos = None
            res.eligibility = CandidateEligibility.NON_SCORABLE_CONTEXTUAL
            if res.severity == FindingSeverity.CRITICAL:
                res.severity = FindingSeverity.HIGH
            return res
    else:
        # Search directly on raw_content using re.finditer without .lower() index math
        occurrences = []
        escaped_quote = re.escape(quote)
        for m in re.finditer(escaped_quote, raw_content, re.IGNORECASE):
            occurrences.append((m.start(), m.end()))

        # If not found directly, try flexible whitespace
        if not occurrences:
            words = [re.escape(w) for w in quote.split()]
            if words:
                ws_pattern = r"\s+".join(words)
                for m in re.finditer(ws_pattern, raw_content, re.IGNORECASE):
                    occurrences.append((m.start(), m.end()))

        if not occurrences:
            res.status = FindingStatus.CONTEXTUAL_INFERENCE
            res.matched_text = None
            res.start_pos = None
            res.end_pos = None
            res.eligibility = CandidateEligibility.NON_SCORABLE_CONTEXTUAL
            if res.severity == FindingSeverity.CRITICAL:
                res.severity = FindingSeverity.HIGH
            return res

        if len(occurrences) == 1:
            # Deterministically resolved to the unique occurrence
            res.start_pos = occurrences[0][0]
            res.end_pos = occurrences[0][1]
            res.matched_text = raw_content[res.start_pos:res.end_pos]
        else:
            # Multiple identical quotations exist without caller-provided offsets.
            # Fail safely: leave offsets unresolved (None) to avoid arbitrary list-order guessing.
            res.start_pos = None
            res.end_pos = None
            # Keep matched_text as the trimmed quote string

    has_active_unnegated_claim = False
    all_negated = True

    for occ_s, occ_e in occurrences:
        pre_clause, post_clause = _extract_clause_bounds(raw_content, occ_s, occ_e, doc_ctx=doc_ctx)
        occ_quote = raw_content[occ_s:occ_e]

        # Check intra-clause warning framing (reported speech in fraud warning)
        if CLAUSE_WARNING_FRAMING_RE.search(pre_clause):
            is_neg = True
        else:
            is_neg = is_occurrence_negated(pre_clause, post_clause, occ_quote)

        if not is_neg:
            all_negated = False
            has_active_unnegated_claim = True

    if all_negated:
        # All occurrences are negated or reported speech
        res.severity = FindingSeverity.INFORMATIONAL
        res.status = FindingStatus.AMBIGUOUS_SIGNAL
        res.eligibility = CandidateEligibility.NON_SCORABLE_INFORMATIONAL
        res.context_required = "Negated or warning-framed claim; non-scorable."
        res.explanation = f"Quotation was evaluated in context and determined to be negated or part of a scam warning: '{res.matched_text}'."
    elif has_active_unnegated_claim:
        # Preserve intrinsically ambiguous findings; only promote to EXPLICIT_CLAIM if not already AMBIGUOUS_SIGNAL
        if res.status != FindingStatus.AMBIGUOUS_SIGNAL:
            res.status = FindingStatus.EXPLICIT_CLAIM

    return res


def reconcile_contradictions(
    items: list[EvidenceItem],
    raw_content: str,
    doc_ctx: DocumentContext | None = None,
) -> list[EvidenceItem]:
    """
    Reconciles multi-paragraph contradiction relationships using the Five-Tier
    Entity and Obligation Framework.
    Guarantees:
      - Caller-Owned Object Immutability: Operates entirely on deep copies of items.
      - Symmetric bidirectional links (zero one-sided edges).
      - Zero dangling references (edges only between active finding IDs).
      - Unique instance identities per scan ({pattern_id}_{occurrence_index}).
      - Deterministic collision-free ID assignment even with pre-existing finding_ids.
      - Preservation of original severity and status on demands (CRITICAL remains 65).
      - Preservation of uncertainty for unverified / unknown charging entities (Tier 5).
    """
    result_items: list[EvidenceItem] = [it.model_copy(deep=True) for it in items]
    existing_finding_ids: set[str] = {it.finding_id for it in result_items if it.finding_id}

    # 1. Discover company no-fee disclaimers in document
    existing_disclaimer_spans: set[tuple[int, int]] = {
        (it.start_pos, it.end_pos)
        for it in result_items
        if it.id == "DISCLAIMER_COMPANY_NO_FEE" and it.start_pos is not None and it.end_pos is not None
    }
    existing_disclaimer_quotes: set[str] = {
        it.matched_text
        for it in result_items
        if it.id == "DISCLAIMER_COMPANY_NO_FEE" and (it.start_pos is None or it.end_pos is None) and it.matched_text
    }

    disclaimer_matches = list(COMPANY_FEE_DISCLAIMER_RE.finditer(raw_content))
    for m in disclaimer_matches:
        d_span = (m.start(), m.end())
        d_quote = m.group(0).strip()
        if d_span in existing_disclaimer_spans:
            continue
        if not existing_disclaimer_spans and d_quote in existing_disclaimer_quotes:
            continue

        d_idx = 1
        while f"DISCLAIMER_COMPANY_NO_FEE_{d_idx}" in existing_finding_ids:
            d_idx += 1
        new_fid = f"DISCLAIMER_COMPANY_NO_FEE_{d_idx}"
        existing_finding_ids.add(new_fid)
        existing_disclaimer_spans.add(d_span)

        disc_item = EvidenceItem(
            id="DISCLAIMER_COMPANY_NO_FEE",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.INFORMATIONAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text=d_quote,
            explanation="Company policy statement disclaiming registration or recruitment fees.",
            finding_id=new_fid,
            start_pos=m.start(),
            end_pos=m.end(),
            eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
        )
        result_items.append(disc_item)

    # 2. Deterministically resolve missing start_pos / end_pos for items where unambiguous
    claimed_spans: set[tuple[int, int]] = {
        (it.start_pos, it.end_pos)
        for it in result_items
        if it.start_pos is not None and it.end_pos is not None
    }

    missing_by_quote: dict[str, list[EvidenceItem]] = defaultdict(list)
    for it in result_items:
        if (it.start_pos is None or it.end_pos is None) and it.matched_text:
            missing_by_quote[it.matched_text.strip()].append(it)

    for quote, it_list in missing_by_quote.items():
        all_occ: list[tuple[int, int]] = [
            (m.start(), m.end())
            for m in re.finditer(re.escape(quote), raw_content, re.IGNORECASE)
        ]
        if not all_occ:
            words = [re.escape(w) for w in quote.split()]
            if words:
                ws_pattern = r"\s+".join(words)
                all_occ = [
                    (m.start(), m.end())
                    for m in re.finditer(ws_pattern, raw_content, re.IGNORECASE)
                ]

        unclaimed = [s for s in all_occ if s not in claimed_spans]
        if len(it_list) == 1 and len(unclaimed) == 1:
            it = it_list[0]
            it.start_pos, it.end_pos = unclaimed[0]
            it.matched_text = raw_content[it.start_pos:it.end_pos]
            claimed_spans.add(unclaimed[0])
        else:
            # Ambiguous: multiple unclaimed occurrences or multiple items missing offsets.
            # Leave start_pos and end_pos as None to prevent false positional association.
            for it in it_list:
                it.start_pos = None
                it.end_pos = None

    # 3. Ensure every item has a unique, deterministic finding_id
    for it in result_items:
        if not it.finding_id:
            c = 1
            while f"{it.id}_{c}" in existing_finding_ids:
                c += 1
            it.finding_id = f"{it.id}_{c}"
            existing_finding_ids.add(it.finding_id)

    # 4. Discover Candidate Contradiction Edges under Five-Tier Framework
    disclaimers = [it for it in result_items if it.id == "DISCLAIMER_COMPANY_NO_FEE"]
    candidate_edges: set[tuple[str, str]] = set()

    for disc in disclaimers:
        for f in result_items:
            # Exclude disclaimers themselves and immune patterns
            if f.id == "DISCLAIMER_COMPANY_NO_FEE":
                continue
            if f.id in DISCLAIMER_IMMUNE_IDS:
                continue
            if f.severity == FindingSeverity.INFORMATIONAL:
                continue
            if f.category != ScamCategory.ADVANCE_FEE:
                continue
            if f.eligibility in (
                CandidateEligibility.REJECTED,
                CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
                CandidateEligibility.NON_SCORABLE_CONTEXTUAL,
            ):
                continue

            # Check Entity / Obligation Tier for advance fee demands
            if f.id == "ADV_FEE_REGISTRATION":
                # Fail-safe: if offsets could not be established deterministically, do NOT guess links across paragraphs
                if f.start_pos is None or f.end_pos is None:
                    if not f.context_required:
                        f.context_required = (
                            "Unverified charging entity: An upfront fee is demanded, but the receiving entity is "
                            "unspecified. A company disclaimer exists elsewhere in the document, but cannot be "
                            "assumed to cover this unverified demand. Verify with the employer directly before "
                            "transferring funds."
                        )
                    continue

                pre_clause, post_clause = _extract_clause_bounds(raw_content, f.start_pos, f.end_pos, doc_ctx=doc_ctx)
                enclosing_context = f"{pre_clause} {f.matched_text or ''} {post_clause}"

                is_employer = bool(TIER1_EMPLOYER_ACTION_RE.search(enclosing_context))
                is_agency = bool(TIER2_AGENCY_ACTION_RE.search(enclosing_context))
                disc_repudiates_agency = bool(DISCLAIMER_AGENCY_REPUDIATION_RE.search(disc.matched_text or ""))

                if is_agency:
                    if disc_repudiates_agency:
                        # Tier 2 Scenario A: Agency demand explicitly repudiated by employer disclaimer
                        candidate_edges.add((f.finding_id, disc.finding_id))
                    else:
                        # Tier 2 Scenario B: Generic employer disclaimer does not link to agency demand
                        if not f.context_required:
                            f.context_required = (
                                "Independent agency fee: An upfront fee is demanded by a third-party agency or "
                                "consultancy. The company disclaims direct recruitment fees, but does not explicitly "
                                "repudiate external agency fees. Verify whether this agency is officially authorized "
                                "before paying."
                            )
                elif is_employer:
                    # Tier 1: Employer intake fee contradicted by employer disclaimer
                    candidate_edges.add((f.finding_id, disc.finding_id))
                else:
                    # Tier 5: Unknown/unstated charging entity -> preserve uncertainty
                    if not f.context_required:
                        f.context_required = (
                            "Unverified charging entity: An upfront fee is demanded, but the receiving entity is "
                            "unspecified. A company disclaimer exists elsewhere in the document, but cannot be "
                            "assumed to cover this unverified demand. Verify with the employer directly before "
                            "transferring funds."
                        )

    # 4. Filter validated edges against active finding registry (Zero Dangling Pointers)
    valid_ids = {it.finding_id for it in result_items if it.finding_id}
    validated_edges = {
        (src, dst) for src, dst in candidate_edges
        if src in valid_ids and dst in valid_ids and src != dst
    }

    # 5. Build symmetric bidirectional adjacency map
    adj: dict[str, set[str]] = defaultdict(set)
    for src, dst in validated_edges:
        adj[src].add(dst)
        adj[dst].add(src)

    # 6. Assign sorted deterministic relationships and update human-readable context
    item_by_id = {it.finding_id: it for it in result_items if it.finding_id}
    for it in result_items:
        if it.finding_id and it.finding_id in adj:
            target_ids = sorted(adj[it.finding_id])
            it.contradicts_finding_ids = target_ids

            # Update context_required
            target_quotes = [
                item_by_id[tid].matched_text
                for tid in target_ids
                if tid in item_by_id and item_by_id[tid].matched_text
            ]
            target_summary = ", ".join(f"'{q}'" for q in target_quotes)
            if it.id == "DISCLAIMER_COMPANY_NO_FEE":
                it.context_required = f"Contradicts upfront payment demand {target_summary}."
            else:
                it.context_required = (
                    f"Contradiction detected: Conflicts with company disclaimer ({target_summary}). "
                    f"Verify with official employer before transferring funds."
                )
        else:
            it.contradicts_finding_ids = []

    return result_items
