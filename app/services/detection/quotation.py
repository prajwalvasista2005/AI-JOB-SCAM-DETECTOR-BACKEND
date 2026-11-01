from collections import defaultdict
import re
from typing import Any
import unicodedata

from app.services.detection.models import (
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
    r"[\.\!\?\;\n]+|,\s*(?:and|but|however|yet|while|whereas|although)\b|\b(?:however|in\s+reality|on\s+the\s+contrary)\b",
    re.IGNORECASE,
)

# Clause-level warning framing: evaluates whether the governing clause is reported speech/scam warning
CLAUSE_WARNING_FRAMING_RE = re.compile(
    r"\b(?:scam\s+alert|fraud\s+alert)\b"
    r"|\b(?:warning|caution)\s*:\s*(?:scam\s+alert|fraud\s+alert|beware|do\s+not\s+(?:pay|transfer|remit|fall\s+for)|scammers?|fraudsters?|fake|unauthorized)\b"
    r"|\bbeware\s+of\s+(?:scams?|scammers?|fraudsters?|fraud|fake|unauthorized)\b"
    r"|\b(?:scammers?|fraudsters?|fake\s+recruiters?|unauthorized\s+(?:agencies|agents?|persons?)|unscrupulous\s+elements?)\s+(?:are\s+)?(?:falsely\s+)?(?:claims?|claiming|demands?|demanding|asks?|asking|collects?|collecting|promising|solicits?|soliciting|who\s+say|who\s+ask|who\s+falsely\s+claim)\b"
    r"|\b(?:falsely\s+(?:claiming|claimed|claim)|fake\s+claims?|fraudulent\s+claims?|do\s+not\s+fall\s+for|do\s+not\s+pay)\b",
    re.IGNORECASE,
)

# Explicit employer attribution check for Tier 1
EMPLOYER_EXPLICIT_RE = re.compile(
    r"\b(?:our\s+company|the\s+company|we|us|our\s+firm|our\s+organization|our\s+hr|company\s+charges?)\b"
    r"|\bpay\s+(?:to\s+)?(?:our\s+company|the\s+company|us|the\s+firm)\b",
    re.IGNORECASE,
)

# Explicit third-party agency / intermediary attribution check for Tier 2
AGENCY_EXPLICIT_RE = re.compile(
    r"\b(?:agenc(?:y|ies)|consultanc(?:y|ies)|consultants?|placement\s+agency|staffing\s+firm|third[-\s]party\s+agency|partner\s+agency)\b",
    re.IGNORECASE,
)

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

# Common abbreviations that contain periods and should not trigger premature clause boundary splitting
ABBREVIATIONS_PATTERN = re.compile(
    r"\b(Rs|INR|Inc|Corp|Ltd|Co|No|approx|Mr|Mrs|Ms|Dr|e\.g|i\.e)\.",
    re.IGNORECASE,
)

# Unrelated idioms containing "no" that do NOT negate a payment demand
UNRELATED_NEGATION_IDIOM_RE = re.compile(
    r"\b(?:no\s+doubt|no\s+mistake|no\s+exceptions?|make\s+no\s+error)\b",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    """Normalizes text by removing diacritics, lowering case, and collapsing whitespace."""
    normalized = unicodedata.normalize("NFKD", text)
    cleaned = "".join(c for c in normalized if not unicodedata.combining(c))
    return " ".join(cleaned.lower().split())


def _extract_clause_bounds(raw_content: str, start_pos: int, end_pos: int) -> tuple[str, str]:
    """
    Extracts preceding and succeeding clause context up to genuine sentence boundaries
    and coordinating/adversative conjunctions, safely preserving currency, legal entity,
    and title abbreviations like 'Rs.', 'Inc.', etc.
    """
    pre_text = raw_content[max(0, start_pos - 150) : start_pos]
    post_text = raw_content[end_pos : min(len(raw_content), end_pos + 150)]

    # Protect abbreviations from premature period splitting
    protected_pre = ABBREVIATIONS_PATTERN.sub(r"\1__DOT__", pre_text)
    protected_post = ABBREVIATIONS_PATTERN.sub(r"\1__DOT__", post_text)

    # Split on sentence boundaries and coordinating/adversative conjunctions
    pre_clause = CLAUSE_BOUNDARY_RE.split(protected_pre)[-1].replace("__DOT__", ".")
    post_clause = CLAUSE_BOUNDARY_RE.split(protected_post)[0].replace("__DOT__", ".")

    return pre_clause, post_clause


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


def validate_evidence_quotation(finding: EvidenceItem, raw_content: str) -> EvidenceItem:
    """
    Validates the quotation and context of an evidence finding against raw input content.
    Enforces:
      1. Verbatim Substring Matching: Fabricated or non-matching AI quotations cannot qualify
         for EXPLICIT_CLAIM status and are capped at HIGH severity.
      2. Bidirectional Negation: Negation appearing before or after the quote suppresses
         the finding from risk scoring (reclassifying as INFORMATIONAL / AMBIGUOUS_SIGNAL).
      3. Provenance Preservation: Preserves source while accurately reflecting textual grounding.
    """
    if not finding.matched_text or not finding.matched_text.strip():
        # Finding has no matched quote; ensure it is marked as CONTEXTUAL_INFERENCE
        # and cap severity to prevent unquoted findings from triggering CRITICAL
        if finding.status == FindingStatus.EXPLICIT_CLAIM:
            finding.status = FindingStatus.CONTEXTUAL_INFERENCE
        if finding.severity == FindingSeverity.CRITICAL:
            finding.severity = FindingSeverity.HIGH
        finding.matched_text = None
        return finding

    quote = finding.matched_text.strip()
    norm_content = _normalize_text(raw_content)
    norm_quote = _normalize_text(quote)

    # 1. Exact verbatim match check
    if norm_quote not in norm_content:
        # Quote does not exist in content; downgrade to CONTEXTUAL_INFERENCE
        finding.status = FindingStatus.CONTEXTUAL_INFERENCE
        finding.matched_text = None
        if finding.severity == FindingSeverity.CRITICAL:
            finding.severity = FindingSeverity.HIGH
        return finding

    # 2. Occurrence Context Analysis
    if finding.start_pos is not None and finding.end_pos is not None:
        occurrences = [(finding.start_pos, finding.end_pos)]
    else:
        raw_lower = raw_content.lower()
        quote_lower = quote.lower()
        occurrences = []
        start = 0
        while True:
            pos = raw_lower.find(quote_lower, start)
            if pos == -1:
                break
            occurrences.append((pos, pos + len(quote_lower)))
            start = pos + len(quote_lower)

        if not occurrences:
            # Matched via normalized whitespace but not raw lower; search using regex
            escaped = re.escape(quote)
            for m in re.finditer(escaped, raw_content, re.IGNORECASE):
                occurrences.append((m.start(), m.end()))

    if not occurrences:
        finding.status = FindingStatus.CONTEXTUAL_INFERENCE
        finding.matched_text = None
        if finding.severity == FindingSeverity.CRITICAL:
            finding.severity = FindingSeverity.HIGH
        return finding

    has_active_unnegated_claim = False
    all_negated = True

    for start_pos, end_pos in occurrences:
        pre_clause, post_clause = _extract_clause_bounds(raw_content, start_pos, end_pos)

        # Check intra-clause warning framing (reported speech in fraud warning)
        if CLAUSE_WARNING_FRAMING_RE.search(pre_clause):
            is_neg = True
        else:
            is_neg = is_occurrence_negated(pre_clause, post_clause, quote)

        if is_neg:
            pass
        else:
            all_negated = False
            has_active_unnegated_claim = True

    if all_negated:
        # All occurrences are negated or reported speech
        finding.severity = FindingSeverity.INFORMATIONAL
        finding.status = FindingStatus.AMBIGUOUS_SIGNAL
        finding.explanation = f"Quotation was evaluated in context and determined to be negated or part of a scam warning: '{quote}'."
    elif has_active_unnegated_claim:
        # Preserve intrinsically ambiguous findings; only promote to EXPLICIT_CLAIM if not already AMBIGUOUS_SIGNAL
        if finding.status != FindingStatus.AMBIGUOUS_SIGNAL:
            finding.status = FindingStatus.EXPLICIT_CLAIM

    return finding


def reconcile_contradictions(items: list[EvidenceItem], raw_content: str) -> list[EvidenceItem]:
    """
    Reconciles multi-paragraph contradiction relationships using the Five-Tier
    Entity and Obligation Framework.
    Guarantees:
      - Symmetric bidirectional links (zero one-sided edges).
      - Zero dangling references (edges only between active finding IDs).
      - Unique instance identities per scan ({pattern_id}_{occurrence_index}).
      - Preservation of original severity and status on demands (CRITICAL remains 65).
      - Preservation of uncertainty for unverified / unknown charging entities (Tier 5).
    """
    result_items: list[EvidenceItem] = list(items)

    # 1. Discover company no-fee disclaimers in document
    existing_disclaimer_quotes = {
        it.matched_text for it in result_items if it.id == "DISCLAIMER_COMPANY_NO_FEE"
    }
    disclaimer_matches = list(COMPANY_FEE_DISCLAIMER_RE.finditer(raw_content))

    d_idx = 1
    for m in disclaimer_matches:
        d_quote = m.group(0).strip()
        if d_quote not in existing_disclaimer_quotes:
            disc_item = EvidenceItem(
                id="DISCLAIMER_COMPANY_NO_FEE",
                category=ScamCategory.ADVANCE_FEE,
                source=DetectionSource.DETERMINISTIC_RULE,
                severity=FindingSeverity.INFORMATIONAL,
                status=FindingStatus.EXPLICIT_CLAIM,
                matched_text=d_quote,
                explanation="Company policy statement disclaiming registration or recruitment fees.",
                finding_id=f"DISCLAIMER_COMPANY_NO_FEE_{d_idx}",
                start_pos=m.start(),
                end_pos=m.end(),
            )
            result_items.append(disc_item)
            existing_disclaimer_quotes.add(d_quote)
            d_idx += 1

    # 2. Ensure every item has a unique, deterministic finding_id
    id_counts: dict[str, int] = defaultdict(int)
    for it in result_items:
        if not it.finding_id:
            id_counts[it.id] += 1
            it.finding_id = f"{it.id}_{id_counts[it.id]}"

    # 3. Discover Candidate Contradiction Edges under Five-Tier Framework
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

            # Check Entity / Obligation Tier for advance fee demands
            if f.id == "ADV_FEE_REGISTRATION":
                # Extract sentence/clause context for entity attribution
                f_start = f.start_pos if f.start_pos is not None else (raw_content.find(f.matched_text) if f.matched_text else -1)
                if f_start != -1:
                    pre_clause, post_clause = _extract_clause_bounds(
                        raw_content,
                        f_start,
                        f_start + (len(f.matched_text) if f.matched_text else 0),
                    )
                    enclosing_context = f"{pre_clause} {f.matched_text or ''} {post_clause}"
                else:
                    enclosing_context = f.matched_text or ""

                is_employer = bool(EMPLOYER_EXPLICIT_RE.search(enclosing_context))
                is_agency = bool(AGENCY_EXPLICIT_RE.search(enclosing_context))
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
