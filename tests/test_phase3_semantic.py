"""
Phase 3 — Fix 2 Semantic Qualification Pipeline & Invariants Unit Test Suite.
Verifies:
  1. The 10 focused rectification tests (T1-T10) from Section 12.3 of V5.2 spec.
  2. Bounded sub-clause isolation & action-scoped local negation.
  3. Reverse UPI active demand vs. category-preserving safety advisory.
  4. Credential harvesting active demand vs. category-preserving advisory vs. self-auth.
  5. Competing actions & mixed actor sentences.
  6. Historical payment report, discretionary payment, descriptive framing, and non-financial idiom rejection.
  7. Decoupled pure qualification decisions and fail-closed scoring eligibility.
"""
import pytest

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
from app.services.detection.scoring import get_effective_weight
from app.services.detection.semantic import (
    ActorRole,
    EpistemicFrame,
    ModalityType,
    PredicateAssociation,
    TAG_ACTOR_AGENCY,
    TAG_ACTOR_CANDIDATE,
    TAG_ACTOR_EMPLOYER,
    TAG_ACTOR_GOVERNMENT,
    TAG_ACTOR_RECRUITER,
    TAG_ACTOR_SCAMMER,
    TAG_CRED_NOUN,
    TAG_FEE_NOUN,
    TAG_OBLIGATION_MANDATORY,
    TAG_PAY_VERB,
    TAG_REV_NOUN,
    TAG_VERB_DENIAL,
    TAG_VERB_REPORTING,
    QualificationDecision,
    SemanticAssociation,
    SemanticEvidenceAdapter,
    SubClause,
    Token,
    associate_predicate_roles,
    associate_tokens_scoped,
    evaluate_recruitment_context_gating,
    identify_epistemic_frame,
    match_semantic_actions,
    partition_subclauses,
    qualify_semantic_association,
    tokenize_and_tag,
)


# =============================================================================
# 1. THE 10 FOCUSED RECTIFICATION TESTS (T1 TO T10) FROM SECTION 12.3
# =============================================================================

def test_t1_action_scoped_negation_fee_demand():
    """T1: 'Do not pay the fake agent, but candidates must pay the registration fee.'
    Sub-Clause 1 negation is action-scoped to the fake agent. Sub-Clause 2 active fee demand is preserved.
    """
    text = "Do not pay the fake agent, but candidates must pay the registration fee."
    matches = match_semantic_actions(text)

    assert len(matches) == 1
    m = matches[0]
    assert m.pattern_id == "ADV_FEE_REGISTRATION"
    assert m.category == ScamCategory.ADVANCE_FEE
    assert m.severity == FindingSeverity.CRITICAL
    assert m.eligibility == CandidateEligibility.SCORABLE_ACTIVE
    assert m.status == FindingStatus.EXPLICIT_CLAIM

    # Exact raw coordinate bounds: [31, 71)
    assert m.local_start == 31
    assert m.local_end == 71
    assert text[m.local_start:m.local_end] == "candidates must pay the registration fee"

    # Scoring weight
    ev = EvidenceItem(
        id=m.pattern_id,
        category=m.category,
        severity=m.severity,
        status=m.status,
        source=DetectionSource.DETERMINISTIC_RULE,
        matched_text=text[m.local_start:m.local_end],
        start_pos=m.local_start,
        end_pos=m.local_end,
        eligibility=m.eligibility,
    )
    assert get_effective_weight(ev) == 65


def test_t2_competing_object_binding_salaries():
    """T2: 'Candidates discuss registration fees and learn how companies pay salaries.'
    Verb 'pay' is bound to competing object 'salaries', and 'how' isolates subordinate clause.
    Zero candidates emitted.
    """
    text = "Candidates discuss registration fees and learn how companies pay salaries."
    matches = match_semantic_actions(text)
    assert len(matches) == 0


def test_t3_category_preserving_credential_safety_advisory():
    """T3: 'Do not share your UPI PIN with anyone.'
    Negated credential directive is classified as ADVISORY_CREDENTIAL_SAFETY (weight 0).
    """
    text = "Do not share your UPI PIN with anyone."
    matches = match_semantic_actions(text)

    assert len(matches) == 1
    m = matches[0]
    assert m.pattern_id == "ADVISORY_CREDENTIAL_SAFETY"
    assert m.category == ScamCategory.CREDENTIAL_HARVESTING
    assert m.severity == FindingSeverity.INFORMATIONAL
    assert m.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert m.status == FindingStatus.EXPLICIT_CLAIM

    # Exact raw coordinate bounds: [0, 25)
    assert m.local_start == 0
    assert m.local_end == 25
    assert text[m.local_start:m.local_end] == "Do not share your UPI PIN"

    ev = EvidenceItem(
        id=m.pattern_id,
        category=m.category,
        severity=m.severity,
        status=m.status,
        source=DetectionSource.DETERMINISTIC_RULE,
        matched_text=text[m.local_start:m.local_end],
        start_pos=m.local_start,
        end_pos=m.local_end,
        eligibility=m.eligibility,
    )
    assert get_effective_weight(ev) == 0


def test_t4_discretionary_optional_payment_rejection():
    """T4: 'Registration fees may vary by institution.'
    Contains optional modifier 'may vary'; non-mandatory payment reference.
    Zero candidates emitted.
    """
    text = "Registration fees may vary by institution."
    matches = match_semantic_actions(text)
    assert len(matches) == 0


def test_t5_historical_payment_report_rejection():
    """T5: 'Candidates previously paid registration fees.'
    Contains historical marker 'previously paid'; descriptive report.
    Zero candidates emitted.
    """
    text = "Candidates previously paid registration fees."
    matches = match_semantic_actions(text)
    assert len(matches) == 0


def test_t6_non_financial_idiom_rejection():
    """T6: 'Please pay attention in class.'
    Verb 'pay' is bound to non-financial object 'attention'.
    Zero candidates emitted.
    """
    text = "Please pay attention in class."
    matches = match_semantic_actions(text)
    assert len(matches) == 0


def test_t7_active_credential_harvesting_directive():
    """T7: 'Apna UPI PIN HR ko bhejo.'
    Active directive instructing candidate to disclose UPI PIN to HR.
    """
    text = "Apna UPI PIN HR ko bhejo."
    matches = match_semantic_actions(text)

    assert len(matches) == 1
    m = matches[0]
    assert m.pattern_id == "CRED_HARVEST_AUTH_SECRETS"
    assert m.category == ScamCategory.CREDENTIAL_HARVESTING
    assert m.severity == FindingSeverity.CRITICAL
    assert m.eligibility == CandidateEligibility.SCORABLE_ACTIVE
    assert m.status == FindingStatus.EXPLICIT_CLAIM

    # Exact raw coordinate bounds: [0, 24)
    assert m.local_start == 0
    assert m.local_end == 24
    assert text[m.local_start:m.local_end] == "Apna UPI PIN HR ko bhejo"

    ev = EvidenceItem(
        id=m.pattern_id,
        category=m.category,
        severity=m.severity,
        status=m.status,
        source=DetectionSource.DETERMINISTIC_RULE,
        matched_text=text[m.local_start:m.local_end],
        start_pos=m.local_start,
        end_pos=m.local_end,
        eligibility=m.eligibility,
    )
    assert get_effective_weight(ev) == 65


def test_t8_descriptive_framing_rejection():
    """T8: 'UPI PIN is used to authorize transactions.'
    Descriptive technical explanation ('is used to'); non-directive.
    Zero candidates emitted.
    """
    text = "UPI PIN is used to authorize transactions."
    matches = match_semantic_actions(text)
    assert len(matches) == 0


def test_t9_self_initiated_authentication_rejection():
    """T9: 'To authorize your own payment, enter your UPI PIN in your banking app.'
    Legitimate self-initiated banking authentication ('in your banking app').
    Zero candidates emitted.
    """
    text = "To authorize your own payment, enter your UPI PIN in your banking app."
    matches = match_semantic_actions(text)
    assert len(matches) == 0


def test_t10_coordinating_conjunction_shared_obligation():
    """T10: 'Candidates must pay the registration fee and submit their documents.'
    Coordinating conjunction 'and' connects shared predicate without splitting subject obligation.
    Active demand emitted.
    """
    text = "Candidates must pay the registration fee and submit their documents."
    matches = match_semantic_actions(text)

    assert len(matches) == 1
    m = matches[0]
    assert m.pattern_id == "ADV_FEE_REGISTRATION"
    assert m.category == ScamCategory.ADVANCE_FEE
    assert m.severity == FindingSeverity.CRITICAL
    assert m.eligibility == CandidateEligibility.SCORABLE_ACTIVE
    assert m.status == FindingStatus.EXPLICIT_CLAIM

    # Exact raw coordinate bounds: [0, 40)
    assert m.local_start == 0
    assert m.local_end == 40
    assert text[m.local_start:m.local_end] == "Candidates must pay the registration fee"

    ev = EvidenceItem(
        id=m.pattern_id,
        category=m.category,
        severity=m.severity,
        status=m.status,
        source=DetectionSource.DETERMINISTIC_RULE,
        matched_text=text[m.local_start:m.local_end],
        start_pos=m.local_start,
        end_pos=m.local_end,
        eligibility=m.eligibility,
    )
    assert get_effective_weight(ev) == 65


# =============================================================================
# 2. REVERSE UPI DETECTION (ACTIVE DEMAND VS. CATEGORY-PRESERVING SAFETY ADVISORY)
# =============================================================================

def test_reverse_upi_active_demand():
    """Reverse UPI scam: Instructing candidate to approve a collect request to receive salary."""
    text = "Salary credit karne ke liye UPI collect request approve karo."
    matches = match_semantic_actions(text)

    assert len(matches) == 1
    m = matches[0]
    assert m.pattern_id == "ADV_FEE_REVERSE_AUTH"
    assert m.category == ScamCategory.ADVANCE_FEE
    assert m.severity == FindingSeverity.CRITICAL
    assert m.eligibility == CandidateEligibility.SCORABLE_ACTIVE
    assert m.status == FindingStatus.EXPLICIT_CLAIM

    # Exact raw coordinate bounds: [28, 60)
    assert m.local_start == 28
    assert m.local_end == 60
    assert text[m.local_start:m.local_end] == "UPI collect request approve karo"


def test_reverse_upi_advisory_safety():
    """Reverse UPI advisory: Warning candidates against approving collect requests."""
    text = "Never approve any payment request to receive funds."
    # 'Never' negates 'approve'
    tokens = tokenize_and_tag(text)
    assocs = associate_tokens_scoped(tokens)
    assert len(assocs) == 1
    assert assocs[0].is_negated is True

    decision = qualify_semantic_association(assocs[0])
    assert decision.is_qualified is True
    assert decision.pattern_id == "ADVISORY_REVERSE_AUTH_SAFETY"
    assert decision.category == ScamCategory.ADVANCE_FEE
    assert decision.severity == FindingSeverity.INFORMATIONAL
    assert decision.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL


# =============================================================================
# 3. COMPETING ACTIONS & MODALITY TESTS
# =============================================================================

def test_competing_actions_free_laptop_vs_security_deposit():
    """Providing free equipment in clause 1 does not suppress a security deposit demand in clause 2."""
    text = "Laptop is provided free of charge, but candidates must pay a security deposit of Rs. 5000."
    matches = match_semantic_actions(text)

    assert len(matches) == 1
    m = matches[0]
    assert m.pattern_id == "ADV_FEE_REGISTRATION"
    assert m.category == ScamCategory.ADVANCE_FEE
    assert m.severity == FindingSeverity.CRITICAL
    assert m.eligibility == CandidateEligibility.SCORABLE_ACTIVE
    assert "security deposit" in text[m.local_start:m.local_end]


def test_mixed_actor_agency_negation_vs_company_demand():
    """Agency's lack of fees in clause 1 does not suppress employer's mandatory fee in clause 2."""
    text = "The agency does not charge any fee, but candidates must pay a registration fee to our company."
    matches = match_semantic_actions(text)

    # Clause 1 generates an informational advisory, Clause 2 generates an active demand
    active = [m for m in matches if m.eligibility == CandidateEligibility.SCORABLE_ACTIVE]
    advisory = [m for m in matches if m.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL]

    assert len(active) == 1
    assert active[0].pattern_id == "ADV_FEE_REGISTRATION"
    assert active[0].severity == FindingSeverity.CRITICAL

    assert len(advisory) == 1
    assert advisory[0].pattern_id == "ADVISORY_GENERAL_NO_FEE"
    assert advisory[0].severity == FindingSeverity.INFORMATIONAL


def test_general_no_fee_advisory():
    """Category-preserving general advisory in Hindi/Hinglish."""
    text = "Koi bhi registration fee mat do, application bilkul free hai."
    matches = match_semantic_actions(text)

    assert len(matches) == 1
    m = matches[0]
    assert m.pattern_id == "ADVISORY_GENERAL_NO_FEE"
    assert m.category == ScamCategory.ADVANCE_FEE
    assert m.severity == FindingSeverity.INFORMATIONAL
    assert m.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert m.local_start == 0
    assert m.local_end == 31
    assert text[m.local_start:m.local_end] == "Koi bhi registration fee mat do"


# =============================================================================
# 4. DIRECT QUALIFICATION DECISION TESTS (ALL REJECTION REASONS)
# =============================================================================

def test_qualify_decision_self_auth_rejected():
    tok_noun = Token(0, "upi pin", "upi pin", 0, 7, 0, 7, {TAG_CRED_NOUN}, False, False)
    tok_verb = Token(1, "enter", "enter", 8, 13, 8, 13, set(), False, False)
    tok_self = Token(2, "in your banking app", "in your banking app", 14, 33, 14, 33, set(), False, False)

    assoc = SemanticAssociation(
        association_type="CRED_DISCLOSE",
        anchor_noun=tok_noun,
        governing_verb=tok_verb,
        obligation_token=None,
        negation_token=None,
        descriptive_token=None,
        amount_token=None,
        subject_token=None,
        recipient_token=None,
        precond_token=None,
        self_auth_token=tok_self,
        participating_tokens=[tok_noun, tok_verb, tok_self],
        is_negated=False,
        is_descriptive=False,
        is_historical=False,
        is_optional=False,
        has_self_auth=True,
    )
    decision = qualify_semantic_association(assoc)
    assert decision.is_qualified is False
    assert decision.rejection_reason == RejectionReason.SELF_AUTH_NOT_DISCLOSURE
    assert decision.eligibility == CandidateEligibility.REJECTED


def test_qualify_decision_descriptive_rejected():
    tok_noun = Token(0, "upi pin", "upi pin", 0, 7, 0, 7, {TAG_CRED_NOUN}, False, False)
    tok_verb = Token(1, "use", "use", 8, 11, 8, 11, set(), False, False)
    assoc = SemanticAssociation(
        association_type="CRED_DISCLOSE",
        anchor_noun=tok_noun,
        governing_verb=tok_verb,
        obligation_token=None,
        negation_token=None,
        descriptive_token=tok_verb,
        amount_token=None,
        subject_token=None,
        recipient_token=None,
        precond_token=None,
        self_auth_token=None,
        participating_tokens=[tok_noun, tok_verb],
        is_negated=False,
        is_descriptive=True,
        is_historical=False,
        is_optional=False,
        has_self_auth=False,
    )
    decision = qualify_semantic_association(assoc)
    assert decision.is_qualified is False
    assert decision.rejection_reason == RejectionReason.DESCRIPTIVE_NOT_DIRECTIVE


def test_qualify_decision_historical_rejected():
    tok_noun = Token(0, "registration fee", "registration fee", 0, 16, 0, 16, {TAG_FEE_NOUN}, False, False)
    tok_verb = Token(1, "paid", "paid", 17, 21, 17, 21, {TAG_PAY_VERB}, False, False)
    assoc = SemanticAssociation(
        association_type="FEE_DEMAND",
        anchor_noun=tok_noun,
        governing_verb=tok_verb,
        obligation_token=None,
        negation_token=None,
        descriptive_token=None,
        amount_token=None,
        subject_token=None,
        recipient_token=None,
        precond_token=None,
        self_auth_token=None,
        participating_tokens=[tok_noun, tok_verb],
        is_negated=False,
        is_descriptive=False,
        is_historical=True,
        is_optional=False,
        has_self_auth=False,
    )
    decision = qualify_semantic_association(assoc)
    assert decision.is_qualified is False
    assert decision.rejection_reason == RejectionReason.HISTORICAL_REPORT


def test_qualify_decision_optional_rejected():
    tok_noun = Token(0, "fee", "fee", 0, 3, 0, 3, {TAG_FEE_NOUN}, False, False)
    tok_verb = Token(1, "pay", "pay", 4, 7, 4, 7, {TAG_PAY_VERB}, False, False)
    assoc = SemanticAssociation(
        association_type="FEE_DEMAND",
        anchor_noun=tok_noun,
        governing_verb=tok_verb,
        obligation_token=None,
        negation_token=None,
        descriptive_token=None,
        amount_token=None,
        subject_token=None,
        recipient_token=None,
        precond_token=None,
        self_auth_token=None,
        participating_tokens=[tok_noun, tok_verb],
        is_negated=False,
        is_descriptive=False,
        is_historical=False,
        is_optional=True,
        has_self_auth=False,
    )
    decision = qualify_semantic_association(assoc)
    assert decision.is_qualified is False
    assert decision.rejection_reason == RejectionReason.DISCRETIONARY_NOT_MANDATORY


# =============================================================================
# 5. TEXT SEGMENT INTEGRATION & RESOURCE BOUND BUDGETS
# =============================================================================

def test_text_segment_adapter():
    """Verifies that match_semantic_actions operates on a TextSegment with custom source_intervals."""
    raw_content = "PREFIX_OFFSET_PADDING... Candidates must pay the registration fee."
    # Suppose normalized text is just the second sentence
    norm_text = "Candidates must pay the registration fee."
    offset = 25  # raw offset of the sentence
    intervals = [(offset + i, offset + i + 1) for i in range(len(norm_text))]

    segment = TextSegment(
        text=norm_text,
        source_intervals=intervals,
        raw_start=offset,
        raw_end=offset + len(norm_text),
        segment_index=0,
        script="LATIN",
    )

    matches = match_semantic_actions(segment.text, segment.source_intervals)
    assert len(matches) == 1
    m = matches[0]
    assert m.pattern_id == "ADV_FEE_REGISTRATION"
    # Local coordinates match norm_text: [0, 40)
    assert m.local_start == 0
    assert m.local_end == 40
    # Raw coordinates map to original text: [25, 65)
    assert m.raw_start == 25
    assert m.raw_end == 65
    assert raw_content[m.raw_start:m.raw_end] == "Candidates must pay the registration fee"


def test_resource_bound_budget_caps():
    """Verifies that max_total and max_per_pattern bounds are strictly respected."""
    # Input with 5 repeated fee demands
    repeated = "Candidates must pay the registration fee. " * 5
    # Cap per pattern at 2
    counts = {}
    tracker = [0]
    matches = match_semantic_actions(
        repeated,
        pattern_candidate_counts=counts,
        total_candidates_tracker=tracker,
        max_per_pattern=2,
        max_total=10,
    )
    assert len(matches) == 2
    assert counts["ADV_FEE_REGISTRATION"] == 2
    assert tracker[0] == 2


# =============================================================================
# 8. STEP 3.1 ENHANCED LEXER & TOKENIZATION TESTS
# =============================================================================

def test_tokenize_actor_tags():
    """Verify that actor roles (scammers, our company, the agency, recruiter) receive their respective tags."""
    text = "Scammers and recruiters contacted our company and the agency."
    tokens = tokenize_and_tag(text)

    # ActorRole enum exists and has expected values
    assert ActorRole.CANDIDATE == "CANDIDATE"
    assert ActorRole.EMPLOYER == "EMPLOYER"
    assert ActorRole.RECRUITER == "RECRUITER"
    assert ActorRole.AGENCY == "AGENCY"
    assert ActorRole.GOVERNMENT == "GOVERNMENT"
    assert ActorRole.SCAMMER == "SCAMMER"
    assert ActorRole.THIRD_PARTY == "THIRD_PARTY"
    assert ActorRole.UNKNOWN == "UNKNOWN"

    assert any("scammers" in t.text.lower() and TAG_ACTOR_SCAMMER in t.tags for t in tokens)
    assert any("our company" in t.text.lower() and TAG_ACTOR_EMPLOYER in t.tags for t in tokens)
    assert any("the agency" in t.text.lower() and TAG_ACTOR_AGENCY in t.tags for t in tokens)
    assert any("recruiters" in t.text.lower() and TAG_ACTOR_RECRUITER in t.tags for t in tokens)


def test_tokenize_reporting_and_denial_verbs():
    """Verify that reporting verbs (said, claims) and denial verbs (denies, refutes) receive their respective tags."""
    text = "The recruiter said candidates pay fees. The applicant claims so, but the company denies it and refutes allegations."
    tokens = tokenize_and_tag(text)

    assert any("said" in t.text.lower() and TAG_VERB_REPORTING in t.tags for t in tokens)
    assert any("claims" in t.text.lower() and TAG_VERB_REPORTING in t.tags for t in tokens)
    assert any("denies" in t.text.lower() and TAG_VERB_DENIAL in t.tags for t in tokens)
    assert any("refutes" in t.text.lower() and TAG_VERB_DENIAL in t.tags for t in tokens)


def test_token_coordinate_preservation_multiscript():
    """Verify Hindi/Hinglish and English offset accuracy and coordinate preservation via TextSegment."""
    raw_content = "NOTICE: Hamari company kabhi bhi registration fees demand nahi karti. Do not pay scammers!"
    seg_start = 8
    seg_text = raw_content[seg_start:]
    source_intervals = [(seg_start + i, seg_start + i + 1) for i in range(len(seg_text))]
    segment = TextSegment(
        text=seg_text,
        source_intervals=source_intervals,
        raw_start=seg_start,
        raw_end=len(raw_content),
        segment_index=0,
        script="Deva+Latn",
    )
    tokens = tokenize_and_tag(segment)
    assert len(tokens) > 0
    for tok in tokens:
        # Every token's raw coordinates must extract the exact character slice from raw_content
        assert raw_content[tok.raw_start:tok.raw_end] == tok.text


def test_tokenize_candidate_cap():
    """Verify candidate cap stops tokenization at max_tokens."""
    long_text = "word " * 300
    tokens = tokenize_and_tag(long_text, max_tokens=50)
    assert len(tokens) == 50


def test_tokenize_fail_closed_on_error():
    """Verify fail-closed behavior on anomalies returns empty list."""
    # Passing an invalid object that cannot be coerced or causes exception
    class BadText:
        def __str__(self):
            raise RuntimeError("Corrupted unicode stream")

    tokens = tokenize_and_tag(BadText())
    assert tokens == []


# =============================================================================
# 9. STEP 3.2 SUB-CLAUSE PARTITIONING & SCOPING TESTS
# =============================================================================

def test_clause_partition_adversative():
    """Verify clause splitting across adversative conjunction (e.g. ', but the company denies it')."""
    text = "Candidates must pay the fee, but the company denies it."
    tokens = tokenize_and_tag(text)
    subclauses = partition_subclauses(tokens)

    assert len(subclauses) == 2
    # Subclause 0 is the affirmative fee directive
    sc0_words = [t.text.lower().strip() for t in subclauses[0]]
    assert "candidates" in sc0_words
    assert "pay" in sc0_words
    assert subclauses[0].has_adversative is False

    # Subclause 1 is the denial clause introduced by adversative 'but'
    sc1_words = [t.text.lower().strip() for t in subclauses[1]]
    assert "denies" in sc1_words
    assert subclauses[1].has_adversative is True

    # Tokens in separate clauses are never grouped in the same SubClause
    assert not any("denies" in t.text.lower() for t in subclauses[0])
    assert not any("pay" in t.text.lower() for t in subclauses[1])


def test_clause_partition_multi_predicate_coordination():
    """Verify splitting on coordinated predicates e.g. 'submit documents and pay fee'."""
    text = "Our company asks candidates to submit documents and pay a registration fee."
    tokens = tokenize_and_tag(text)
    subclauses = partition_subclauses(tokens)

    assert len(subclauses) == 2
    # Clause 1: submit documents
    sc0_words = [t.text.lower().strip() for t in subclauses[0]]
    assert "submit" in sc0_words
    assert "documents" in sc0_words

    # Clause 2: pay registration fee
    sc1_words = [t.text.lower().strip() for t in subclauses[1]]
    assert "pay" in sc1_words
    assert "registration fee" in sc1_words

    # SubClause coordinates match token boundaries
    assert subclauses[0].start_pos == subclauses[0][0].raw_start
    assert subclauses[0].end_pos == subclauses[0][-1].raw_end
    assert subclauses[1].start_pos == subclauses[1][0].raw_start
    assert subclauses[1].end_pos == subclauses[1][-1].raw_end


def test_clause_partition_no_boundaries():
    """Verify entire token stream constitutes a single sub-clause when no boundary is present."""
    text = "Candidates must submit verified documents"
    tokens = tokenize_and_tag(text)
    subclauses = partition_subclauses(tokens)
    assert len(subclauses) == 1
    assert len(subclauses[0]) == len(tokens)


# =============================================================================
# 10. STEP 3.3 EPISTEMIC FRAMING, ROLE BINDING & CONTEXT GATING TESTS
# =============================================================================

def test_epistemic_framing_scam_warning():
    """Verify that explicit warning phrasing classifies as EpistemicFrame.SCAM_WARNING."""
    text = "Do not pay anyone who promises a guaranteed job."
    tokens = tokenize_and_tag(text)
    sc = partition_subclauses(tokens)[0]
    frame = identify_epistemic_frame(sc)
    assert frame == EpistemicFrame.SCAM_WARNING


def test_epistemic_framing_reported_claim_and_denial():
    """Verify reported claims (Case 9/12) and denials classify distinctly."""
    text_c9 = "The recruiter said that candidates must pay, but the company denies authorizing it."
    tokens_c9 = tokenize_and_tag(text_c9)
    scs_c9 = partition_subclauses(tokens_c9)
    # Splits into: (0) 'The recruiter said' [REPORTED_CLAIM], (1) 'candidates must pay', (2) 'the company denies authorizing it' [DENIAL_REPUDIATION]
    assert len(scs_c9) == 3
    assert identify_epistemic_frame(scs_c9[0]) == EpistemicFrame.REPORTED_CLAIM
    assert identify_epistemic_frame(scs_c9[2]) == EpistemicFrame.DENIAL_REPUDIATION

    text_c12 = "The agency claims that the company collects fees, but the company denies it."
    tokens_c12 = tokenize_and_tag(text_c12)
    scs_c12 = partition_subclauses(tokens_c12)
    assert len(scs_c12) == 3
    assert identify_epistemic_frame(scs_c12[0]) == EpistemicFrame.REPORTED_CLAIM
    assert identify_epistemic_frame(scs_c12[2]) == EpistemicFrame.DENIAL_REPUDIATION


def test_context_gating_accepted_prepositional():
    """Verify prepositional purpose attachment gates to True (Cases 5B / 9B)."""
    text = "Candidates must pay the agency for application processing"
    tokens = tokenize_and_tag(text)
    sc = partition_subclauses(tokens)[0]
    assert evaluate_recruitment_context_gating(sc) is True

    text_infinitive = "The recruiter said that candidates must pay to attend the interview"
    toks_inf = tokenize_and_tag(text_infinitive)
    # The payment directive is in the complement clause after 'that'
    sc_inf = [sc for sc in partition_subclauses(toks_inf) if any(TAG_PAY_VERB in t.tags for t in sc)][0]
    assert evaluate_recruitment_context_gating(sc_inf) is True


def test_context_gating_rejected_sentence_conjunction():
    """Verify sentence-level conjunction without syntactic dependency fails closed to False (Cases 5A / 9A)."""
    text = "Candidates must pay the agency, and the company is hiring engineers"
    tokens = tokenize_and_tag(text)
    scs = partition_subclauses(tokens)
    # The payment directive subclause is isolated from the independent hiring clause
    assert evaluate_recruitment_context_gating(scs[0]) is False

    # Also verify unrelated commercial service is rejected
    text_comm = "Candidates must pay the agency for resume coaching"
    toks_comm = tokenize_and_tag(text_comm)
    sc_comm = partition_subclauses(toks_comm)[0]
    assert evaluate_recruitment_context_gating(sc_comm) is False


def test_coordinated_predicate_inheritance_shared_subject():
    """Verify coordinated predicates inherit instruction_source and action_performer while preserving distinct coordinates."""
    text = "Our company asks candidates to submit documents and pay a registration fee."
    tokens = tokenize_and_tag(text)
    assocs = associate_predicate_roles(tokens)

    assert len(assocs) == 2
    submit_assoc = assocs[0]
    pay_assoc = assocs[1]

    assert submit_assoc.governing_verb.text == "submit"
    assert pay_assoc.governing_verb.text == "pay"

    # Permitted inherited roles (F-01)
    assert pay_assoc.instruction_source == ActorRole.EMPLOYER
    assert pay_assoc.action_performer == ActorRole.CANDIDATE

    # Non-inherited roles must NOT carry over blindly
    assert pay_assoc.payment_recipient == ActorRole.UNKNOWN
    assert pay_assoc.collection_actor == ActorRole.UNKNOWN

    # Offsets are preserved from its own governing verb's predicate span
    assert pay_assoc.raw_start != submit_assoc.raw_start
    assert pay_assoc.raw_start >= 45  # 'and pay...' begins at or after offset 45


def test_coordinated_predicate_inheritance_blocked_by_adversative():
    """Verify adversative conjunctions block role inheritance and reset context."""
    text = "Candidates must submit documents, but the company denies collecting fees."
    tokens = tokenize_and_tag(text)
    assocs = associate_predicate_roles(tokens)

    denial_assoc = [a for a in assocs if a.epistemic_frame == EpistemicFrame.DENIAL_REPUDIATION][0]
    # Inheritance blocked by adversative: instruction_source does not bleed across 'but'
    assert denial_assoc.instruction_source == ActorRole.UNKNOWN
    assert denial_assoc.claim_subject == ActorRole.EMPLOYER


def test_coordinated_predicate_inheritance_fail_closed_ambiguous_subject():
    """Verify ambiguous subject NP fails closed to ActorRole.UNKNOWN."""
    text = "There are positions that require documents and pay fees."
    tokens = tokenize_and_tag(text)
    assocs = associate_predicate_roles(tokens)

    pay_assocs = [a for a in assocs if a.governing_verb.text == "pay"]
    if pay_assocs:
        assert pay_assocs[0].instruction_source == ActorRole.UNKNOWN
        assert pay_assocs[0].action_performer == ActorRole.UNKNOWN


# =============================================================================
# 11. STEP 3.4 CONSOLIDATED ADVERSARIAL ACCEPTANCE MATRIX (20 CASES)
# =============================================================================

def test_acceptance_matrix_case_1():
    """Case 1: 'Candidates must pay a registration fee' -> SCORABLE_ACTIVE (wt=65)."""
    raw = "Candidates must pay a registration fee"
    assoc = PredicateAssociation(
        governing_verb=Token(0, "pay", "pay", 16, 19, 16, 19, {TAG_PAY_VERB}, False, False),
        anchor_noun=Token(1, "registration fee", "registration fee", 22, 38, 22, 38, {TAG_FEE_NOUN}, False, False),
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.CANDIDATE,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.DIRECT_INSTRUCTION,
        modality=ModalityType.MANDATORY,
        participating_tokens=[],
        raw_start=16,
        raw_end=38,
    )
    dec = qualify_semantic_association(assoc)
    assert dec.is_qualified is True
    assert dec.pattern_id == "ADV_FEE_REGISTRATION"
    assert dec.category == ScamCategory.ADVANCE_FEE
    assert dec.severity == FindingSeverity.CRITICAL
    assert dec.status == FindingStatus.EXPLICIT_CLAIM
    assert dec.eligibility == CandidateEligibility.SCORABLE_ACTIVE
    assert dec.rejection_reason is None
    item = SemanticEvidenceAdapter.adapt(dec, raw)
    assert item is not None
    assert get_effective_weight(item) == 65


def test_acceptance_matrix_case_2():
    """Case 2: 'Candidates must not pay a registration fee' -> NON_SCORABLE_INFORMATIONAL (wt=0)."""
    raw = "Candidates must not pay a registration fee"
    assoc = PredicateAssociation(
        governing_verb=Token(0, "pay", "pay", 20, 23, 20, 23, {TAG_PAY_VERB}, False, False),
        anchor_noun=Token(1, "registration fee", "registration fee", 26, 42, 26, 42, {TAG_FEE_NOUN}, False, False),
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.CANDIDATE,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.DIRECT_INSTRUCTION,
        modality=ModalityType.PROHIBITED,
        participating_tokens=[],
        raw_start=20,
        raw_end=42,
        is_negated=True,
    )
    dec = qualify_semantic_association(assoc)
    assert dec.is_qualified is True
    assert dec.pattern_id == "ADVISORY_GENERAL_NO_FEE"
    assert dec.category == ScamCategory.ADVANCE_FEE
    assert dec.severity == FindingSeverity.INFORMATIONAL
    assert dec.status == FindingStatus.EXPLICIT_CLAIM
    assert dec.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert dec.rejection_reason is None
    item = SemanticEvidenceAdapter.adapt(dec, raw)
    assert item is not None
    assert get_effective_weight(item) == 0


def test_acceptance_matrix_case_3_1():
    """Case 3.1: 'Our company does not charge candidates' -> NON_SCORABLE_INFORMATIONAL (wt=0)."""
    raw = "Our company does not charge candidates"
    assoc = PredicateAssociation(
        governing_verb=Token(0, "charge", "charge", 21, 27, 21, 27, {TAG_PAY_VERB}, False, False),
        anchor_noun=None,
        instruction_source=ActorRole.EMPLOYER,
        action_performer=ActorRole.UNKNOWN,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.CANDIDATE,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.EMPLOYER,
        epistemic_frame=EpistemicFrame.DESCRIPTIVE_FACT,
        modality=ModalityType.NEGATED_ACTION,
        participating_tokens=[Token(0, "company", "company", 4, 11, 4, 11, set(), False, False)],
        raw_start=4,
        raw_end=38,
        is_negated=True,
    )
    dec = qualify_semantic_association(assoc)
    assert dec.is_qualified is True
    assert dec.pattern_id == "DISCLAIMER_COMPANY_NO_FEE"
    assert dec.category == ScamCategory.ADVANCE_FEE
    assert dec.severity == FindingSeverity.INFORMATIONAL
    assert dec.status == FindingStatus.EXPLICIT_CLAIM
    assert dec.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert dec.rejection_reason is None
    item = SemanticEvidenceAdapter.adapt(dec, raw)
    assert item is not None
    assert get_effective_weight(item) == 0


def test_acceptance_matrix_case_3_2():
    """Case 3.2: 'the agency collects a fee' -> NON_SCORABLE_INFORMATIONAL (wt=0)."""
    raw = "the agency collects a fee"
    assoc = PredicateAssociation(
        governing_verb=Token(0, "collects", "collects", 11, 19, 11, 19, {TAG_PAY_VERB}, False, False),
        anchor_noun=Token(1, "fee", "fee", 22, 25, 22, 25, {TAG_FEE_NOUN}, False, False),
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.UNKNOWN,
        collection_actor=ActorRole.AGENCY,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.DESCRIPTIVE_FACT,
        modality=ModalityType.DECLARATIVE,
        participating_tokens=[],
        raw_start=0,
        raw_end=25,
    )
    dec = qualify_semantic_association(assoc)
    assert dec.is_qualified is True
    assert dec.pattern_id == "ADVISORY_AGENCY_FEE"
    assert dec.category == ScamCategory.ADVANCE_FEE
    assert dec.severity == FindingSeverity.INFORMATIONAL
    assert dec.status == FindingStatus.AMBIGUOUS_SIGNAL
    assert dec.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert dec.rejection_reason is None
    item = SemanticEvidenceAdapter.adapt(dec, raw)
    assert item is not None
    assert get_effective_weight(item) == 0


def test_acceptance_matrix_case_4():
    """Case 4: 'Scammers ask candidates to pay registration fees' -> NON_SCORABLE_INFORMATIONAL (wt=0)."""
    raw = "Scammers ask candidates to pay registration fees"
    assoc = PredicateAssociation(
        governing_verb=Token(0, "pay", "pay", 27, 30, 27, 30, {TAG_PAY_VERB}, False, False),
        anchor_noun=Token(1, "registration fees", "registration fees", 31, 48, 31, 48, {TAG_FEE_NOUN}, False, False),
        instruction_source=ActorRole.SCAMMER,
        action_performer=ActorRole.CANDIDATE,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.CANDIDATE,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.DESCRIPTIVE_FACT,
        modality=ModalityType.MANDATORY,
        participating_tokens=[],
        raw_start=0,
        raw_end=48,
    )
    dec = qualify_semantic_association(assoc)
    assert dec.is_qualified is True
    assert dec.pattern_id == "ADVISORY_SCAMMER_ATTRIBUTION"
    assert dec.category == ScamCategory.ADVANCE_FEE
    assert dec.severity == FindingSeverity.INFORMATIONAL
    assert dec.status == FindingStatus.EXPLICIT_CLAIM
    assert dec.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert dec.rejection_reason is None
    item = SemanticEvidenceAdapter.adapt(dec, raw)
    assert item is not None
    assert get_effective_weight(item) == 0


def test_acceptance_matrix_case_5a_and_5b():
    """Case 5A & 5B: 'Candidates must pay the agency' (Isolated vs With Context)."""
    raw = "Candidates must pay the agency"
    assoc = PredicateAssociation(
        governing_verb=Token(0, "pay", "pay", 16, 19, 16, 19, {TAG_PAY_VERB}, False, False),
        anchor_noun=None,
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.CANDIDATE,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.AGENCY,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.DIRECT_INSTRUCTION,
        modality=ModalityType.MANDATORY,
        participating_tokens=[],
        raw_start=0,
        raw_end=30,
    )
    # Case 5A: Isolated -> REJECTED
    dec_5a = qualify_semantic_association(assoc, has_recruitment_context=False)
    assert dec_5a.is_qualified is False
    assert dec_5a.eligibility == CandidateEligibility.REJECTED
    assert dec_5a.rejection_reason == RejectionReason.NO_RECRUITMENT_CONTEXT
    assert dec_5a.severity == FindingSeverity.LOW
    assert dec_5a.status == FindingStatus.AMBIGUOUS_SIGNAL
    item_5a = SemanticEvidenceAdapter.adapt(dec_5a, raw)
    assert item_5a is not None
    assert get_effective_weight(item_5a) == 0

    # Case 5B: With Context -> SCORABLE_AMBIGUOUS (wt=10)
    dec_5b = qualify_semantic_association(assoc, has_recruitment_context=True)
    assert dec_5b.is_qualified is True
    assert dec_5b.pattern_id == "ADV_FEE_AGENCY_RECRUITMENT"
    assert dec_5b.category == ScamCategory.ADVANCE_FEE
    assert dec_5b.severity == FindingSeverity.LOW
    assert dec_5b.status == FindingStatus.AMBIGUOUS_SIGNAL
    assert dec_5b.eligibility == CandidateEligibility.SCORABLE_AMBIGUOUS
    assert dec_5b.rejection_reason is None
    item_5b = SemanticEvidenceAdapter.adapt(dec_5b, raw)
    assert item_5b is not None
    assert get_effective_weight(item_5b) == 10


def test_acceptance_matrix_case_6_1_and_6_2():
    """Case 6.1 (submit documents -> REJECTED) & Case 6.2 (pay registration fee -> SCORABLE_ACTIVE)."""
    raw_61 = "Our company asks candidates to submit documents"
    assoc_61 = PredicateAssociation(
        governing_verb=Token(0, "submit", "submit", 32, 38, 32, 38, set(), False, False),
        anchor_noun=None,
        instruction_source=ActorRole.EMPLOYER,
        action_performer=ActorRole.CANDIDATE,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.CANDIDATE,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.DIRECT_INSTRUCTION,
        modality=ModalityType.MANDATORY,
        participating_tokens=[],
        raw_start=0,
        raw_end=47,
        association_type="NON_FINANCIAL_ACTION",
    )
    dec_61 = qualify_semantic_association(assoc_61)
    assert dec_61.is_qualified is False
    assert dec_61.category == ScamCategory.UNKNOWN
    assert dec_61.eligibility == CandidateEligibility.REJECTED
    assert dec_61.rejection_reason == RejectionReason.NON_FINANCIAL_ACTION
    item_61 = SemanticEvidenceAdapter.adapt(dec_61, raw_61)
    assert item_61 is not None
    assert get_effective_weight(item_61) == 0

    # 6.2
    raw_62 = "Our company asks candidates to submit documents and pay a registration fee"
    assoc_62 = PredicateAssociation(
        governing_verb=Token(0, "pay", "pay", 52, 55, 52, 55, {TAG_PAY_VERB}, False, False),
        anchor_noun=Token(1, "registration fee", "registration fee", 58, 74, 58, 74, {TAG_FEE_NOUN}, False, False),
        instruction_source=ActorRole.EMPLOYER,
        action_performer=ActorRole.CANDIDATE,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.DIRECT_INSTRUCTION,
        modality=ModalityType.MANDATORY,
        participating_tokens=[],
        raw_start=52,
        raw_end=74,
    )
    dec_62 = qualify_semantic_association(assoc_62)
    assert dec_62.is_qualified is True
    assert dec_62.pattern_id == "ADV_FEE_REGISTRATION"
    assert dec_62.severity == FindingSeverity.CRITICAL
    assert dec_62.eligibility == CandidateEligibility.SCORABLE_ACTIVE
    item_62 = SemanticEvidenceAdapter.adapt(dec_62, raw_62)
    assert item_62 is not None
    assert get_effective_weight(item_62) == 65


def test_acceptance_matrix_case_7_1_and_7_2():
    """Case 7.1 (register online -> NON_FINANCIAL) & Case 7.2 (payment optional -> DISCRETIONARY)."""
    raw_71 = "Candidates must register online"
    assoc_71 = PredicateAssociation(
        governing_verb=Token(0, "register", "register", 16, 24, 16, 24, set(), False, False),
        anchor_noun=None,
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.CANDIDATE,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.DIRECT_INSTRUCTION,
        modality=ModalityType.MANDATORY,
        participating_tokens=[],
        raw_start=0,
        raw_end=31,
        association_type="NON_FINANCIAL_ACTION",
    )
    dec_71 = qualify_semantic_association(assoc_71)
    assert dec_71.is_qualified is False
    assert dec_71.rejection_reason == RejectionReason.NON_FINANCIAL_ACTION
    item_71 = SemanticEvidenceAdapter.adapt(dec_71, raw_71)
    assert item_71 is not None
    assert get_effective_weight(item_71) == 0

    # 7.2
    raw_72 = "payment is optional"
    assoc_72 = PredicateAssociation(
        governing_verb=Token(0, "pay", "pay", 0, 3, 0, 3, {TAG_PAY_VERB}, False, False),
        anchor_noun=None,
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.UNKNOWN,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.DESCRIPTIVE_FACT,
        modality=ModalityType.OPTIONAL,
        participating_tokens=[],
        raw_start=0,
        raw_end=19,
        is_optional=True,
    )
    dec_72 = qualify_semantic_association(assoc_72)
    assert dec_72.is_qualified is False
    assert dec_72.rejection_reason == RejectionReason.DISCRETIONARY_NOT_MANDATORY
    assert dec_72.severity == FindingSeverity.LOW
    item_72 = SemanticEvidenceAdapter.adapt(dec_72, raw_72)
    assert item_72 is not None
    assert get_effective_weight(item_72) == 0


def test_acceptance_matrix_case_8_1_and_8_2():
    """Case 8.1 (company does not require fee -> DISCLAIMER) & Case 8.2 (gov exam fee -> NON_SCORABLE_INFORMATIONAL)."""
    raw_81 = "The company does not require a fee"
    assoc_81 = PredicateAssociation(
        governing_verb=Token(0, "require", "require", 21, 28, 21, 28, {TAG_PAY_VERB}, False, False),
        anchor_noun=Token(1, "fee", "fee", 31, 34, 31, 34, {TAG_FEE_NOUN}, False, False),
        instruction_source=ActorRole.EMPLOYER,
        action_performer=ActorRole.UNKNOWN,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.EMPLOYER,
        epistemic_frame=EpistemicFrame.DESCRIPTIVE_FACT,
        modality=ModalityType.NOT_REQUIRED,
        participating_tokens=[Token(0, "company", "company", 4, 11, 4, 11, set(), False, False)],
        raw_start=0,
        raw_end=34,
        is_negated=True,
    )
    dec_81 = qualify_semantic_association(assoc_81)
    assert dec_81.is_qualified is True
    assert dec_81.pattern_id == "DISCLAIMER_COMPANY_NO_FEE"
    assert dec_81.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    item_81 = SemanticEvidenceAdapter.adapt(dec_81, raw_81)
    assert item_81 is not None
    assert get_effective_weight(item_81) == 0

    # 8.2 (Resolved in v1.6.0 Correction A as NON_SCORABLE_INFORMATIONAL wt=0)
    raw_82 = "candidates may voluntarily pay the government examination fee"
    assoc_82 = PredicateAssociation(
        governing_verb=Token(0, "pay", "pay", 27, 30, 27, 30, {TAG_PAY_VERB}, False, False),
        anchor_noun=Token(1, "examination fee", "examination fee", 46, 61, 46, 61, {TAG_FEE_NOUN}, False, False),
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.CANDIDATE,
        collection_actor=ActorRole.GOVERNMENT,
        payment_recipient=ActorRole.GOVERNMENT,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.DESCRIPTIVE_FACT,
        modality=ModalityType.OPTIONAL,
        participating_tokens=[
            Token(0, "government", "government", 35, 45, 35, 45, {TAG_ACTOR_GOVERNMENT}, False, False),
            Token(1, "examination", "examination", 46, 57, 46, 57, set(), False, False),
        ],
        raw_start=0,
        raw_end=61,
        is_optional=True,
    )
    dec_82 = qualify_semantic_association(assoc_82)
    assert dec_82.is_qualified is True
    assert dec_82.pattern_id == "ADVISORY_GOVERNMENT_EXAM_FEE"
    assert dec_82.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert dec_82.severity == FindingSeverity.INFORMATIONAL
    assert dec_82.status == FindingStatus.AMBIGUOUS_SIGNAL
    assert dec_82.rejection_reason is None
    item_82 = SemanticEvidenceAdapter.adapt(dec_82, raw_82)
    assert item_82 is not None
    assert get_effective_weight(item_82) == 0


def test_acceptance_matrix_case_9a_9b_and_9_2():
    """Case 9A (hearsay isolated), 9B (hearsay with context), 9.2 (denial)."""
    raw = "The recruiter said that candidates must pay to attend the interview"
    assoc_9 = PredicateAssociation(
        governing_verb=Token(0, "pay", "pay", 39, 42, 39, 42, {TAG_PAY_VERB}, False, False),
        anchor_noun=None,
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.CANDIDATE,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.RECRUITER,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.REPORTED_CLAIM,
        modality=ModalityType.MANDATORY,
        participating_tokens=[],
        raw_start=0,
        raw_end=42,
    )
    # 9A: Isolated -> REJECTED
    dec_9a = qualify_semantic_association(assoc_9, has_recruitment_context=False)
    assert dec_9a.is_qualified is False
    assert dec_9a.eligibility == CandidateEligibility.REJECTED
    assert dec_9a.rejection_reason == RejectionReason.NO_RECRUITMENT_CONTEXT
    assert dec_9a.severity == FindingSeverity.LOW
    item_9a = SemanticEvidenceAdapter.adapt(dec_9a, raw)
    assert item_9a is not None
    assert get_effective_weight(item_9a) == 0

    # 9B: Demonstrably connected -> SCORABLE_AMBIGUOUS (wt=10)
    dec_9b = qualify_semantic_association(assoc_9, has_recruitment_context=True)
    assert dec_9b.is_qualified is True
    assert dec_9b.pattern_id == "ADV_FEE_REPORTED_RECRUITMENT"
    assert dec_9b.eligibility == CandidateEligibility.SCORABLE_AMBIGUOUS
    assert dec_9b.severity == FindingSeverity.LOW
    assert dec_9b.status == FindingStatus.AMBIGUOUS_SIGNAL
    item_9b = SemanticEvidenceAdapter.adapt(dec_9b, raw)
    assert item_9b is not None
    assert get_effective_weight(item_9b) == 10

    # 9.2: Denial -> NON_SCORABLE_INFORMATIONAL
    raw_92 = "the company denies authorizing it"
    assoc_92 = PredicateAssociation(
        governing_verb=Token(0, "denies", "denies", 12, 18, 12, 18, {TAG_VERB_DENIAL}, False, False),
        anchor_noun=None,
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.UNKNOWN,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.EMPLOYER,
        epistemic_frame=EpistemicFrame.DENIAL_REPUDIATION,
        modality=ModalityType.NEGATED_ACTION,
        participating_tokens=[],
        raw_start=0,
        raw_end=33,
        is_negated=True,
    )
    dec_92 = qualify_semantic_association(assoc_92)
    assert dec_92.is_qualified is True
    assert dec_92.pattern_id == "DENIAL_COMPANY_REPUDIATION"
    assert dec_92.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert dec_92.rejection_reason is None
    item_92 = SemanticEvidenceAdapter.adapt(dec_92, raw_92)
    assert item_92 is not None
    assert get_effective_weight(item_92) == 0


def test_acceptance_matrix_case_10():
    """Case 10: 'Do not pay anyone who promises a guaranteed job' -> SCAM_WARNING."""
    raw = "Do not pay anyone who promises a guaranteed job"
    assoc = PredicateAssociation(
        governing_verb=Token(0, "pay", "pay", 7, 10, 7, 10, {TAG_PAY_VERB}, False, False),
        anchor_noun=None,
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.CANDIDATE,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.SCAM_WARNING,
        modality=ModalityType.PROHIBITED,
        participating_tokens=[],
        raw_start=0,
        raw_end=47,
        is_negated=True,
    )
    dec = qualify_semantic_association(assoc)
    assert dec.is_qualified is True
    assert dec.pattern_id == "ADVISORY_SCAM_WARNING"
    assert dec.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert dec.severity == FindingSeverity.INFORMATIONAL
    assert dec.rejection_reason is None
    item = SemanticEvidenceAdapter.adapt(dec, raw)
    assert item is not None
    assert get_effective_weight(item) == 0


def test_acceptance_matrix_case_11():
    """Case 11: 'Candidates are not required to pay unless they request optional equipment' -> REJECTED."""
    raw = "Candidates are not required to pay unless they request optional equipment"
    assoc = PredicateAssociation(
        governing_verb=Token(0, "pay", "pay", 31, 34, 31, 34, {TAG_PAY_VERB}, False, False),
        anchor_noun=None,
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.CANDIDATE,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.UNKNOWN,
        epistemic_frame=EpistemicFrame.DESCRIPTIVE_FACT,
        modality=ModalityType.CONDITIONAL,
        participating_tokens=[Token(0, "unless", "unless", 35, 41, 35, 41, set(), False, False)],
        raw_start=0,
        raw_end=73,
    )
    dec = qualify_semantic_association(assoc)
    assert dec.is_qualified is False
    assert dec.eligibility == CandidateEligibility.REJECTED
    assert dec.rejection_reason == RejectionReason.DISCRETIONARY_NOT_MANDATORY
    assert dec.severity == FindingSeverity.LOW
    item = SemanticEvidenceAdapter.adapt(dec, raw)
    assert item is not None
    assert get_effective_weight(item) == 0


def test_acceptance_matrix_case_12_1_and_12_2():
    """Case 12.1 (allegation -> NON_SCORABLE_INFORMATIONAL) & Case 12.2 (denial)."""
    # 12.1
    raw_121 = "The agency claims that the company collects fees"
    assoc_121 = PredicateAssociation(
        governing_verb=Token(0, "collects", "collects", 35, 43, 35, 43, {TAG_PAY_VERB}, False, False),
        anchor_noun=Token(1, "fees", "fees", 44, 48, 44, 48, {TAG_FEE_NOUN}, False, False),
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.UNKNOWN,
        collection_actor=ActorRole.EMPLOYER,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.AGENCY,
        claim_subject=ActorRole.EMPLOYER,
        epistemic_frame=EpistemicFrame.REPORTED_CLAIM,
        modality=ModalityType.DECLARATIVE,
        participating_tokens=[],
        raw_start=0,
        raw_end=48,
    )
    dec_121 = qualify_semantic_association(assoc_121)
    assert dec_121.is_qualified is True
    assert dec_121.pattern_id == "ADVISORY_ALLEGED_EMPLOYER_FEE"
    assert dec_121.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    assert dec_121.severity == FindingSeverity.INFORMATIONAL
    assert dec_121.status == FindingStatus.AMBIGUOUS_SIGNAL
    item_121 = SemanticEvidenceAdapter.adapt(dec_121, raw_121)
    assert item_121 is not None
    assert get_effective_weight(item_121) == 0

    # 12.2
    raw_122 = "the company denies it"
    assoc_122 = PredicateAssociation(
        governing_verb=Token(0, "denies", "denies", 12, 18, 12, 18, {TAG_VERB_DENIAL}, False, False),
        anchor_noun=None,
        instruction_source=ActorRole.UNKNOWN,
        action_performer=ActorRole.UNKNOWN,
        collection_actor=ActorRole.UNKNOWN,
        payment_recipient=ActorRole.UNKNOWN,
        affected_patient=ActorRole.UNKNOWN,
        information_recipient=ActorRole.UNKNOWN,
        reporting_source=ActorRole.UNKNOWN,
        claim_subject=ActorRole.EMPLOYER,
        epistemic_frame=EpistemicFrame.DENIAL_REPUDIATION,
        modality=ModalityType.NEGATED_ACTION,
        participating_tokens=[],
        raw_start=0,
        raw_end=21,
        is_negated=True,
    )
    dec_122 = qualify_semantic_association(assoc_122)
    assert dec_122.is_qualified is True
    assert dec_122.pattern_id == "DENIAL_COMPANY_REPUDIATION"
    assert dec_122.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
    item_122 = SemanticEvidenceAdapter.adapt(dec_122, raw_122)
    assert item_122 is not None
    assert get_effective_weight(item_122) == 0


# =============================================================================
# 12. STEP 3.5 HARDENED SEMANTIC EVIDENCE ADAPTER TESTS
# =============================================================================

def test_adapter_provenance_completeness():
    """Verify that adapted EvidenceItem populates all 4 required provenance fields in model_fields_set."""
    raw = "Candidates must pay the registration fee."
    dec = QualificationDecision(
        is_qualified=True,
        pattern_id="ADV_FEE_REGISTRATION",
        category=ScamCategory.ADVANCE_FEE,
        severity=FindingSeverity.CRITICAL,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
        status=FindingStatus.EXPLICIT_CLAIM,
        explanation="Mandatory fee directive",
        semantic_target="target:fee",
        action_type="demand:pay",
        rejection_reason=None,
        raw_start=0,
        raw_end=41,
    )

    item = SemanticEvidenceAdapter.adapt(dec, raw)
    assert item is not None
    # 4 provenance fields strictly present in model_fields_set
    assert {"category", "source", "severity", "status"}.issubset(item.model_fields_set)
    assert item.source == DetectionSource.DETERMINISTIC_RULE
    assert item.category == ScamCategory.ADVANCE_FEE
    assert item.severity == FindingSeverity.CRITICAL
    assert item.status == FindingStatus.EXPLICIT_CLAIM
    assert item.eligibility == CandidateEligibility.SCORABLE_ACTIVE


def test_adapter_deterministic_finding_id():
    """Verify deterministic, collision-free instance IDs based on pattern_id and raw offsets."""
    raw = "Candidates must pay the fee. Applicants must pay the deposit."
    dec1 = QualificationDecision(
        is_qualified=True,
        pattern_id="ADV_FEE_REGISTRATION",
        category=ScamCategory.ADVANCE_FEE,
        severity=FindingSeverity.CRITICAL,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
        status=FindingStatus.EXPLICIT_CLAIM,
        explanation="Fee",
        raw_start=0,
        raw_end=27,
    )
    dec2 = QualificationDecision(
        is_qualified=True,
        pattern_id="ADV_FEE_REGISTRATION",
        category=ScamCategory.ADVANCE_FEE,
        severity=FindingSeverity.CRITICAL,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
        status=FindingStatus.EXPLICIT_CLAIM,
        explanation="Fee",
        raw_start=29,
        raw_end=60,
    )

    item1 = SemanticEvidenceAdapter.adapt(dec1, raw)
    item2 = SemanticEvidenceAdapter.adapt(dec2, raw)
    assert item1 is not None and item2 is not None
    assert item1.finding_id == "ADV_FEE_REGISTRATION_0_27"
    assert item2.finding_id == "ADV_FEE_REGISTRATION_29_60"
    assert item1.finding_id != item2.finding_id


def test_adapter_fail_closed_on_invalid_bounds():
    """Verify that out-of-range character bounds fail closed safely returning None."""
    raw = "Short text."
    dec_bad = QualificationDecision(
        is_qualified=True,
        pattern_id="ADV_FEE_REGISTRATION",
        category=ScamCategory.ADVANCE_FEE,
        severity=FindingSeverity.CRITICAL,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
        status=FindingStatus.EXPLICIT_CLAIM,
        explanation="Fee",
        raw_start=0,
        raw_end=999,  # out of bounds
    )
    assert SemanticEvidenceAdapter.adapt(dec_bad, raw) is None
    assert SemanticEvidenceAdapter.adapt(dec_bad, None) is None  # type: ignore


def test_adapter_caller_object_immutability():
    """Verify that adapting a QualificationDecision does not mutate the decision object."""
    raw = "Candidates must pay the fee."
    dec = QualificationDecision(
        is_qualified=True,
        pattern_id="ADV_FEE_REGISTRATION",
        category=ScamCategory.ADVANCE_FEE,
        severity=FindingSeverity.CRITICAL,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
        status=FindingStatus.EXPLICIT_CLAIM,
        explanation="Fee",
        raw_start=0,
        raw_end=27,
    )
    # NamedTuple is immutable; verifying adapt produces an independent EvidenceItem
    item = SemanticEvidenceAdapter.adapt(dec, raw)
    assert item is not None
    assert dec.eligibility == CandidateEligibility.SCORABLE_ACTIVE
    assert dec.raw_start == 0


def test_adapter_batch():
    """Verify that adapt_batch processes lists of decisions, filtering failed adaptations."""
    raw = "Candidates must pay the fee. Applicants must pay the deposit."
    dec1 = QualificationDecision(
        is_qualified=True,
        pattern_id="ADV_FEE_REGISTRATION",
        category=ScamCategory.ADVANCE_FEE,
        severity=FindingSeverity.CRITICAL,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
        status=FindingStatus.EXPLICIT_CLAIM,
        explanation="Fee 1",
        raw_start=0,
        raw_end=27,
    )
    dec_invalid = QualificationDecision(
        is_qualified=True,
        pattern_id="ADV_FEE_INVALID",
        category=ScamCategory.ADVANCE_FEE,
        severity=FindingSeverity.CRITICAL,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
        status=FindingStatus.EXPLICIT_CLAIM,
        explanation="Invalid",
        raw_start=500,
        raw_end=999,
    )
    items = SemanticEvidenceAdapter.adapt_batch([dec1, dec_invalid], raw)
    assert len(items) == 1
    assert items[0].finding_id == "ADV_FEE_REGISTRATION_0_27"
