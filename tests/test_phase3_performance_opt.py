"""
Phase 3 Priority 1 Performance Optimization Test Suite (OPT-01 & OPT-02).
Verifies:
  1. OPT-01: Per-Request DocumentContext & 1,000-case automated differential verification.
  2. OPT-02: Bounded per-request clause-bounds memoization, hard 500 capacity, invalid bounds bypass.
  3. OPT-02: Stage 3 to Stage 5 cache hit reuse with empirical coordinate evidence.
  4. Isolation: Independent per-request cache lifecycles without global state leakage.
"""
import copy
import random
import re
import pytest

from app.services.detection import (
    CandidateEligibility,
    DetectionResult,
    DetectionSource,
    DocumentContext,
    EvidenceItem,
    FindingSeverity,
    FindingStatus,
    ScamCategory,
    analyze_job_posting,
    extract_action_context,
    reconcile_contradictions,
    validate_evidence_quotation,
)
from app.services.detection.quotation import (
    _extract_clause_bounds,
    mask_abbreviations,
)


# =============================================================================
# 1. OPT-01: 1,000-CASE AUTOMATED DIFFERENTIAL VERIFICATION
# =============================================================================

def _generate_differential_corpus() -> list[tuple[str, int, int]]:
    """
    Generates at least 1,000 diverse, strictly unique test cases covering:
      - Decimal values (10.50, Rs. 499.99, v1.2.3)
      - Abbreviations (Rs., Pvt. Ltd., Inc., Corp., Dr., Mr., etc.)
      - Hindi Unicode tokens (शुल्क, पंजीकरण, आदि)
      - CRLF and LF line endings
      - Multiple consecutive terminators (!!, ??, ;;\n\n)
      - Long paragraphs (500 to 3,000+ characters)
      - Adversarial boundary positions (start=0, end=len, right at period, etc.)
    Guarantees that every returned (raw_content, start_pos, end_pos) tuple is 100% unique.
    """
    seen: set[tuple[str, int, int]] = set()
    cases: list[tuple[str, int, int]] = []
    rng = random.Random(1337)  # Deterministic seed

    # Group 1: Decimal numbers and monetary values (250 unique cases)
    for i in range(250):
        val = f"{rng.randint(1, 9999)}.{rng.randint(10, 99)}"
        val2 = f"{rng.randint(1, 999)}.{rng.randint(1, 9)}"
        text = (
            f"Job #{i:04d}: Fee is Rs. {val} for registration. "
            f"Version {val2} updates require deposit of ${val}. Candidates must remit."
        )
        s = rng.randint(0, len(text) - 15)
        e = s + rng.randint(5, 20)
        c = (text, s, e)
        if c not in seen:
            seen.add(c)
            cases.append(c)

    # Group 2: Abbreviations and company names (250 unique cases)
    abbrevs = [
        "Rs.", "INR.", "Inc.", "Corp.", "Ltd.", "Co.", "Pvt. Ltd.", "No.",
        "Dr.", "Mr.", "Mrs.", "Ms.", "approx.", "e.g.", "i.e.", "vs."
    ]
    for i in range(250):
        abbr = abbrevs[i % len(abbrevs)]
        text = (
            f"File ID {i:04d}: Contact Dr. Sharma at {abbr} regarding placement. "
            f"All applicants to {abbr} must pay fee of Rs. {i * 10 + 50}."
        )
        s = rng.randint(0, len(text) - 12)
        e = s + rng.randint(4, 18)
        c = (text, s, e)
        if c not in seen:
            seen.add(c)
            cases.append(c)

    # Group 3: Hindi Unicode tokens & mixed scripts (250 unique cases)
    hindi_templates = [
        "उम्मीदवारों को पंजीकरण शुल्क Rs. {fee} जमा करना होगा। कृपया तुरंत संपर्क करें। संदर्भ #{i}.",
        "कंपनी कोई भर्ती शुल्क नहीं लेती है। Scammers are demanding registration charges of Rs. {fee} (केस #{i}).",
        "नौकरी के लिए आवेदन संख्या {i} पूरी तरह से निःशुल्क है। All candidates must pay an onboarding fee of INR {fee}.",
        "चेतावनी संदर्भ {i}: फर्जी भर्ती करने वालों से सावधान रहें। Do not pay any fee of ₹{fee} to external agencies.",
        "Apex Services Pvt. Ltd. संदर्भ {i}: सुरक्षा जमा राशि ₹{fee} अनिवार्य है।",
    ]
    for i in range(250):
        tmpl = hindi_templates[i % len(hindi_templates)]
        text = tmpl.format(i=i, fee=i * 15 + 100)
        s = rng.randint(0, len(text) - 8)
        e = s + rng.randint(3, 15)
        c = (text, s, e)
        if c not in seen:
            seen.add(c)
            cases.append(c)

    # Group 4: CRLF/LF line endings and multiple consecutive terminators (250 unique cases)
    terms = ["\r\n", "\n", "!!\r\n", "???\n", "....\n", ";;\n\n", "!\n\r\n"]
    for i in range(250):
        t1 = terms[i % len(terms)]
        t2 = terms[(i + 2) % len(terms)]
        text = (
            f"Batch {i:04d}{t1}"
            f"Notice: Candidates must register online{t2}"
            f"Our company charges verification fee of Rs. {i * 5 + 200}!!{t1}"
            f"Failure to pay will disqualify;;"
        )
        s = rng.randint(0, len(text) - 10)
        e = s + rng.randint(4, 16)
        c = (text, s, e)
        if c not in seen:
            seen.add(c)
            cases.append(c)

    # Group 5: Long paragraphs & adversarial boundaries (250 unique cases)
    base_long = (
        "This is an extensive job posting description for Enterprise-{i}. We are hiring engineers. "
        "Candidates must pay a documentation fee of Rs. {i}. No agency is authorized to collect fees. "
    )
    for i in range(250):
        repeats = (i % 5) + 3
        text = base_long.format(i=i) * repeats
        if i % 5 == 0:
            # Adversarial: start index 0
            s, e = 0, min(15, len(text))
        elif i % 5 == 1:
            # Adversarial: end index len(text)
            e = len(text)
            s = max(0, e - 20)
        elif i % 5 == 2:
            # Adversarial: immediately after period
            p = text.find(".")
            s = p + 1
            e = s + 12
        elif i % 5 == 3:
            # Adversarial: spanning abbreviation
            p = text.find("Rs.")
            s = p
            e = p + 8
        else:
            s = rng.randint(0, len(text) - 30)
            e = s + rng.randint(5, 25)
        c = (text, s, e)
        if c not in seen:
            seen.add(c)
            cases.append(c)

    return cases


def test_opt01_1000_case_automated_differential_verification():
    """
    Mandatory Requirement:
      Execute at least 1,000 automated differential test cases comparing cached (with doc_ctx)
      and legacy (doc_ctx=None) execution.
      Compare:
        - sent_start
        - next_stop
        - pre_clause
        - post_clause
      All 4 outputs MUST match in 100% of cases.
      Asserts that every case tuple in the corpus is strictly unique.
    """
    corpus = _generate_differential_corpus()
    assert len(corpus) >= 1000, f"Corpus size {len(corpus)} must be >= 1000"
    unique_count = len(set(corpus))
    assert unique_count == len(corpus), f"Corpus contains duplicate cases: {len(corpus) - unique_count} duplicates found"
    assert unique_count >= 1000, f"Unique cases {unique_count} must be >= 1000"

    mismatches = []
    for idx, (raw_content, start_pos, end_pos) in enumerate(corpus):
        # 1. Uncached / Legacy execution (doc_ctx=None)
        res_legacy = extract_action_context(
            raw_content, start_pos, end_pos, doc_ctx=None, return_sentence_bounds=True
        )
        sent_start_leg, next_stop_leg, pre_clause_leg, post_clause_leg = res_legacy

        # 2. Cached execution (with DocumentContext)
        ctx = DocumentContext(raw_content)
        res_cached = extract_action_context(
            raw_content, start_pos, end_pos, doc_ctx=ctx, return_sentence_bounds=True
        )
        sent_start_c, next_stop_c, pre_clause_c, post_clause_c = res_cached

        # Verify exact equivalence across all 4 parameters
        if (
            sent_start_leg != sent_start_c
            or next_stop_leg != next_stop_c
            or pre_clause_leg != pre_clause_c
            or post_clause_leg != post_clause_c
        ):
            mismatches.append(
                f"Case #{idx} failed: "
                f"legacy=({sent_start_leg}, {next_stop_leg}, '{pre_clause_leg}', '{post_clause_leg}') vs "
                f"cached=({sent_start_c}, {next_stop_c}, '{pre_clause_c}', '{post_clause_c}')"
            )

    assert not mismatches, f"Differential verification failed with {len(mismatches)} mismatches:\n" + "\n".join(mismatches[:5])


# =============================================================================
# 2. OPT-02: BOUNDED PER-REQUEST CLAUSE-BOUNDS MEMOIZATION TESTS
# =============================================================================

def test_opt02_cache_lifecycle_and_hits():
    """
    Verifies that:
      - DocumentContext initializes an empty _clause_bounds_cache.
      - First call is a cache miss and populates cache.
      - Second call with identical coordinates is a cache hit.
      - Exact coordinate keys (start_pos, end_pos) map to (pre_clause, post_clause).
    """
    text = "Candidates must register with our company and pay a registration fee of Rs. 500."
    quote = "pay a registration fee"
    s = text.find(quote)
    e = s + len(quote)

    ctx = DocumentContext(text)
    assert len(ctx._clause_bounds_cache) == 0
    assert ctx.cache_hits == 0
    assert ctx.cache_misses == 0

    # First call: cache miss
    pre1, post1 = extract_action_context(text, s, e, doc_ctx=ctx)
    assert ctx.cache_misses == 1
    assert ctx.cache_hits == 0
    assert (s, e) in ctx._clause_bounds_cache
    assert ctx._clause_bounds_cache[(s, e)] == (pre1, post1)

    # Second call: cache hit
    pre2, post2 = extract_action_context(text, s, e, doc_ctx=ctx)
    assert ctx.cache_misses == 1
    assert ctx.cache_hits == 1
    assert (pre1, post1) == (pre2, post2)


def test_opt02_hard_capacity_ceiling_and_uncached_fallback():
    """
    Verifies that:
      - Cache accepts entries up to MAX_CACHE_ENTRIES (500).
      - When cache reaches 500 entries, subsequent entries bypass insertion (uncached fallback).
      - Subsequent calls still compute and return correct results without raising exceptions.
    """
    text = "A " * 1000 + "Candidates must pay a registration fee of Rs. 500."
    ctx = DocumentContext(text)

    # Pre-populate cache with 500 dummy coordinates
    for i in range(500):
        ctx._clause_bounds_cache[(i * 2, i * 2 + 1)] = (f"pre_{i}", f"post_{i}")
    assert len(ctx._clause_bounds_cache) == 500

    quote = "pay a registration fee"
    s = text.find(quote)
    e = s + len(quote)

    # 501st entry lookup
    pre, post = extract_action_context(text, s, e, doc_ctx=ctx)
    # Cache size must remain bounded at 500
    assert len(ctx._clause_bounds_cache) == 500
    assert (s, e) not in ctx._clause_bounds_cache
    # Output must still be correct
    assert "pay a registration fee" not in pre
    assert ctx.cache_misses == 1


def test_opt02_invalid_and_malformed_coordinates_handling():
    """
    Verifies defensive coordinate validation:
      - Invalid coordinates bypass cache lookup and insertion.
      - Return safe fallback ("", "") without raising exceptions.
    """
    text = "Candidates must pay a registration fee of Rs. 500."
    ctx = DocumentContext(text)

    invalid_cases = [
        (-1, 10),           # Negative start_pos
        (10, 5),            # start_pos >= end_pos
        (10, 10),           # Empty span
        (10, 99999),        # end_pos > len(raw_content)
        (None, 10),         # None start_pos
        (10, None),         # None end_pos
        ("0", 10),          # String type
        (True, 10),         # Boolean type
    ]

    for s, e in invalid_cases:
        res = extract_action_context(text, s, e, doc_ctx=ctx)
        assert res == ("", ""), f"Failed for coords ({s}, {e})"
        assert len(ctx._clause_bounds_cache) == 0, f"Cache contaminated by invalid coords ({s}, {e})"
        assert ctx.cache_hits == 0


def test_opt02_independent_request_isolation():
    """
    Verifies zero cross-request state leakage:
      - Two DocumentContext instances have completely independent caches and counters.
      - No shared global state.
    """
    text1 = "Candidates must pay a registration fee of Rs. 500."
    text2 = "Applicants must remit training charges of Rs. 1000."

    ctx1 = DocumentContext(text1)
    ctx2 = DocumentContext(text2)

    s1, e1 = text1.find("pay"), text1.find("pay") + 3
    s2, e2 = text2.find("remit"), text2.find("remit") + 5

    extract_action_context(text1, s1, e1, doc_ctx=ctx1)
    assert len(ctx1._clause_bounds_cache) == 1
    assert len(ctx2._clause_bounds_cache) == 0

    extract_action_context(text2, s2, e2, doc_ctx=ctx2)
    assert len(ctx1._clause_bounds_cache) == 1
    assert len(ctx2._clause_bounds_cache) == 1
    assert (s1, e1) not in ctx2._clause_bounds_cache
    assert (s2, e2) not in ctx1._clause_bounds_cache


# =============================================================================
# 3. STAGE 3 TO STAGE 5 CACHE REUSE EMPIRICAL EVIDENCE
# =============================================================================

def test_opt02_stage3_to_stage5_cache_reuse_empirical_verification():
    """
    Empirically verifies that:
      - Coordinate spans evaluated during Stage 3 (validate_evidence_quotation)
      - Are accessed and reused in Stage 5 (reconcile_contradictions)
      - Producing cache hits on the exact coordinate spans without recomputing regexes.
    """
    text = (
        "We are hiring software engineers. "
        "Our company requires a registration fee of Rs. 1000 before onboarding. "
        "We never charge any registration fees."
    )
    doc_ctx = DocumentContext(text)

    # 1. Finding arriving at Stage 3
    quote = "requires a registration fee"
    s = text.find(quote)
    e = s + len(quote)
    finding = EvidenceItem(
        id="ADV_FEE_REGISTRATION",
        category=ScamCategory.ADVANCE_FEE,
        source=DetectionSource.DETERMINISTIC_RULE,
        severity=FindingSeverity.CRITICAL,
        status=FindingStatus.EXPLICIT_CLAIM,
        matched_text=quote,
        explanation="Upfront fee demand",
        finding_id="ADV_FEE_REGISTRATION_1",
        start_pos=s,
        end_pos=e,
        eligibility=CandidateEligibility.SCORABLE_ACTIVE,
    )

    # Run Stage 3 quotation validation
    validated_item = validate_evidence_quotation(finding, text, doc_ctx=doc_ctx)
    assert (s, e) in doc_ctx._clause_bounds_cache
    hits_after_stage3 = doc_ctx.cache_hits
    misses_after_stage3 = doc_ctx.cache_misses
    assert misses_after_stage3 == 1
    assert hits_after_stage3 == 0

    # 2. Run Stage 5 contradiction resolution with the same doc_ctx
    # Stage 5 evaluates contradiction relationships between ADV_FEE_REGISTRATION and company disclaimer
    result_items = reconcile_contradictions([validated_item], text, doc_ctx=doc_ctx)

    # Empirical assertion: Stage 5 must have produced a cache hit on (s, e)
    hits_after_stage5 = doc_ctx.cache_hits
    misses_after_stage5 = doc_ctx.cache_misses
    assert hits_after_stage5 == 1, (
        f"Stage 5 failed to reuse Stage 3 cached span! Hits before={hits_after_stage3}, hits after={hits_after_stage5}"
    )
    assert misses_after_stage5 == 1, "Stage 5 must not incur additional cache misses on evaluated span"

    # Verify that the finding was correctly linked to the disclaimer
    adv_fee = next(it for it in result_items if it.id == "ADV_FEE_REGISTRATION")
    assert len(adv_fee.contradicts_finding_ids) > 0
    assert "DISCLAIMER_COMPANY_NO_FEE_1" in adv_fee.contradicts_finding_ids
