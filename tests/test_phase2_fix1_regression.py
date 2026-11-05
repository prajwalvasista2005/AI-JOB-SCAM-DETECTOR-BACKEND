import copy
import re
import pytest

from app.services.detection.models import (
    CandidateEligibility,
    DetectionSource,
    EvidenceItem,
    FindingSeverity,
    FindingStatus,
    RejectionReason,
    ScamCategory,
)
from app.services.detection.patterns import (
    COMPANY_FEE_DISCLAIMER_RE,
    DETECTION_PATTERNS,
    DISCLAIMER_IMMUNE_IDS,
)
from app.services.detection.quotation import (
    CLAUSE_BOUNDARY_RE,
    CLAUSE_WARNING_FRAMING_RE,
    DISCLAIMER_AGENCY_REPUDIATION_RE,
    TIER1_EMPLOYER_ACTION_RE,
    TIER2_AGENCY_ACTION_RE,
    extract_action_context,
    mask_abbreviations,
    reconcile_contradictions,
    validate_evidence_quotation,
)
from app.services.detection.scoring import get_effective_weight


class TestPhase2Fix1Regression:
    """
    Phase 2: Fix 1 Preservation & Regression Test Suite.
    Verifies all Fix 1 semantic invariants, action-linked attribution (Tiers 1, 2, 5),
    fail-safe repeated quotation resolution, clause boundaries, warning scopes A-E,
    and immunity rules.
    """

    # =========================================================================
    # 1. Action-Linked Charging-Entity Attribution (Tier 1 vs Tier 2 vs Tier 5)
    # =========================================================================

    def test_mandatory_action_linked_attribution_vectors(self):
        """
        Tests the 5 mandatory attribution vectors required by Phase 2:
          1. "We are hiring software engineers; candidates must pay a registration fee of Rs. 500." -> Tier 5 (unlinked)
          2. "Our company is hiring immediately. Candidates must pay an application processing fee." -> Tier 5 (unlinked)
          3. "Pay Rs. 500 directly to our company." -> Tier 1 (linked to company disclaimer)
          4. "Our company collects a mandatory registration fee." -> Tier 1 (linked to company disclaimer)
          5. "Pay ABC Placement Agency Rs. 1,000." -> Tier 2 (unlinked with generic disclaimer)
        """
        adv_pat = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        company_disclaimer = "Our company does not charge candidates any registration or recruitment fees."

        # Vector 1: Employer in hiring clause, but fee demand is unattributed -> Tier 5
        doc1 = f"{company_disclaimer}\n\nWe are hiring software engineers; candidates must pay a registration fee of Rs. 500."
        m1 = adv_pat.regex.search(doc1)
        assert m1 is not None, "Vector 1 must match ADV_FEE_REGISTRATION"
        item1 = adv_pat.to_evidence_item(
            matched_text=m1.group(0),
            finding_id="ADV_1",
            start_pos=m1.start(),
            end_pos=m1.end(),
        )
        res1 = reconcile_contradictions([item1], doc1)
        fee1 = next(it for it in res1 if it.id == "ADV_FEE_REGISTRATION")
        assert fee1.contradicts_finding_ids == [], "Vector 1 must be Tier 5 (unlinked)"
        assert "Unverified charging entity" in (fee1.context_required or ""), "Vector 1 must have Tier 5 warning"

        # Vector 2: Employer in separate sentence, fee demand unattributed -> Tier 5
        doc2 = f"{company_disclaimer}\n\nOur company is hiring immediately. Candidates must pay an application processing fee."
        m2 = adv_pat.regex.search(doc2)
        assert m2 is not None, "Vector 2 must match ADV_FEE_REGISTRATION"
        item2 = adv_pat.to_evidence_item(
            matched_text=m2.group(0),
            finding_id="ADV_2",
            start_pos=m2.start(),
            end_pos=m2.end(),
        )
        res2 = reconcile_contradictions([item2], doc2)
        fee2 = next(it for it in res2 if it.id == "ADV_FEE_REGISTRATION")
        assert fee2.contradicts_finding_ids == [], "Vector 2 must be Tier 5 (unlinked)"
        assert "Unverified charging entity" in (fee2.context_required or ""), "Vector 2 must have Tier 5 warning"

        # Vector 3: Pay directly to our company -> Tier 1 (linked)
        doc3 = f"{company_disclaimer}\n\nPay Rs. 500 directly to our company."
        m3 = adv_pat.regex.search(doc3)
        assert m3 is not None, "Vector 3 must match ADV_FEE_REGISTRATION"
        item3 = adv_pat.to_evidence_item(
            matched_text=m3.group(0),
            finding_id="ADV_3",
            start_pos=m3.start(),
            end_pos=m3.end(),
        )
        res3 = reconcile_contradictions([item3], doc3)
        fee3 = next(it for it in res3 if it.id == "ADV_FEE_REGISTRATION")
        disc3 = next(it for it in res3 if it.id == "DISCLAIMER_COMPANY_NO_FEE")
        assert disc3.finding_id in fee3.contradicts_finding_ids, "Vector 3 must link to disclaimer (Tier 1)"
        assert fee3.finding_id in disc3.contradicts_finding_ids, "Disclaimer must link back to Vector 3"

        # Vector 4: Our company collects a mandatory registration fee -> Tier 1 (linked)
        doc4 = f"{company_disclaimer}\n\nOur company collects a mandatory registration fee."
        m4 = adv_pat.regex.search(doc4)
        assert m4 is not None, "Vector 4 must match ADV_FEE_REGISTRATION"
        item4 = adv_pat.to_evidence_item(
            matched_text=m4.group(0),
            finding_id="ADV_4",
            start_pos=m4.start(),
            end_pos=m4.end(),
        )
        res4 = reconcile_contradictions([item4], doc4)
        fee4 = next(it for it in res4 if it.id == "ADV_FEE_REGISTRATION")
        disc4 = next(it for it in res4 if it.id == "DISCLAIMER_COMPANY_NO_FEE")
        assert disc4.finding_id in fee4.contradicts_finding_ids, "Vector 4 must link to disclaimer (Tier 1)"
        assert fee4.finding_id in disc4.contradicts_finding_ids, "Disclaimer must link back to Vector 4"

        # Vector 5: Pay ABC Placement Agency Rs. 1,000 -> Tier 2 (unlinked with generic disclaimer)
        doc5 = f"{company_disclaimer}\n\nPay ABC Placement Agency Rs. 1,000."
        m5 = adv_pat.regex.search(doc5)
        assert m5 is not None, "Vector 5 must match ADV_FEE_REGISTRATION"
        item5 = adv_pat.to_evidence_item(
            matched_text=m5.group(0),
            finding_id="ADV_5",
            start_pos=m5.start(),
            end_pos=m5.end(),
        )
        res5 = reconcile_contradictions([item5], doc5)
        fee5 = next(it for it in res5 if it.id == "ADV_FEE_REGISTRATION")
        assert fee5.contradicts_finding_ids == [], "Vector 5 must NOT link to generic disclaimer"
        assert "Independent agency fee" in (fee5.context_required or ""), "Vector 5 must warn of independent agency"

    # =========================================================================
    # 2. Tier 2 Agency Repudiation: Scenario A vs Scenario B
    # =========================================================================

    def test_tier2_scenario_a_agency_repudiated(self):
        """
        Tier 2 Scenario A: Employer disclaimer explicitly repudiates third-party agencies.
        Agency fee demand MUST be linked to disclaimer.
        """
        adv_pat = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        repudiation_doc = (
            "Our company does not charge fees. No agency is authorized to collect recruitment fees.\n\n"
            "Pay ABC Placement Agency Rs. 1,000."
        )
        m = adv_pat.regex.search(repudiation_doc)
        assert m is not None
        fee_item = adv_pat.to_evidence_item(
            matched_text=m.group(0),
            finding_id="ADV_AGENCY",
            start_pos=m.start(),
            end_pos=m.end(),
        )
        res = reconcile_contradictions([fee_item], repudiation_doc)
        fee_res = next(it for it in res if it.id == "ADV_FEE_REGISTRATION")
        disc_res = next(
            it for it in res
            if it.id == "DISCLAIMER_COMPANY_NO_FEE" and "agency" in it.matched_text.lower()
        )
        assert disc_res.finding_id in fee_res.contradicts_finding_ids, "Repudiated agency fee must link to disclaimer"
        assert fee_res.finding_id in disc_res.contradicts_finding_ids, "Disclaimer must link back to agency fee"

    def test_tier2_scenario_b_generic_company_disclaimer(self):
        """
        Tier 2 Scenario B: Generic employer disclaimer does NOT repudiate agencies.
        Agency fee demand MUST NOT be linked, and must contain independent agency warning.
        """
        adv_pat = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        generic_doc = (
            "Our company does not charge candidates any fees.\n\n"
            "Pay ABC Placement Agency Rs. 1,000."
        )
        m = adv_pat.regex.search(generic_doc)
        assert m is not None
        fee_item = adv_pat.to_evidence_item(
            matched_text=m.group(0),
            finding_id="ADV_AGENCY",
            start_pos=m.start(),
            end_pos=m.end(),
        )
        res = reconcile_contradictions([fee_item], generic_doc)
        fee_res = next(it for it in res if it.id == "ADV_FEE_REGISTRATION")
        assert fee_res.contradicts_finding_ids == [], "Generic disclaimer must not link to independent agency fee"
        assert "Independent agency fee" in (fee_res.context_required or "")

    # =========================================================================
    # 3. Immunity Rules (Tier 3, Tier 4, Payment Rails, Credentials, Checks)
    # =========================================================================

    def test_tier3_government_examination_fee_immunity(self):
        """
        Tier 3: Public examination or licensing fee (ADV_FEE_AMBIGUOUS_MENTION) is immune
        from company disclaimer links and maintains AMBIGUOUS_SIGNAL status.
        """
        doc = (
            "Our company never charges fees.\n\n"
            "Official examination fees of Rs. 500 must be paid to the state testing authority."
        )
        item = EvidenceItem(
            id="ADV_FEE_AMBIGUOUS_MENTION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.LOW,
            status=FindingStatus.AMBIGUOUS_SIGNAL,
            matched_text="examination fees of Rs. 500",
            start_pos=doc.find("examination fees of Rs. 500"),
            end_pos=doc.find("examination fees of Rs. 500") + len("examination fees of Rs. 500"),
        )
        res = reconcile_contradictions([item], doc)
        t3_item = next(it for it in res if it.id == "ADV_FEE_AMBIGUOUS_MENTION")
        assert t3_item.contradicts_finding_ids == [], "Tier 3 exam fee must not link to company disclaimer"
        assert get_effective_weight(t3_item) == 10, "Tier 3 exam fee contributes low weight"

    def test_tier4_security_deposit_immunity(self):
        """
        Tier 4: Equipment / laptop security deposit (ADV_FEE_SECURITY_DEPOSIT) is in
        DISCLAIMER_IMMUNE_IDS and cannot be excused by recruitment disclaimer.
        """
        doc = (
            "Our company does not charge recruitment fees.\n\n"
            "A refundable security deposit of Rs. 5000 is required for company laptop."
        )
        item = EvidenceItem(
            id="ADV_FEE_SECURITY_DEPOSIT",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="security deposit of Rs. 5000 is required",
            start_pos=doc.find("security deposit of Rs. 5000 is required"),
            end_pos=doc.find("security deposit of Rs. 5000 is required") + len("security deposit of Rs. 5000 is required"),
        )
        assert "ADV_FEE_SECURITY_DEPOSIT" in DISCLAIMER_IMMUNE_IDS
        res = reconcile_contradictions([item], doc)
        sec_item = next(it for it in res if it.id == "ADV_FEE_SECURITY_DEPOSIT")
        assert sec_item.contradicts_finding_ids == [], "Tier 4 security deposit must be immune from disclaimer links"
        assert sec_item.severity == FindingSeverity.CRITICAL
        assert get_effective_weight(sec_item) == 65

    def test_payment_rail_credential_and_check_immunity(self):
        """
        Verifies that direct UPI rails, auth credentials, and check overpayment schemes
        are strictly immune from disclaimer linking.
        """
        doc = "Our company does not charge candidates any fees.\n\nInstructions follow."
        immune_items = [
            EvidenceItem(
                id="ADV_FEE_PAYMENT_RAIL_DIRECT",
                category=ScamCategory.ADVANCE_FEE,
                source=DetectionSource.DETERMINISTIC_RULE,
                severity=FindingSeverity.CRITICAL,
                status=FindingStatus.EXPLICIT_CLAIM,
                matched_text="pay Rs. 500 to recruiter@oksbi",
                start_pos=10,
                end_pos=40,
            ),
            EvidenceItem(
                id="CRED_HARVEST_AUTH_SECRETS",
                category=ScamCategory.CREDENTIAL_HARVESTING,
                source=DetectionSource.DETERMINISTIC_RULE,
                severity=FindingSeverity.CRITICAL,
                status=FindingStatus.EXPLICIT_CLAIM,
                matched_text="share your bank OTP for verification",
                start_pos=50,
                end_pos=87,
            ),
            EvidenceItem(
                id="EQUIP_OVERPAY_CHECK_VENDOR",
                category=ScamCategory.EQUIPMENT_OVERPAYMENT,
                source=DetectionSource.DETERMINISTIC_RULE,
                severity=FindingSeverity.CRITICAL,
                status=FindingStatus.EXPLICIT_CLAIM,
                matched_text="cashier check will be sent to buy equipment from approved vendor",
                start_pos=90,
                end_pos=155,
            ),
        ]
        res = reconcile_contradictions(immune_items, doc)
        for it in res:
            if it.id in DISCLAIMER_IMMUNE_IDS:
                assert it.contradicts_finding_ids == [], f"{it.id} must never link to disclaimer"

    # =========================================================================
    # 4. Warning Scope Boundaries (Cases A through E)
    # =========================================================================

    def test_warning_scope_case_a_separate_paragraphs(self):
        """
        Case A: Warning in paragraph 1 does not suppress active demand in paragraph 2.
        """
        text = (
            "Caution: Fraud alert. Beware of scammers asking for payments.\n\n"
            "Candidates must pay a registration fee of Rs. 500."
        )
        adv_pat = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        m = adv_pat.regex.search(text)
        assert m is not None
        item = adv_pat.to_evidence_item(
            matched_text=m.group(0),
            start_pos=m.start(),
            end_pos=m.end(),
        )
        validated = validate_evidence_quotation(item, text)
        assert validated.status == FindingStatus.EXPLICIT_CLAIM, "Para 2 demand must not be suppressed by Para 1 warning"
        assert validated.severity == FindingSeverity.CRITICAL

    def test_warning_scope_case_b_adversative_conjunction(self):
        """
        Case B: Adversative conjunction (', but', ', yet', ', whereas') terminates warning scope.
        "We do not charge candidates, but our agency collects a fee"
        """
        text = "We do not charge candidates, but our agency collects a fee"
        adv_pat = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        m = adv_pat.regex.search(text)
        assert m is not None
        item = adv_pat.to_evidence_item(
            matched_text=m.group(0),
            start_pos=m.start(),
            end_pos=m.end(),
        )
        validated = validate_evidence_quotation(item, text)
        assert validated.status == FindingStatus.EXPLICIT_CLAIM, "Adversative conjunction must protect active fee from preceding disclaimer"
        assert validated.severity == FindingSeverity.CRITICAL

    def test_warning_scope_case_c_coordinating_conjunction_independent_obligation(self):
        """
        Case C: Coordinating conjunction introducing an independent obligation with new subject/modal.
        "Beware of scammers, and candidates must pay the official application fee of Rs. 500"
        """
        text = "Beware of scammers, and candidates must pay the official application fee of Rs. 500"
        adv_pat = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        m = adv_pat.regex.search(text)
        assert m is not None
        item = adv_pat.to_evidence_item(
            matched_text=m.group(0),
            start_pos=m.start(),
            end_pos=m.end(),
        )
        validated = validate_evidence_quotation(item, text)
        assert validated.status == FindingStatus.EXPLICIT_CLAIM, "Independent obligation clause must not be suppressed by warning"
        assert validated.severity == FindingSeverity.CRITICAL

    def test_warning_scope_case_d_transition_adverb(self):
        """
        Case D: Transition adverbs ('. However,', '; however,', '. In reality,') terminate disclaimer scope.
        "We do not charge fees. However, candidates must pay a registration fee of Rs. 500."
        """
        text = "We do not charge fees. However, candidates must pay a registration fee of Rs. 500."
        adv_pat = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        m = adv_pat.regex.search(text)
        assert m is not None
        item = adv_pat.to_evidence_item(
            matched_text=m.group(0),
            start_pos=m.start(),
            end_pos=m.end(),
        )
        validated = validate_evidence_quotation(item, text)
        assert validated.status == FindingStatus.EXPLICIT_CLAIM
        assert validated.severity == FindingSeverity.CRITICAL

    def test_warning_scope_case_e_ambiguous_warning_fails_safely(self):
        """
        Case E: Ambiguous context fails safely, preserving active risk rather than falsely suppressing.
        """
        text = "Notice: Some third parties charge candidates. Candidates must pay an application fee of Rs. 500."
        adv_pat = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        m = adv_pat.regex.search(text)
        assert m is not None
        item = adv_pat.to_evidence_item(
            matched_text=m.group(0),
            start_pos=m.start(),
            end_pos=m.end(),
        )
        validated = validate_evidence_quotation(item, text)
        assert validated.status == FindingStatus.EXPLICIT_CLAIM
        assert validated.severity == FindingSeverity.CRITICAL

    # =========================================================================
    # 5. Clause Boundaries & Coordinated Action Semantics
    # =========================================================================

    def test_clause_boundary_vector1_shared_modal_preserved(self):
        """
        "Candidates must submit documents and pay a registration fee"
        'and' coordinates two verb phrases sharing subject and modal 'Candidates must'.
        The payment demand must NOT be severed or suppressed.
        """
        text = "Candidates must submit documents and pay a registration fee of Rs. 500"
        adv_pat = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        m = adv_pat.regex.search(text)
        assert m is not None
        item = adv_pat.to_evidence_item(
            matched_text=m.group(0),
            start_pos=m.start(),
            end_pos=m.end(),
        )
        validated = validate_evidence_quotation(item, text)
        assert validated.status == FindingStatus.EXPLICIT_CLAIM
        assert validated.severity == FindingSeverity.CRITICAL

    def test_clause_boundary_vector2_warning_isolated_from_independent_clause(self):
        """
        "Beware of scammers, and candidates must pay the official application fee"
        Warning is isolated; active demand is preserved.
        """
        text = "Beware of scammers, and candidates must pay the official application fee of Rs. 500"
        adv_pat = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        m = adv_pat.regex.search(text)
        assert m is not None
        item = adv_pat.to_evidence_item(
            matched_text=m.group(0),
            start_pos=m.start(),
            end_pos=m.end(),
        )
        validated = validate_evidence_quotation(item, text)
        assert validated.status == FindingStatus.EXPLICIT_CLAIM

    def test_clause_boundary_vector3_adversative_conjunction_actor_separation(self):
        """
        "We do not charge candidates, but our agency collects a fee"
        Preceding negation does NOT negate agency fee, nor misattribute agency fee to employer.
        """
        text = "We do not charge candidates, but our agency collects a fee"
        adv_pat = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        m = adv_pat.regex.search(text)
        assert m is not None
        item = adv_pat.to_evidence_item(
            matched_text=m.group(0),
            start_pos=m.start(),
            end_pos=m.end(),
        )
        validated = validate_evidence_quotation(item, text)
        assert validated.status == FindingStatus.EXPLICIT_CLAIM

        # Contradiction reconciliation: generic company disclaimer does NOT link to agency fee
        res = reconcile_contradictions([validated], text)
        fee_item = next(it for it in res if it.id == "ADV_FEE_REGISTRATION")
        assert fee_item.contradicts_finding_ids == [], "Agency fee must not link to employer disclaimer"
        assert "Independent agency fee" in (fee_item.context_required or "")

    def test_clause_boundary_vector4_scammer_action_reported_speech_suppression(self):
        """
        "Scammers ask for money, and candidates should report them"
        The scammer action is governed by warning/reporting framing and does not become active candidate demand.
        """
        text = "Scammers ask for money, and candidates should report them"
        match = CLAUSE_WARNING_FRAMING_RE.search(text)
        assert match is not None, "Scammer action must match warning framing"

    # =========================================================================
    # 6. Repeated Identical Quotations & Fail-Safe Occurrence Resolution
    # =========================================================================

    def test_repeated_quotations_offset_present_resolves_accurately(self):
        """
        When offsets are present, identical quotations across paragraphs are accurately resolved
        to their respective paragraph contexts.
        """
        raw_doc = (
            "Paragraph 1: In 2021, we asked for a registration fee of Rs. 500 for records.\n\n"
            "Paragraph 2: Our company does not charge candidates any registration or recruitment fees.\n\n"
            "Paragraph 3: Beware of fraud: scammers say a registration fee of Rs. 500 is needed.\n\n"
            "Paragraph 4: Pay Rs. 500 directly to our company."
        )
        quote = "Pay Rs. 500 directly to our company"
        s4 = raw_doc.find(quote)
        e4 = s4 + len(quote)
        item = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text=quote,
            start_pos=s4,
            end_pos=e4,
        )
        res = reconcile_contradictions([item], raw_doc)
        fee = next(it for it in res if it.id == "ADV_FEE_REGISTRATION")
        disc = next(it for it in res if it.id == "DISCLAIMER_COMPANY_NO_FEE")
        assert fee.start_pos == s4
        assert fee.end_pos == e4
        assert disc.finding_id in fee.contradicts_finding_ids, "Accurately offset item in Para 4 must link to Para 2 disclaimer"

    def test_repeated_quotations_missing_offsets_fail_safely_unlinked(self):
        """
        When identical quotations occur multiple times across paragraphs and offsets are MISSING,
        the system must NOT guess list-order or positions; it leaves offsets as None and fails safely
        without creating false contradiction links.
        """
        raw_doc = (
            "Paragraph 1: Beware of scammers claiming registration fee is required.\n\n"
            "Paragraph 2: Our company does not charge candidates any fees.\n\n"
            "Paragraph 3: Notice: some agencies require a registration fee.\n\n"
            "Paragraph 4: Candidates must pay a registration fee."
        )
        # "registration fee" appears 3+ times in raw_doc
        ambiguous_item = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="registration fee",
            start_pos=None,
            end_pos=None,
        )
        res = reconcile_contradictions([ambiguous_item], raw_doc)
        fee = next(it for it in res if it.id == "ADV_FEE_REGISTRATION")
        assert fee.start_pos is None, "Ambiguous repeated item must remain unresolved (start_pos=None)"
        assert fee.end_pos is None, "Ambiguous repeated item must remain unresolved (end_pos=None)"
        assert fee.contradicts_finding_ids == [], "Ambiguous repeated item must NOT have guessed contradiction links"
        assert "Unverified charging entity" in (fee.context_required or ""), "Must have Tier 5 warning"

    # =========================================================================
    # 7. Bidirectional Symmetry, Zero Dangling References, Caller Immutability
    # =========================================================================

    def test_bidirectional_symmetry_and_zero_dangling_references(self):
        """
        Verifies that all contradiction edges are strictly symmetric (A <-> B)
        with zero one-sided edges and zero dangling references.
        """
        raw_doc = (
            "Our company does not charge candidates any recruitment fees.\n\n"
            "Pay Rs. 500 directly to our company.\n\n"
            "Our company collects a mandatory registration fee."
        )
        adv_pat = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        items = []
        for m in adv_pat.regex.finditer(raw_doc):
            items.append(
                adv_pat.to_evidence_item(
                    matched_text=m.group(0),
                    start_pos=m.start(),
                    end_pos=m.end(),
                )
            )

        res = reconcile_contradictions(items, raw_doc)
        item_map = {it.finding_id: it for it in res}
        all_ids = set(item_map.keys())

        for it in res:
            for target_id in it.contradicts_finding_ids:
                # Zero dangling references
                assert target_id in all_ids, f"Dangling reference detected: {target_id} not in result IDs"
                # Strict bidirectional symmetry
                target = item_map[target_id]
                assert it.finding_id in target.contradicts_finding_ids, (
                    f"One-sided edge detected: {it.finding_id} -> {target_id}, but {target_id} does not link back!"
                )

    def test_caller_immutability_throughout_phase2_pipeline(self):
        """
        Verifies that caller-owned EvidenceItem objects are never mutated across
        validate_evidence_quotation and reconcile_contradictions.
        """
        raw_doc = "Our company does not charge fees. Pay Rs. 500 directly to our company."
        original_item = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="Pay Rs. 500 directly to our company",
            start_pos=34,
            end_pos=69,
            contradicts_finding_ids=[],
        )
        caller_copy = copy.deepcopy(original_item)

        # 1. validate_evidence_quotation
        v_out = validate_evidence_quotation(original_item, raw_doc)
        assert original_item == caller_copy, "Caller item mutated by validate_evidence_quotation!"

        # 2. reconcile_contradictions
        r_out = reconcile_contradictions([original_item], raw_doc)
        assert original_item == caller_copy, "Caller item mutated by reconcile_contradictions!"

    # =========================================================================
    # 8. Phase 2 Final Approved Design Verification & Regression Suite
    # =========================================================================

    def test_fail_closed_eligibility_and_invalid_provenance(self):
        """
        Covers:
          - Default eligibility is REJECTED
          - Missing eligibility with incomplete provenance remains REJECTED
          - Explicit None is coerced fail-closed to REJECTED
          - Explicit invalid string is coerced fail-closed to REJECTED
          - Explicit SCORABLE_ACTIVE with missing provenance is demoted to REJECTED
          - Explicit SCORABLE_ACTIVE with UNKNOWN category or UNKNOWN source is REJECTED
        """
        # Default with incomplete provenance -> REJECTED
        item_inc = EvidenceItem(
            id="ADV_1",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            # source missing!
        )
        assert item_inc.eligibility == CandidateEligibility.REJECTED
        assert item_inc.rejection_reason == RejectionReason.UNVERIFIED_PROVENANCE
        assert get_effective_weight(item_inc) == 0

        # Explicit None -> REJECTED
        item_none = EvidenceItem(
            id="ADV_2",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            eligibility=None,
        )
        assert item_none.eligibility == CandidateEligibility.REJECTED
        assert get_effective_weight(item_none) == 0

        # Explicit invalid string -> REJECTED
        item_inv = EvidenceItem(
            id="ADV_3",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            eligibility="unrecognized_status",
        )
        assert item_inv.eligibility == CandidateEligibility.REJECTED
        assert get_effective_weight(item_inv) == 0

        # Explicit SCORABLE_ACTIVE with missing provenance (anti-bypass)
        item_bypass = EvidenceItem(
            id="ADV_4",
            eligibility=CandidateEligibility.SCORABLE_ACTIVE,
        )
        assert item_bypass.eligibility == CandidateEligibility.REJECTED
        assert item_bypass.rejection_reason == RejectionReason.UNVERIFIED_PROVENANCE
        assert get_effective_weight(item_bypass) == 0

        # Explicit SCORABLE_ACTIVE with UNKNOWN category -> REJECTED
        item_unk_cat = EvidenceItem(
            id="ADV_5",
            category=ScamCategory.UNKNOWN,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            eligibility=CandidateEligibility.SCORABLE_ACTIVE,
        )
        assert item_unk_cat.eligibility == CandidateEligibility.REJECTED
        assert get_effective_weight(item_unk_cat) == 0

        # Explicit SCORABLE_ACTIVE with UNKNOWN source -> REJECTED
        item_unk_src = EvidenceItem(
            id="ADV_6",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.UNKNOWN,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            eligibility=CandidateEligibility.SCORABLE_ACTIVE,
        )
        assert item_unk_src.eligibility == CandidateEligibility.REJECTED
        assert get_effective_weight(item_unk_src) == 0

    def test_explicit_versus_inferred_eligibility(self):
        """
        Covers:
          - Missing eligibility with complete provenance qualifies as SCORABLE_ACTIVE
          - Explicit SCORABLE_ACTIVE with complete provenance remains SCORABLE_ACTIVE
          - Explicit REJECTED is preserved fail-closed even with complete provenance
        """
        # Inferred from complete provenance
        item_inf = EvidenceItem(
            id="ADV_INF",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
        )
        assert item_inf.eligibility == CandidateEligibility.SCORABLE_ACTIVE
        assert get_effective_weight(item_inf) == 65

        # Explicit SCORABLE_ACTIVE
        item_act = EvidenceItem(
            id="ADV_ACT",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            eligibility=CandidateEligibility.SCORABLE_ACTIVE,
        )
        assert item_act.eligibility == CandidateEligibility.SCORABLE_ACTIVE
        assert get_effective_weight(item_act) == 65

        # Explicit REJECTED preserved
        item_rej = EvidenceItem(
            id="ADV_REJ",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            eligibility=CandidateEligibility.REJECTED,
        )
        assert item_rej.eligibility == CandidateEligibility.REJECTED
        assert get_effective_weight(item_rej) == 0

    def test_low_severity_tier3_qualification_and_active_rejection(self):
        """
        Covers:
          - LOW severity + AMBIGUOUS_SIGNAL + trusted source -> SCORABLE_AMBIGUOUS (weight 10)
          - LOW severity + EXPLICIT_CLAIM -> REJECTED (never promoted to SCORABLE_ACTIVE)
          - Explicit SCORABLE_AMBIGUOUS on explicit critical claim is REJECTED (contradictory)
        """
        # Valid Tier 3 exam fee
        item_t3 = EvidenceItem(
            id="ADV_FEE_AMBIGUOUS_MENTION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.LOW,
            status=FindingStatus.AMBIGUOUS_SIGNAL,
        )
        assert item_t3.eligibility == CandidateEligibility.SCORABLE_AMBIGUOUS
        assert get_effective_weight(item_t3) == 10

        # LOW severity with EXPLICIT_CLAIM must NOT become SCORABLE_ACTIVE
        item_low_claim = EvidenceItem(
            id="ADV_FEE_LOW_CLAIM",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.LOW,
            status=FindingStatus.EXPLICIT_CLAIM,
        )
        assert item_low_claim.eligibility == CandidateEligibility.REJECTED
        assert get_effective_weight(item_low_claim) == 0

        # Contradictory provenance: explicit claim labeled as SCORABLE_AMBIGUOUS
        item_contra = EvidenceItem(
            id="ADV_FEE_CONTRA",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            eligibility=CandidateEligibility.SCORABLE_AMBIGUOUS,
        )
        assert item_contra.eligibility == CandidateEligibility.REJECTED
        assert item_contra.rejection_reason == RejectionReason.CONTRADICTORY_PROVENANCE

    def test_tier1_employer_payment_instructions_positive_and_negative(self):
        """
        Covers:
          - Positive: explicit employer payment directives and instructions
          - Negative: unrelated company mentions (portal, website, founding date, bare possessives)
        """
        positives = [
            "our company asks them to pay a registration fee",
            "the company requires candidates to pay a mandatory onboarding fee",
            "we instruct you to deposit a documentation fee of Rs. 500",
            "our firm directs applicants to pay a placement charge",
            "pay a registration fee directly to our company",
            "our company collects a non-refundable application fee",
            "pay us a processing fee",
        ]
        for phrase in positives:
            assert TIER1_EMPLOYER_ACTION_RE.search(phrase), f"Tier 1 must match: '{phrase}'"

        negatives = [
            "Candidates register on our company portal and pay a registration fee of Rs. 500",
            "Visit our company website and pay a registration fee of Rs. 500",
            "Our company was founded in 2010. Pay a registration fee of Rs. 500",
            "our company's registration fees",
            "pay a registration fee of Rs. 500",
        ]
        for phrase in negatives:
            assert not TIER1_EMPLOYER_ACTION_RE.search(phrase), f"Tier 1 must NOT match: '{phrase}'"

    def test_tier2_agency_ditransitive_and_prepositional_attribution(self):
        """
        Covers:
          - Ditransitive agency demands: "Pay Apex Placement Agency a fee of Rs. 1000"
          - Prepositional agency demands: "Pay Rs. 500 to XYZ Consultancy"
          - Active agency collection: "Apex Placement Agency collects a registration fee"
          - Negative exclusions: bare possessives, company fees, unspecified fees
        """
        positives = [
            "Pay Apex Placement Agency a fee of Rs. 1000",
            "Pay Apex Placement Agency Rs. 1000",
            "Candidates must pay ABC Consultancy a registration fee",
            "Pay directly to Apex Placement Agency",
            "Pay Rs. 500 to XYZ Consultancy",
            "pay to ABC Staffing firm Rs. 500",
            "Apex Placement Agency collects a registration fee",
            "The consultancy charges an upfront fee",
        ]
        for phrase in positives:
            assert TIER2_AGENCY_ACTION_RE.search(phrase), f"Tier 2 must match: '{phrase}'"

        negatives = [
            "pay our company for uniforms",
            "our company's registration fees",
            "agency's registration fees",
            "pay a registration fee of Rs. 500",
        ]
        for phrase in negatives:
            assert not TIER2_AGENCY_ACTION_RE.search(phrase), f"Tier 2 must NOT match: '{phrase}'"

    def test_tier2_agency_repudiation_vs_generic_disclaimer(self):
        """
        Covers:
          - Disclaimer with explicit agency repudiation links to agency fee
          - Generic employer disclaimer does NOT link to agency fee (attaches advisory note)
        """
        # Tier 2A: Explicit agency repudiation
        doc_repudiated = (
            "Our company does not charge fees and we do not authorize any agency to collect fees.\n\n"
            "Pay Apex Placement Agency a fee of Rs. 1000."
        )
        item_rep = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="Pay Apex Placement Agency a fee of Rs. 1000",
            start_pos=doc_repudiated.find("Pay Apex Placement Agency a fee of Rs. 1000"),
            end_pos=doc_repudiated.find("Pay Apex Placement Agency a fee of Rs. 1000") + len("Pay Apex Placement Agency a fee of Rs. 1000"),
        )
        res_rep = reconcile_contradictions([item_rep], doc_repudiated)
        fee_rep = next(it for it in res_rep if it.id == "ADV_FEE_REGISTRATION")
        repudiating_discs = [
            it for it in res_rep
            if it.id == "DISCLAIMER_COMPANY_NO_FEE" and DISCLAIMER_AGENCY_REPUDIATION_RE.search(it.matched_text or "")
        ]
        assert len(repudiating_discs) > 0, "Repudiating disclaimer must be detected"
        rep_disc = repudiating_discs[0]
        assert rep_disc.finding_id in fee_rep.contradicts_finding_ids
        assert fee_rep.finding_id in rep_disc.contradicts_finding_ids

        # Tier 2B: Generic disclaimer without agency repudiation
        doc_generic = (
            "Our company does not charge candidates any recruitment fees.\n\n"
            "Pay Apex Placement Agency a fee of Rs. 1000."
        )
        item_gen = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="Pay Apex Placement Agency a fee of Rs. 1000",
            start_pos=doc_generic.find("Pay Apex Placement Agency a fee of Rs. 1000"),
            end_pos=doc_generic.find("Pay Apex Placement Agency a fee of Rs. 1000") + len("Pay Apex Placement Agency a fee of Rs. 1000"),
        )
        res_gen = reconcile_contradictions([item_gen], doc_generic)
        fee_gen = next(it for it in res_gen if it.id == "ADV_FEE_REGISTRATION")
        assert fee_gen.contradicts_finding_ids == [], "Generic disclaimer must NOT link to third-party agency fee"
        assert "Independent agency fee" in (fee_gen.context_required or "")

    def test_tier3_statutory_exam_fee_zero_contradiction_edges(self):
        """
        Covers:
          - Public/statutory exam fees produce zero contradiction edges against employer disclaimers
          - Preserves low effective weight 10
        """
        doc = (
            "Our company never charges fees.\n\n"
            "Application fees of Rs. 200 must be submitted to the Public Service Commission."
        )
        quote = "Application fees of Rs. 200"
        s = doc.find(quote)
        e = s + len(quote)
        exam_item = EvidenceItem(
            id="ADV_FEE_AMBIGUOUS_MENTION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.LOW,
            status=FindingStatus.AMBIGUOUS_SIGNAL,
            matched_text=quote,
            start_pos=s,
            end_pos=e,
        )
        res = reconcile_contradictions([exam_item], doc)
        t3 = next(it for it in res if it.id == "ADV_FEE_AMBIGUOUS_MENTION")
        assert t3.contradicts_finding_ids == []
        assert get_effective_weight(t3) == 10

    def test_action_context_abbreviation_aware_preservation(self):
        """
        Covers:
          - Currency abbreviations (Rs. 500) and legal entity abbreviations (Pvt. Ltd., Inc., Dr.)
            do not trigger premature sentence termination.
          - Original raw string offsets remain 100% exact.
        """
        # Case 1: Abbreviation in non-payment predicate does not truncate sentence; allows subject inheritance
        text1 = "Candidates must register with Apex Services Pvt. Ltd. and pay a registration fee of Rs. 500."
        quote1 = "pay a registration fee"
        s1 = text1.find(quote1)
        e1 = s1 + len(quote1)
        pre1, post1 = extract_action_context(text1, s1, e1)
        full_ctx1 = f"{pre1} {quote1} {post1}".strip()
        assert pre1 == "Candidates must", "Should inherit governing subject/modal root across Pvt. Ltd.!"
        assert "Rs. 500" in full_ctx1, "Rs. 500 must not be truncated by raw period split!"

        # Case 2: Abbreviation inside payment action context
        text2 = "Candidates must pay ABC Services Pvt. Ltd. a registration fee of Rs. 500."
        quote2 = "pay ABC Services Pvt. Ltd. a registration fee"
        s2 = text2.find(quote2)
        e2 = s2 + len(quote2)
        pre2, post2 = extract_action_context(text2, s2, e2)
        full_ctx2 = f"{pre2} {quote2} {post2}".strip()
        assert "Pvt. Ltd." in full_ctx2, "Pvt. Ltd. must not be truncated when inside action!"
        assert "Rs. 500" in full_ctx2, "Rs. 500 must not be truncated by raw period split!"

    def test_action_context_actor_precedence_and_cross_action_isolation(self):
        """
        Covers:
          - Actor precedence: when current segment introduces an actor, backward inheritance is skipped.
          - Prior payment/collection actions stop backward inheritance, preventing attribution leakage.
        """
        # Case 1: Actor precedence ("our company" in current segment)
        text1 = "Candidates register online and our company asks them to pay a registration fee."
        quote1 = "pay a registration fee"
        s1 = text1.find(quote1)
        e1 = s1 + len(quote1)
        pre1, _ = extract_action_context(text1, s1, e1)
        assert "our company asks them to" in pre1
        assert "Candidates" not in pre1, "Candidates must not overwrite current actor 'our company'!"

        # Case 2: Prior payment action isolation
        text2 = "Pay our company for uniforms and pay a registration fee of Rs. 500."
        quote2 = "pay a registration fee"
        s2 = text2.rfind(quote2)
        e2 = s2 + len(quote2)
        pre2, post2 = extract_action_context(text2, s2, e2)
        full2 = f"{pre2} {quote2} {post2}".strip()
        assert "Pay our company" not in full2, "Prior payment action must NOT leak employer attribution into second fee!"
        assert not TIER1_EMPLOYER_ACTION_RE.search(full2)

        # Case 3: Prior collection action isolation
        text3 = "Our company charges for uniforms and requires a registration fee of Rs. 500."
        quote3 = "requires a registration fee"
        s3 = text3.find(quote3)
        e3 = s3 + len(quote3)
        pre3, post3 = extract_action_context(text3, s3, e3)
        full3 = f"{pre3} {quote3} {post3}".strip()
        assert "charges for uniforms" not in full3, "Prior collection action must NOT leak into second fee!"
        assert not TIER1_EMPLOYER_ACTION_RE.search(full3)

    def test_clause_splitting_inherited_subject_modal_and_adversatives(self):
        """
        Covers:
          - Multiple coordinate non-payment predicates inherit governing subject/modal
          - Adversative boundaries ('but', 'however') halt backward subject inheritance
        """
        # Multiple non-payment predicates
        text1 = "Candidates must register online, submit documents and pay a registration fee directly to our company."
        quote1 = "pay a registration fee"
        s1 = text1.find(quote1)
        e1 = s1 + len(quote1)
        pre1, post1 = extract_action_context(text1, s1, e1)
        full1 = f"{pre1} {quote1} {post1}".strip()
        assert pre1 == "Candidates must"
        assert "submit documents" not in full1
        assert TIER1_EMPLOYER_ACTION_RE.search(full1)

        # Adversative boundary
        text2 = "We do not charge candidates, but our agency collects a fee of Rs. 500."
        quote2 = "fee of Rs. 500"
        s2 = text2.find(quote2)
        e2 = s2 + len(quote2)
        pre2, post2 = extract_action_context(text2, s2, e2)
        full2 = f"{pre2} {quote2} {post2}".strip()
        assert "We do not charge" not in full2, "Adversative boundary must stop backward traversal!"
        assert TIER2_AGENCY_ACTION_RE.search(full2)

    def test_warning_and_negation_eligibility_downgrade(self):
        """
        Covers:
          - Fully negated or warning-framed occurrence receives INFORMATIONAL, AMBIGUOUS_SIGNAL,
            NON_SCORABLE_INFORMATIONAL, weight 0.
          - In mixed text with an independent active claim, the active claim remains SCORABLE_ACTIVE
            and the negated occurrence is NOT promoted.
        """
        # Fully negated single occurrence
        doc_neg = "Scam Alert: Beware of recruiters demanding registration fees. Do not pay any fees."
        item_neg = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="registration fees",
            start_pos=doc_neg.find("registration fees"),
            end_pos=doc_neg.find("registration fees") + len("registration fees"),
        )
        val_neg = validate_evidence_quotation(item_neg, doc_neg)
        assert val_neg.severity == FindingSeverity.INFORMATIONAL
        assert val_neg.status == FindingStatus.AMBIGUOUS_SIGNAL
        assert val_neg.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
        assert get_effective_weight(val_neg) == 0

        # Mixed text: Warning quote in Para 1 + Active demand in Para 2
        doc_mixed = (
            "Para 1: Beware of scammers asking for a registration fee.\n\n"
            "Para 2: Candidates must pay our company a registration fee of Rs. 500."
        )
        quote = "registration fee"
        s1 = doc_mixed.find(quote)
        e1 = s1 + len(quote)
        s2 = doc_mixed.rfind(quote)
        e2 = s2 + len(quote)

        item_w = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text=quote,
            start_pos=s1,
            end_pos=e1,
        )
        item_a = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text=quote,
            start_pos=s2,
            end_pos=e2,
        )

        val_w = validate_evidence_quotation(item_w, doc_mixed)
        val_a = validate_evidence_quotation(item_a, doc_mixed)

        # Warning occurrence is downgraded to non-scorable informational
        assert val_w.severity == FindingSeverity.INFORMATIONAL
        assert val_w.eligibility == CandidateEligibility.NON_SCORABLE_INFORMATIONAL
        assert get_effective_weight(val_w) == 0

        # Active demand remains critical scorable active
        assert val_a.severity == FindingSeverity.CRITICAL
        assert val_a.status == FindingStatus.EXPLICIT_CLAIM
        assert val_a.eligibility == CandidateEligibility.SCORABLE_ACTIVE
        assert get_effective_weight(val_a) == 65
