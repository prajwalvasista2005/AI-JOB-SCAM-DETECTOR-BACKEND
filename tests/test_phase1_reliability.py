"""
Unit tests for Phase 1 Reliability Prerequisites:
  - ReDoS immunity on adversarial inputs.
  - Non-mutation of caller-owned EvidenceItem instances.
  - Unicode length-changing characters (e.g. Turkish 'İ') preserving exact raw offsets and verbatim quotes.
  - Out-of-bounds or mismatched start_pos/end_pos failing closed to CONTEXTUAL_INFERENCE and 0 weight.
  - finding_id collision resistance when mixing pre-assigned and unassigned items.
  - CONTEXTUAL_INFERENCE contributing 0 to calculate_evidence_score.
"""
import time
import unittest

from app.services.detection import (
    CandidateEligibility,
    DetectionSource,
    EvidenceItem,
    FindingSeverity,
    FindingStatus,
    ScamCategory,
    calculate_evidence_score,
    get_effective_weight,
    reconcile_contradictions,
    validate_evidence_quotation,
)
from app.services.detection.patterns import (
    COMPANY_FEE_DISCLAIMER_RE,
    DETECTION_PATTERNS,
)


class TestPhase1Reliability(unittest.TestCase):
    def test_redos_immunity_on_adversarial_fee_input(self):
        """Pattern ADV_FEE_REGISTRATION must terminate in < 50ms on long repeated 'pay to ' sequences."""
        p_fee = next(p for p in DETECTION_PATTERNS if p.pattern_id == "ADV_FEE_REGISTRATION")
        adversarial_text = "pay to " * 500 + "someone"

        t0 = time.perf_counter()
        m = p_fee.regex.search(adversarial_text)
        t1 = time.perf_counter()

        self.assertIsNone(m)
        elapsed = t1 - t0
        self.assertLess(elapsed, 0.05, f"ReDoS vulnerability detected: took {elapsed:.4f}s")

    def test_redos_immunity_on_adversarial_task_input(self):
        """Pattern TASK_RATING_OPTIMIZATION_QUOTA must terminate in < 50ms on long non-matching text."""
        p_task = next(p for p in DETECTION_PATTERNS if p.pattern_id == "TASK_RATING_OPTIMIZATION_QUOTA")
        adversarial_text = "rating apps " + "word " * 2000 + "end"

        t0 = time.perf_counter()
        m = p_task.regex.search(adversarial_text)
        t1 = time.perf_counter()

        self.assertIsNone(m)
        elapsed = t1 - t0
        self.assertLess(elapsed, 0.05, f"ReDoS vulnerability detected: took {elapsed:.4f}s")

    def test_redos_immunity_on_adversarial_disclaimer_input(self):
        """COMPANY_FEE_DISCLAIMER_RE must terminate in < 50ms on repeated fee category phrases."""
        adversarial_text = (
            "Our company does not charge candidates any "
            + "registration or recruitment or application " * 100
            + "not_fee"
        )

        t0 = time.perf_counter()
        m = COMPANY_FEE_DISCLAIMER_RE.search(adversarial_text)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        self.assertLess(elapsed, 0.05, f"ReDoS vulnerability detected: took {elapsed:.4f}s")

    def test_caller_owned_object_immutability_in_validate_quotation(self):
        """validate_evidence_quotation must not mutate the caller's EvidenceItem instance."""
        content = "We have an opening. Please pay a registration fee of Rs. 500 to proceed."
        original = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.AI_INFERENCE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="registration fee of Rs. 500",
            explanation="Caller explanation",
            finding_id=None,
            start_pos=None,
            end_pos=None,
        )

        validated = validate_evidence_quotation(original, content)

        # Validated result has resolved offsets and status
        self.assertIsNot(validated, original)
        self.assertIsNotNone(validated.start_pos)
        self.assertIsNotNone(validated.end_pos)

        # Original caller object remains completely untouched
        self.assertIsNone(original.start_pos)
        self.assertIsNone(original.end_pos)
        self.assertIsNone(original.finding_id)
        self.assertEqual(original.matched_text, "registration fee of Rs. 500")

    def test_caller_owned_object_immutability_in_reconcile_contradictions(self):
        """reconcile_contradictions must not mutate caller-supplied EvidenceItem instances."""
        content = (
            "Our company does not charge candidates any recruitment fees.\n\n"
            "Selected candidates must pay our company a registration fee of Rs. 500."
        )
        f_demand = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="pay our company a registration fee of Rs. 500",
            explanation="Direct demand",
            finding_id=None,
            contradicts_finding_ids=[],
            start_pos=78,
            end_pos=126,
        )

        reconciled = reconcile_contradictions([f_demand], content)

        # The reconciled list has 2 items with populated IDs and contradiction links
        self.assertEqual(len(reconciled), 2)
        rec_demand = next(it for it in reconciled if it.id == "ADV_FEE_REGISTRATION")
        self.assertIsNot(rec_demand, f_demand)
        self.assertTrue(len(rec_demand.contradicts_finding_ids) > 0)
        self.assertEqual(rec_demand.finding_id, "ADV_FEE_REGISTRATION_1")

        # Original caller object must NOT have been mutated
        self.assertIsNone(f_demand.finding_id)
        self.assertEqual(f_demand.contradicts_finding_ids, [])

    def test_unicode_length_changing_lowercase_exact_offsets(self):
        """Turkish 'İ' (whose lowercase 'i\\u0307' has length 2) must not cause offset drift."""
        # 'İ' is at index 0. Length of raw_content is 39.
        # If .lower() was used, raw_content.lower() length is 40.
        raw_content = "\u0130LK: Candidates must pay registration fee of Rs. 500."
        quote = "pay registration fee of Rs. 500"

        finding = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text=quote,
            start_pos=None,
            end_pos=None,
        )

        validated = validate_evidence_quotation(finding, raw_content)

        self.assertIsNotNone(validated.start_pos)
        self.assertIsNotNone(validated.end_pos)
        # Verify exact verbatim slice equality on raw_content
        self.assertEqual(
            raw_content[validated.start_pos : validated.end_pos],
            "pay registration fee of Rs. 500",
        )
        self.assertEqual(validated.matched_text, "pay registration fee of Rs. 500")
        self.assertEqual(validated.status, FindingStatus.EXPLICIT_CLAIM)

    def test_mismatched_or_out_of_bounds_offsets_fail_closed(self):
        """If start_pos/end_pos do not correspond to matched_text, finding fails closed to CONTEXTUAL_INFERENCE."""
        raw_content = "Please pay registration fee of Rs. 500 to proceed."
        finding = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="pay registration fee of Rs. 500",
            start_pos=100,  # Deliberately out of range
            end_pos=150,
        )

        validated = validate_evidence_quotation(finding, raw_content)

        # Fails closed
        self.assertIsNone(validated.matched_text)
        self.assertIsNone(validated.start_pos)
        self.assertIsNone(validated.end_pos)
        self.assertEqual(validated.status, FindingStatus.CONTEXTUAL_INFERENCE)
        self.assertEqual(validated.severity, FindingSeverity.HIGH)  # Capped from CRITICAL
        self.assertEqual(get_effective_weight(validated), 0)

    def test_mismatched_span_content_fails_closed(self):
        """If start_pos/end_pos point to different text, finding fails closed to CONTEXTUAL_INFERENCE."""
        raw_content = "Please pay registration fee of Rs. 500 to proceed."
        finding = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="pay registration fee of Rs. 500",
            start_pos=0,
            end_pos=10,  # Points to "Please pay" instead of quote
        )

        validated = validate_evidence_quotation(finding, raw_content)

        self.assertIsNone(validated.matched_text)
        self.assertEqual(validated.status, FindingStatus.CONTEXTUAL_INFERENCE)
        self.assertEqual(validated.severity, FindingSeverity.HIGH)
        self.assertEqual(get_effective_weight(validated), 0)

    def test_finding_id_collision_resistance_with_preassigned_ids(self):
        """reconcile_contradictions must never produce colliding finding_ids even when mixed with preassigned IDs."""
        raw_content = (
            "Our company does not charge candidates any recruitment fees.\n\n"
            "Our company does not charge any registration fees.\n\n"
            "Pay registration fee Rs. 500."
        )

        # Item 1 has preassigned ID "DISCLAIMER_COMPANY_NO_FEE_1"
        pre_item = EvidenceItem(
            id="DISCLAIMER_COMPANY_NO_FEE",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.INFORMATIONAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="Our company does not charge candidates any recruitment fees",
            finding_id="DISCLAIMER_COMPANY_NO_FEE_1",
            start_pos=0,
            end_pos=59,
        )

        # Item 2 has no finding_id and ID "ADV_FEE_REGISTRATION"
        demand_item = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="Pay registration fee Rs. 500",
            finding_id=None,
            start_pos=113,
            end_pos=141,
        )

        # Item 3 has preassigned ID "ADV_FEE_REGISTRATION_1"
        other_demand = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="Pay registration fee Rs. 500",
            finding_id="ADV_FEE_REGISTRATION_1",
            start_pos=113,
            end_pos=141,
        )

        reconciled = reconcile_contradictions([pre_item, demand_item, other_demand], raw_content)

        all_fids = [it.finding_id for it in reconciled]
        # Verify all finding_ids are unique
        self.assertEqual(len(all_fids), len(set(all_fids)))
        # Verify pre-assigned IDs were preserved
        self.assertIn("DISCLAIMER_COMPANY_NO_FEE_1", all_fids)
        self.assertIn("ADV_FEE_REGISTRATION_1", all_fids)
        # Verify unassigned item got ADV_FEE_REGISTRATION_2 (collision avoided)
        self.assertIn("ADV_FEE_REGISTRATION_2", all_fids)

    def test_contextual_inference_contributes_zero_weight(self):
        """Findings with FindingStatus.CONTEXTUAL_INFERENCE must yield weight 0 in get_effective_weight and scoring."""
        item_contextual = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.AI_INFERENCE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.CONTEXTUAL_INFERENCE,
            matched_text=None,
            explanation="Unquoted contextual inference",
        )

        # Weight must strictly be 0
        self.assertEqual(get_effective_weight(item_contextual), 0)
        # Score with only contextual items must strictly be 0
        self.assertEqual(calculate_evidence_score([item_contextual]), 0)

        # Adding a contextual item to an active item must NOT add any residual boost
        active_fee = EvidenceItem(
            id="ADV_FEE_REGISTRATION",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.DETERMINISTIC_RULE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            matched_text="registration fee Rs. 500",
        )
        # Active fee alone = 65
        self.assertEqual(calculate_evidence_score([active_fee]), 65)
        # Active fee + contextual inference (even in different category) = 65 (+0 residual)
        task_contextual = EvidenceItem(
            id="TASK_RATING",
            category=ScamCategory.TASK_OPTIMIZATION,
            source=DetectionSource.AI_INFERENCE,
            severity=FindingSeverity.HIGH,
            status=FindingStatus.CONTEXTUAL_INFERENCE,
        )
        self.assertEqual(calculate_evidence_score([active_fee, task_contextual]), 65)

    def test_candidate_eligibility_fail_closed(self):
        """CandidateEligibility non-scorable or rejected items yield 0 weight."""
        item_rej = EvidenceItem(
            id="ADV_FEE",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            eligibility=CandidateEligibility.REJECTED,
        )
        self.assertEqual(get_effective_weight(item_rej), 0)

        item_info = EvidenceItem(
            id="ADVISORY_GENERAL",
            category=ScamCategory.ADVANCE_FEE,
            severity=FindingSeverity.INFORMATIONAL,
            status=FindingStatus.EXPLICIT_CLAIM,
            eligibility=CandidateEligibility.NON_SCORABLE_INFORMATIONAL,
        )
        self.assertEqual(get_effective_weight(item_info), 0)

        item_ambig = EvidenceItem(
            id="ADV_FEE_AMBIGUOUS",
            category=ScamCategory.ADVANCE_FEE,
            source=DetectionSource.UNKNOWN,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.AMBIGUOUS_SIGNAL,
            eligibility=CandidateEligibility.SCORABLE_AMBIGUOUS,
        )
        # Scorable ambiguous is capped at LOW (10)
        self.assertEqual(get_effective_weight(item_ambig), 10)


if __name__ == "__main__":
    unittest.main()
