import unittest

import testimonial_pipeline as pipeline


class TestPipelineRegression(unittest.TestCase):
    def test_quote_fallback_recovers_verbatim_quote(self) -> None:
        comment_text = (
            "why_recommend: Yes, things are going smoothly, and I'm satisfied with my studies.\n"
            "why_aberdeen: I was primarily attracted to the University of Aberdeen due to its esteemed reputation "
            "as a UK university. What truly distinguished Aberdeen for me, however, was the remarkably clear and "
            "simple presentation of degree information, costs, courses, and academic pathways on your website."
        )
        analysis = {"label": "very_positive", "score": 90.1}
        profile = pipeline.build_quote_candidate_profile(comment_text)
        quote_payload = pipeline.quote_fallback("verbatim_quote_unusable")

        repaired, used_fallback = pipeline.apply_verbatim_quote_fallback(quote_payload, comment_text, analysis, profile)

        self.assertTrue(used_fallback)
        self.assertTrue(repaired["short_quote"])
        self.assertTrue(repaired["long_quote"])
        self.assertIn("why_aberdeen", repaired["source_field"])
        self.assertTrue(pipeline.quote_status_is_ok(pipeline.quote_status_from_payload(repaired)))

    def test_quote_gate_allows_short_but_strong_candidate(self) -> None:
        comment_text = (
            "typical_day: After work at home\n"
            "why_recommend: Really good structure, website easy to navigate\n"
            "why_aberdeen: Approved supplier with company"
        )
        analysis = {"label": "very_positive", "score": 97.5}
        profile = pipeline.build_quote_candidate_profile(comment_text)

        should_extract, reason = pipeline.should_extract_quote(analysis, True, comment_text, profile)

        self.assertTrue(should_extract)
        self.assertEqual(reason, "")

    def test_quote_gate_rejects_too_short_fragment(self) -> None:
        comment_text = (
            "typical_day: Evenings at home.\n"
            "why_recommend: Good learning environment.\n"
            "why_aberdeen: Alumni discount"
        )
        analysis = {"label": "positive", "score": 78.8}
        profile = pipeline.build_quote_candidate_profile(comment_text)

        should_extract, reason = pipeline.should_extract_quote(analysis, True, comment_text, profile)

        self.assertFalse(should_extract)
        self.assertEqual(reason, "too_short")

    def test_quote_gate_rejects_without_consent(self) -> None:
        comment_text = (
            "why_recommend: The course was excellent and the support was fantastic.\n"
            "why_aberdeen: Prestigious reputation and flexible online structure."
        )
        analysis = {"label": "very_positive", "score": 94.0}
        profile = pipeline.build_quote_candidate_profile(comment_text)

        should_extract, reason = pipeline.should_extract_quote(analysis, False, comment_text, profile)

        self.assertFalse(should_extract)
        self.assertEqual(reason, "no_consent")

    def test_low_evidence_extreme_positive_is_capped(self) -> None:
        score = pipeline.recalibrate_sentiment_score(
            raw_score=98.0,
            label="very_positive",
            comment_text="why_recommend: Course was perfect.\nwhy_aberdeen: Great resources.",
            summary="The student loved the course.",
            notes="",
        )

        self.assertLess(score, 89.0)

    def test_manual_review_flags_severe_distress(self) -> None:
        flags = pipeline.build_manual_review_flags(
            "negative_feedback: I lost 5k and got nothing out of this experience, just depression and loss of will to go on.",
            "very_negative",
            1.0,
        )

        self.assertTrue(flags["severe_distress_flag"])
        self.assertTrue(flags["manual_review_required"])
        self.assertIn("severe_distress", flags["manual_review_reason"])

    def test_quote_status_helper_outputs_specific_states(self) -> None:
        self.assertEqual(pipeline.quote_status_from_payload({"short_quote": "Great support.", "long_quote": ""}), "ok_short_only")
        self.assertEqual(
            pipeline.quote_status_from_payload({"short_quote": "", "long_quote": "The course was fantastic and flexible."}),
            "ok_long_only",
        )
        self.assertEqual(
            pipeline.quote_status_from_payload({"short_quote": "Great support.", "long_quote": "The course was fantastic and flexible."}),
            "ok_short_and_long",
        )


if __name__ == "__main__":
    unittest.main()
