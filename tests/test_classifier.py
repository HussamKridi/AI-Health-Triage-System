import unittest
from types import SimpleNamespace
from unittest.mock import patch

import classifier


class ClassifyVitalsTests(unittest.TestCase):
    def test_returns_question_response_with_normalized_low_risk_label(self):
        gemini_response = SimpleNamespace(
            parsed={
                "type": "question",
                "question": "Are you short of breath right now?",
                "input_type": "yes_no",
                "options": ["yes", "no"],
                "should_continue_questions": True,
            }
        )

        with patch.object(classifier.TRIAGE_MODEL, "predict", return_value=["Low Risk"]) as mock_predict:
            with patch.object(
                classifier.GEMINI_CLIENT.models,
                "generate_content",
                return_value=gemini_response,
            ) as mock_generate:
                result = classifier.classify_vitals(97, 37.1, 84, [])

        self.assertEqual(result["risk_level"], "low risk")
        self.assertEqual(result["triage_response"]["type"], "question")
        self.assertEqual(result["triage_response"]["input_type"], "yes_no")
        self.assertEqual(result["triage_response"]["options"], ["yes", "no"])

        features = mock_predict.call_args.args[0]
        self.assertIn("Oxygen Saturation", list(features.columns))
        self.assertIn("Body Temperature", list(features.columns))
        self.assertIn("Heart Rate", list(features.columns))
        self.assertIn("Temperature Elevation", list(features.columns))
        self.assertIn("Tachycardia Flag", list(features.columns))
        self.assertEqual(features.iloc[0].to_dict()["Oxygen Saturation"], 97.0)
        self.assertIn("Local model risk label: low risk", mock_generate.call_args.kwargs["contents"])

    def test_returns_final_response_with_strict_labels(self):
        gemini_response = SimpleNamespace(
            parsed={
                "type": "final",
                "risk_label": "high risk",
                "ui_color": "red",
                "confidence": 0.91,
                "reasoning": "Low oxygen and severe symptoms increase concern.",
                "advice": "Seek urgent in-person evaluation now.",
                "should_continue_questions": False,
            }
        )

        with patch.object(classifier.TRIAGE_MODEL, "predict", return_value=["High Risk"]):
            with patch.object(
                classifier.GEMINI_CLIENT.models,
                "generate_content",
                return_value=gemini_response,
            ):
                result = classifier.classify_vitals(
                    89,
                    38.5,
                    120,
                    [{"question": "Are you short of breath?", "answer": "Yes"}],
                )

        self.assertEqual(result["risk_level"], "high risk")
        self.assertEqual(result["triage_response"]["type"], "final")
        self.assertEqual(result["triage_response"]["risk_label"], "high risk")
        self.assertEqual(result["triage_response"]["ui_color"], "red")
        self.assertFalse(result["triage_response"]["should_continue_questions"])

    def test_falls_back_to_default_question_when_gemini_output_is_invalid(self):
        gemini_response = SimpleNamespace(parsed={"type": "final", "risk_label": "medium"})

        with patch.object(classifier.TRIAGE_MODEL, "predict", return_value=["Low Risk"]):
            with patch.object(
                classifier.GEMINI_CLIENT.models,
                "generate_content",
                return_value=gemini_response,
            ):
                result = classifier.classify_vitals(98, 36.9, 75)

        self.assertEqual(result["risk_level"], "low risk")
        self.assertEqual(result["triage_response"]["type"], "question")
        self.assertEqual(result["triage_response"]["input_type"], "yes_no")
        self.assertTrue(result["triage_response"]["should_continue_questions"])

    def test_repeated_question_is_retried_and_replaced(self):
        repeated = SimpleNamespace(
            parsed={
                "type": "question",
                "question": "Are you short of breath?",
                "input_type": "yes_no",
                "options": ["yes", "no"],
                "should_continue_questions": True,
            }
        )
        replacement = SimpleNamespace(
            parsed={
                "type": "question",
                "question": "Do you have chest pain right now?",
                "input_type": "yes_no",
                "options": ["yes", "no"],
                "should_continue_questions": True,
            }
        )

        with patch.object(classifier.TRIAGE_MODEL, "predict", return_value=["High Risk"]):
            with patch.object(
                classifier.GEMINI_CLIENT.models,
                "generate_content",
                side_effect=[repeated, replacement],
            ) as mock_generate:
                result = classifier.classify_vitals(
                    92,
                    38.0,
                    115,
                    [{"question": "Are you short of breath?", "answer": "no"}],
                )

        self.assertEqual(mock_generate.call_count, 2)
        self.assertEqual(result["triage_response"]["question"], "Do you have chest pain right now?")

    def test_repeated_question_falls_back_to_non_repeating_prompt(self):
        repeated = SimpleNamespace(
            parsed={
                "type": "question",
                "question": "Are you short of breath?",
                "input_type": "yes_no",
                "options": ["yes", "no"],
                "should_continue_questions": True,
            }
        )

        with patch.object(classifier.TRIAGE_MODEL, "predict", return_value=["High Risk"]):
            with patch.object(
                classifier.GEMINI_CLIENT.models,
                "generate_content",
                side_effect=[repeated, repeated],
            ):
                result = classifier.classify_vitals(
                    92,
                    38.0,
                    115,
                    [{"question": "Are you short of breath?", "answer": "no"}],
                )

        self.assertEqual(result["triage_response"]["type"], "question")
        self.assertNotEqual(result["triage_response"]["question"], "Are you short of breath?")


if __name__ == "__main__":
    unittest.main()
