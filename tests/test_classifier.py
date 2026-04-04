import unittest
from types import SimpleNamespace
from unittest.mock import patch

import classifier


def sample_patient(**overrides):
    patient = {
        "age": 52,
        "gender": "Female",
        "weight": 72.5,
        "height": 1.68,
        "spo2": 97,
        "temperature": 37.1,
        "heart_rate": 84,
    }
    patient.update(overrides)
    return patient


class RunTriageTests(unittest.TestCase):
    def test_returns_question_response_and_model_assessment(self):
        gemini_response = SimpleNamespace(
            parsed={
                "type": "question",
                "question": "Are you short of breath right now?",
                "input_type": "yes_no",
                "options": ["yes", "no"],
                "should_continue_questions": True,
            }
        )

        with patch.object(
            classifier,
            "_local_risk_assessment",
            return_value={
                "features": classifier._build_features(sample_patient()),
                "risk_label": "low risk",
                "high_risk_probability": 0.22,
                "used_safety_override": False,
                "is_crucial": False,
            },
        ):
            with patch.object(
                classifier.GEMINI_CLIENT.models,
                "generate_content",
                return_value=gemini_response,
            ) as mock_generate:
                result = classifier.run_triage(sample_patient(), [])

        self.assertEqual(result["triage_response"]["type"], "question")
        self.assertEqual(result["model_assessment"]["risk_label"], "low risk")
        self.assertAlmostEqual(result["model_assessment"]["high_risk_probability"], 0.22)
        self.assertFalse(result["disagreement"])
        self.assertIn('"age": 52', mock_generate.call_args.kwargs["contents"])
        self.assertIn("local model risk label: low risk", mock_generate.call_args.kwargs["contents"])

    def test_returns_final_response_and_tracks_disagreement(self):
        gemini_response = SimpleNamespace(
            parsed={
                "type": "final",
                "risk_label": "high risk",
                "ui_color": "red",
                "is_crucial": True,
                "reasoning": "Breathing symptoms and abnormal vitals increase concern.",
                "advice": "Seek urgent in-person evaluation now.",
                "recommended_next_action": "Go to urgent care now.",
                "should_continue_questions": False,
            }
        )

        with patch.object(
            classifier,
            "_local_risk_assessment",
            return_value={
                "features": classifier._build_features(sample_patient()),
                "risk_label": "low risk",
                "high_risk_probability": 0.41,
                "used_safety_override": False,
                "is_crucial": False,
            },
        ):
            with patch.object(
                classifier.GEMINI_CLIENT.models,
                "generate_content",
                return_value=gemini_response,
            ):
                result = classifier.run_triage(sample_patient(), [{"question": "Do you have chest pain?", "answer": "yes"}])

        self.assertEqual(result["triage_response"]["type"], "final")
        self.assertEqual(result["triage_response"]["risk_label"], "high risk")
        self.assertTrue(result["triage_response"]["is_crucial"])
        self.assertTrue(result["disagreement"])

    def test_falls_back_when_gemini_output_is_invalid(self):
        gemini_response = SimpleNamespace(parsed={"type": "final", "risk_label": "medium"})

        with patch.object(
            classifier,
            "_local_risk_assessment",
            return_value={
                "features": classifier._build_features(sample_patient()),
                "risk_label": "low risk",
                "high_risk_probability": 0.18,
                "used_safety_override": False,
                "is_crucial": False,
            },
        ):
            with patch.object(
                classifier.GEMINI_CLIENT.models,
                "generate_content",
                return_value=gemini_response,
            ):
                result = classifier.run_triage(sample_patient())

        self.assertEqual(result["triage_response"]["type"], "question")
        self.assertTrue(result["triage_response"]["should_continue_questions"])

    def test_repeated_question_is_retried(self):
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

        with patch.object(
            classifier,
            "_local_risk_assessment",
            return_value={
                "features": classifier._build_features(sample_patient()),
                "risk_label": "high risk",
                "high_risk_probability": 0.76,
                "used_safety_override": False,
                "is_crucial": False,
            },
        ):
            with patch.object(
                classifier.GEMINI_CLIENT.models,
                "generate_content",
                side_effect=[repeated, replacement],
            ) as mock_generate:
                result = classifier.run_triage(
                    sample_patient(),
                    [{"question": "Are you short of breath?", "answer": "no"}],
                )

        self.assertEqual(mock_generate.call_count, 2)
        self.assertEqual(result["triage_response"]["question"], "Do you have chest pain right now?")

    def test_safety_override_forces_high_risk(self):
        with patch.object(
            classifier,
            "MODEL_META",
            {
                "selected_threshold": 0.5,
                "safety_override": {
                    "spo2_le": 92,
                    "temperature_ge": 39.0,
                    "heart_rate_ge": 130,
                    "heart_rate_le": 45,
                },
            },
        ):
            with patch.object(classifier, "_predict_high_risk_probability", return_value=0.12):
                result = classifier._local_risk_assessment(sample_patient(spo2=91))

        self.assertEqual(result["risk_label"], "high risk")
        self.assertTrue(result["used_safety_override"])


if __name__ == "__main__":
    unittest.main()
