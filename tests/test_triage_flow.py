import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from flask import Flask

import api
import models
import routes


def initial_question_state():
    return {
        "result": {
            "patient_data": {
                "age": 42,
                "gender": "Female",
                "weight": 68.5,
                "height": 1.66,
                "spo2": 97.0,
                "temperature": 37.1,
                "heart_rate": 84,
            },
            "model_assessment": {
                "risk_label": "low risk",
                "high_risk_probability": 0.28,
                "used_safety_override": False,
                "is_crucial": False,
            },
            "triage_response": {
                "type": "question",
                "question": "Are you short of breath?",
                "input_type": "yes_no",
                "options": ["yes", "no"],
                "should_continue_questions": True,
            },
            "disagreement": False,
        },
        "conversation_history": [],
        "classification": None,
        "triage_status": "questioning",
        "final_source": "pending",
        "disagreement_logged": 0,
        "ai_recommendation_json": json.dumps(
            {
                "type": "question",
                "question": "Are you short of breath?",
                "input_type": "yes_no",
                "options": ["yes", "no"],
                "should_continue_questions": True,
            }
        ),
        "conversation_history_json": json.dumps([]),
        "model_assessment_json": json.dumps(
            {
                "risk_label": "low risk",
                "high_risk_probability": 0.28,
                "used_safety_override": False,
                "is_crucial": False,
            }
        ),
    }


def final_state():
    return {
        "result": {
            "patient_data": {
                "age": 42,
                "gender": "Female",
                "weight": 68.5,
                "height": 1.66,
                "spo2": 97.0,
                "temperature": 37.1,
                "heart_rate": 84,
            },
            "model_assessment": {
                "risk_label": "low risk",
                "high_risk_probability": 0.28,
                "used_safety_override": False,
                "is_crucial": False,
            },
            "triage_response": {
                "type": "final",
                "risk_label": "high risk",
                "ui_color": "red",
                "is_crucial": True,
                "reasoning": "Breathing symptoms increase concern.",
                "advice": "Seek urgent medical care.",
                "recommended_next_action": "Go to urgent care now.",
                "should_continue_questions": False,
            },
            "disagreement": True,
        },
        "conversation_history": [{"question": "Are you short of breath?", "answer": "yes"}],
        "classification": "high risk",
        "triage_status": "completed",
        "final_source": "gemini",
        "disagreement_logged": 1,
        "ai_recommendation_json": json.dumps(
            {
                "type": "final",
                "risk_label": "high risk",
                "ui_color": "red",
                "is_crucial": True,
                "reasoning": "Breathing symptoms increase concern.",
                "advice": "Seek urgent medical care.",
                "recommended_next_action": "Go to urgent care now.",
                "should_continue_questions": False,
            }
        ),
        "conversation_history_json": json.dumps(
            [{"question": "Are you short of breath?", "answer": "yes"}]
        ),
        "model_assessment_json": json.dumps(
            {
                "risk_label": "low risk",
                "high_risk_probability": 0.28,
                "used_safety_override": False,
                "is_crucial": False,
            }
        ),
    }


class TriageFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.stack = ExitStack()

        def test_connection():
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn

        self.stack.enter_context(patch.object(models, "get_db_connection", side_effect=test_connection))
        self.stack.enter_context(patch.object(routes, "get_db_connection", side_effect=test_connection))
        self.stack.enter_context(patch.object(api, "get_db_connection", side_effect=test_connection))

        models.init_db()

        self.app = Flask(__name__, template_folder="../templates")
        self.app.secret_key = "test-secret"
        self.app.register_blueprint(routes.routes_bp)
        self.app.register_blueprint(api.api_bp, url_prefix="/api")
        self.client = self.app.test_client()

    def tearDown(self):
        self.stack.close()
        self.temp_dir.cleanup()

    def test_dashboard_flow_persists_questioning_then_final_result(self):
        with self.client.session_transaction() as session:
            session["patient_id"] = "PAT-001"

        with patch.object(routes, "run_initial_triage", return_value=initial_question_state()):
            response = self.client.post(
                "/triage/start",
                data={
                    "age": "42",
                    "gender": "Female",
                    "weight": "68.5",
                    "height": "1.66",
                    "spo2": "97",
                    "temperature": "37.1",
                    "heart_rate": "84",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 302)
        conn = models.get_db_connection()
        row = conn.execute("SELECT * FROM vitals").fetchone()
        conn.close()
        self.assertEqual(row["triage_status"], "questioning")
        self.assertIsNone(row["classification"])
        self.assertEqual(row["age"], 42)
        self.assertEqual(row["gender"], "Female")

        with patch.object(routes, "continue_triage", return_value=final_state()):
            response = self.client.post(
                f"/triage/{row['id']}/answer",
                data={"answer": "yes"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 302)
        conn = models.get_db_connection()
        updated = conn.execute("SELECT * FROM vitals WHERE id = ?", (row["id"],)).fetchone()
        conn.close()
        self.assertEqual(updated["classification"], "high risk")
        self.assertEqual(updated["triage_status"], "completed")
        self.assertEqual(updated["final_source"], "gemini")
        self.assertEqual(updated["disagreement_logged"], 1)
        self.assertEqual(json.loads(updated["conversation_history"]), [{"question": "Are you short of breath?", "answer": "yes"}])

    def test_api_flow_uses_extended_fields_and_returns_final(self):
        with patch.object(api, "run_initial_triage", return_value=initial_question_state()):
            response = self.client.post(
                "/api/vitals",
                json={
                    "patient_id": "PAT-API",
                    "age": 42,
                    "gender": "Female",
                    "weight": 68.5,
                    "height": 1.66,
                    "spo2": 97,
                    "temperature": 37.1,
                    "heart_rate": 84,
                },
            )

        self.assertEqual(response.status_code, 201)
        created_payload = response.get_json()
        self.assertEqual(created_payload["triage_status"], "questioning")
        self.assertEqual(created_payload["triage_response"]["type"], "question")

        with patch.object(api, "continue_triage", return_value=final_state()):
            response = self.client.post(
                f"/api/vitals/{created_payload['vital_id']}/triage",
                json={"patient_id": "PAT-API", "answer": "yes"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["triage_status"], "completed")
        self.assertEqual(payload["triage_response"]["type"], "final")
        self.assertEqual(payload["triage_response"]["risk_label"], "high risk")
        self.assertTrue(payload["disagreement"])

    def test_api_nearby_hospitals_returns_sorted_results_for_live_location(self):
        mock_payload = {
            "mode": "live_location",
            "origin": {
                "label": "Your current location",
                "lat": 36.201,
                "lng": 36.161,
            },
            "hospitals": [
                {
                    "name": "Kirikhan State Hospital",
                    "address": "Hatay",
                    "lat": 36.205,
                    "lng": 36.169,
                    "distance_meters": 640,
                },
                {
                    "name": "Nearby Hospital",
                    "address": "Hatay",
                    "lat": 36.214,
                    "lng": 36.171,
                    "distance_meters": 1800,
                },
            ],
            "fallback_used": False,
        }

        with patch.object(api, "lookup_nearby_hospitals", return_value=mock_payload):
            response = self.client.get("/api/nearby-hospitals?lat=36.201&lng=36.161")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "live_location")
        self.assertFalse(payload["fallback_used"])
        self.assertEqual(payload["origin"]["label"], "Your current location")
        self.assertEqual(payload["hospitals"][0]["name"], "Kirikhan State Hospital")
        self.assertEqual(payload["hospitals"][0]["distance_meters"], 640)

    def test_api_nearby_hospitals_rejects_invalid_coordinates(self):
        response = self.client.get("/api/nearby-hospitals?lat=abc&lng=36.1")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_api_nearby_hospitals_returns_maps_fallback_when_lookup_fails(self):
        with patch.object(api, "lookup_nearby_hospitals", side_effect=RuntimeError("service down")):
            response = self.client.get("/api/nearby-hospitals?lat=36.201&lng=36.161")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "maps_fallback")
        self.assertTrue(payload["fallback_used"])
        self.assertEqual(payload["hospitals"], [])
        self.assertIn("fallback_map_url", payload)

    def test_dashboard_shows_final_result_only_after_completion(self):
        with self.client.session_transaction() as session:
            session["patient_id"] = "PAT-FOCUS"

        conn = models.get_db_connection()
        conn.execute(
            """
            INSERT INTO vitals
            (patient_id, age, gender, weight, height, spo2, temperature, heart_rate, classification,
             triage_status, ai_recommendation, conversation_history, model_assessment, disagreement_logged, final_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "PAT-FOCUS",
                42,
                "Female",
                68.5,
                1.66,
                97,
                37.1,
                84,
                None,
                "questioning",
                json.dumps(
                    {
                        "type": "question",
                        "question": "Are you short of breath?",
                        "input_type": "yes_no",
                        "options": ["yes", "no"],
                        "should_continue_questions": True,
                    }
                ),
                json.dumps([]),
                json.dumps({"risk_label": "low risk"}),
                0,
                "pending",
            ),
        )
        conn.execute(
            """
            INSERT INTO vitals
            (patient_id, age, gender, weight, height, spo2, temperature, heart_rate, classification,
             triage_status, ai_recommendation, conversation_history, model_assessment, disagreement_logged, final_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "PAT-FOCUS",
                76,
                "Male",
                80.0,
                1.72,
                91,
                38.2,
                118,
                "high risk",
                "completed",
                json.dumps(
                    {
                        "type": "final",
                        "risk_label": "high risk",
                        "ui_color": "red",
                        "is_crucial": True,
                        "reasoning": "Breathing symptoms increase concern.",
                        "advice": "Seek urgent medical care.",
                        "recommended_next_action": "Go to urgent care now.",
                        "should_continue_questions": False,
                    }
                ),
                json.dumps([{"question": "Are you short of breath?", "answer": "yes"}]),
                json.dumps({"risk_label": "low risk"}),
                1,
                "gemini",
            ),
        )
        conn.commit()
        rows = conn.execute("SELECT id FROM vitals ORDER BY id").fetchall()
        conn.close()

        response = self.client.get(f"/dashboard?triage_id={rows[0]['id']}")
        page = response.get_data(as_text=True)
        self.assertIn("Current Question", page)
        self.assertNotIn("Final Result", page)

        response = self.client.get(f"/dashboard?triage_id={rows[1]['id']}")
        page = response.get_data(as_text=True)
        self.assertIn("Final Result", page)
        self.assertIn("Go to urgent care now.", page)
        self.assertIn("Emergency Assistance", page)
        self.assertIn('href="tel:112"', page)
        self.assertIn("Nearby Hospitals", page)
        self.assertNotIn("Directions to Nearest Hospital", page)
        self.assertIn("Your current location will be used to find nearby hospitals.", page)
        self.assertIn('id="nearbyHospitalsBtn"', page)
        self.assertIn('id="hospitalLocationStatus"', page)

    def test_dashboard_shows_emergency_section_only_for_high_risk_final(self):
        with self.client.session_transaction() as session:
            session["patient_id"] = "PAT-RISK"

        conn = models.get_db_connection()
        conn.execute(
            """
            INSERT INTO vitals
            (patient_id, age, gender, weight, height, spo2, temperature, heart_rate, classification,
             triage_status, ai_recommendation, conversation_history, model_assessment, disagreement_logged, final_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "PAT-RISK",
                34,
                "Female",
                61.0,
                1.62,
                98,
                36.9,
                78,
                "low risk",
                "completed",
                json.dumps(
                    {
                        "type": "final",
                        "risk_label": "low risk",
                        "ui_color": "green",
                        "is_crucial": False,
                        "reasoning": "Stable vitals and no concerning symptoms.",
                        "advice": "Continue monitoring symptoms.",
                        "recommended_next_action": "Arrange routine follow-up if needed.",
                        "should_continue_questions": False,
                    }
                ),
                json.dumps([]),
                json.dumps({"risk_label": "low risk"}),
                0,
                "gemini",
            ),
        )
        conn.commit()
        row = conn.execute("SELECT id FROM vitals WHERE patient_id = ?", ("PAT-RISK",)).fetchone()
        conn.close()

        response = self.client.get(f"/dashboard?triage_id={row['id']}")
        page = response.get_data(as_text=True)
        self.assertIn("Final Result", page)
        self.assertNotIn("Emergency Assistance", page)
        self.assertNotIn('href="tel:112"', page)
        self.assertNotIn("Nearby Hospitals", page)


if __name__ == "__main__":
    unittest.main()
