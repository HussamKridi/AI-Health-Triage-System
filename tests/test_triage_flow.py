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

    def test_dashboard_flow_persists_history_and_final_decision(self):
        with self.client.session_transaction() as session:
            session["patient_id"] = "PAT-001"

        with patch.object(
            routes,
            "classify_vitals",
            return_value={
                "risk_level": "low risk",
                "triage_response": {
                    "type": "question",
                    "question": "Are you short of breath?",
                    "input_type": "yes_no",
                    "options": ["yes", "no"],
                    "should_continue_questions": True,
                },
            },
        ):
            response = self.client.post(
                "/add_vitals",
                data={"spo2": "97", "temperature": "37.1", "heart_rate": "84"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 302)
        self.assertIn("triage_id=", response.headers["Location"])
        self.assertTrue(response.headers["Location"].endswith("#active-triage"))

        conn = models.get_db_connection()
        row = conn.execute("SELECT * FROM vitals").fetchone()
        conn.close()
        self.assertEqual(row["classification"], "low risk")
        self.assertEqual(json.loads(row["conversation_history"]), [])

        with patch.object(
            routes,
            "classify_vitals",
            return_value={
                "risk_level": "low risk",
                "triage_response": {
                    "type": "final",
                    "risk_label": "high risk",
                    "ui_color": "red",
                    "confidence": 0.93,
                    "reasoning": "Breathing symptoms increase concern.",
                    "advice": "Seek urgent medical care.",
                    "should_continue_questions": False,
                },
            },
        ):
            response = self.client.post(
                f"/triage/{row['id']}/answer",
                data={"answer": "yes"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 302)
        self.assertIn(f"triage_id={row['id']}", response.headers["Location"])
        self.assertTrue(response.headers["Location"].endswith("#active-triage"))

        conn = models.get_db_connection()
        updated = conn.execute("SELECT * FROM vitals WHERE id = ?", (row["id"],)).fetchone()
        conn.close()

        self.assertEqual(updated["classification"], "high risk")
        self.assertEqual(
            json.loads(updated["conversation_history"]),
            [{"question": "Are you short of breath?", "answer": "yes"}],
        )
        self.assertEqual(json.loads(updated["ai_recommendation"])["type"], "final")

    def test_api_flow_returns_vital_id_and_continues_triage(self):
        with patch.object(
            api,
            "classify_vitals",
            return_value={
                "risk_level": "high risk",
                "triage_response": {
                    "type": "question",
                    "question": "Do you have chest pain?",
                    "input_type": "yes_no",
                    "options": ["yes", "no"],
                    "should_continue_questions": True,
                },
            },
        ):
            response = self.client.post(
                "/api/vitals",
                json={
                    "patient_id": "PAT-API",
                    "spo2": 93,
                    "temperature": 38.4,
                    "heart_rate": 118,
                },
            )

        self.assertEqual(response.status_code, 201)
        created_payload = response.get_json()
        self.assertIn("vital_id", created_payload)
        self.assertEqual(created_payload["triage_response"]["type"], "question")

        with patch.object(
            api,
            "classify_vitals",
            return_value={
                "risk_level": "high risk",
                "triage_response": {
                    "type": "final",
                    "risk_label": "high risk",
                    "ui_color": "red",
                    "confidence": 0.97,
                    "reasoning": "Chest pain with abnormal vitals is high risk.",
                    "advice": "Go to urgent care or the emergency department now.",
                    "should_continue_questions": False,
                },
            },
        ):
            response = self.client.post(
                f"/api/vitals/{created_payload['vital_id']}/triage",
                json={"patient_id": "PAT-API", "answer": "yes"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["triage_response"]["type"], "final")
        self.assertEqual(payload["triage_response"]["risk_label"], "high risk")
        self.assertEqual(
            payload["conversation_history"],
            [{"question": "Do you have chest pain?", "answer": "yes"}],
        )

    def test_dashboard_can_focus_a_specific_session(self):
        with self.client.session_transaction() as session:
            session["patient_id"] = "PAT-FOCUS"

        conn = models.get_db_connection()
        conn.execute(
            """
            INSERT INTO vitals
            (patient_id, spo2, temperature, heart_rate, classification, ai_recommendation, conversation_history)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "PAT-FOCUS",
                98,
                36.8,
                74,
                "low risk",
                json.dumps(
                    {
                        "type": "final",
                        "risk_label": "low risk",
                        "ui_color": "green",
                        "confidence": 0.88,
                        "reasoning": "Stable vitals.",
                        "advice": "Monitor at home.",
                        "should_continue_questions": False,
                    }
                ),
                json.dumps([]),
            ),
        )
        conn.execute(
            """
            INSERT INTO vitals
            (patient_id, spo2, temperature, heart_rate, classification, ai_recommendation, conversation_history)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "PAT-FOCUS",
                91,
                38.2,
                118,
                "high risk",
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
            ),
        )
        conn.commit()
        rows = conn.execute("SELECT id FROM vitals ORDER BY id").fetchall()
        conn.close()

        response = self.client.get(f"/dashboard?triage_id={rows[0]['id']}")

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn(f"Session #{rows[0]['id']}", page)
        self.assertIn("Monitor at home.", page)


if __name__ == "__main__":
    unittest.main()
