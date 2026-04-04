from flask import Blueprint, jsonify, request

from models import get_db_connection
from triage_flow import continue_triage, load_json, run_initial_triage


api_bp = Blueprint("api", __name__)


def _parse_payload(data):
    return {
        "age": int(data["age"]),
        "gender": str(data["gender"]),
        "weight": float(data["weight"]),
        "height": float(data["height"]),
        "spo2": float(data["spo2"]),
        "temperature": float(data["temperature"]),
        "heart_rate": int(data["heart_rate"]),
    }


@api_bp.route("/vitals", methods=["POST"])
def receive_vitals():
    data = request.get_json() or {}
    required = {"patient_id", "age", "gender", "weight", "height", "spo2", "temperature", "heart_rate"}
    if not required.issubset(data):
        return jsonify({"error": "Missing required triage fields"}), 400

    patient_id = data["patient_id"]
    payload = _parse_payload(data)
    state = run_initial_triage(payload)

    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO vitals
        (patient_id, age, gender, weight, height, spo2, temperature, heart_rate, classification,
         triage_status, ai_recommendation, conversation_history, model_assessment, disagreement_logged, final_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            payload["age"],
            payload["gender"],
            payload["weight"],
            payload["height"],
            payload["spo2"],
            payload["temperature"],
            payload["heart_rate"],
            state["classification"],
            state["triage_status"],
            state["ai_recommendation_json"],
            state["conversation_history_json"],
            state["model_assessment_json"],
            state["disagreement_logged"],
            state["final_source"],
        ),
    )
    vital_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()

    return jsonify(
        {
            "status": "success",
            "vital_id": vital_id,
            "triage_status": state["triage_status"],
            **state["result"],
        }
    ), 201


@api_bp.route("/vitals/<int:vital_id>/triage", methods=["POST"])
def continue_triage_route(vital_id):
    data = request.get_json() or {}
    patient_id = data.get("patient_id")
    answer = str(data.get("answer", "")).strip()

    if not patient_id or not answer:
        return jsonify({"error": "patient_id and answer are required"}), 400

    conn = get_db_connection()
    vital = conn.execute(
        "SELECT * FROM vitals WHERE id = ? AND patient_id = ?",
        (vital_id, patient_id),
    ).fetchone()
    if vital is None:
        conn.close()
        return jsonify({"error": "Vitals record not found"}), 404

    current_triage = load_json(vital["ai_recommendation"], {})
    if current_triage.get("type") != "question":
        conn.close()
        return jsonify({"error": "Triage has already reached a final decision"}), 409

    state = continue_triage(vital, answer)
    conn.execute(
        """
        UPDATE vitals
        SET classification = ?, triage_status = ?, ai_recommendation = ?, conversation_history = ?,
            model_assessment = ?, disagreement_logged = ?, final_source = ?
        WHERE id = ?
        """,
        (
            state["classification"],
            state["triage_status"],
            state["ai_recommendation_json"],
            state["conversation_history_json"],
            state["model_assessment_json"],
            state["disagreement_logged"],
            state["final_source"],
            vital_id,
        ),
    )
    conn.commit()
    conn.close()

    return jsonify(
        {
            "status": "success",
            "vital_id": vital_id,
            "triage_status": state["triage_status"],
            "conversation_history": state["conversation_history"],
            **state["result"],
        }
    )
