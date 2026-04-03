from flask import Blueprint, request, jsonify
from models import get_db_connection
from classifier import classify_vitals
import json

api_bp = Blueprint('api', __name__)


def _load_json(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default

@api_bp.route('/vitals', methods=['POST'])
def receive_vitals():
    data = request.get_json()
    if not data or 'patient_id' not in data:
        return jsonify({"error": "Invalid input"}), 400

    patient_id = data['patient_id']
    spo2 = float(data.get('spo2', 0))
    temperature = float(data.get('temperature', 0))
    heart_rate = int(data.get('heart_rate', 0))

    result = classify_vitals(spo2, temperature, heart_rate)
    classification = result["triage_response"].get("risk_label", result["risk_level"])
    triage_response = json.dumps(result["triage_response"])
    conversation_history = json.dumps([])

    conn = get_db_connection()
    conn.execute(
        'INSERT INTO vitals (patient_id, spo2, temperature, heart_rate, classification, ai_recommendation, conversation_history) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (patient_id, spo2, temperature, heart_rate, classification, triage_response, conversation_history)
    )
    vital_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "vital_id": vital_id, **result}), 201


@api_bp.route('/vitals/<int:vital_id>/triage', methods=['POST'])
def continue_triage(vital_id):
    data = request.get_json() or {}
    patient_id = data.get('patient_id')
    answer = str(data.get('answer', '')).strip()

    if not patient_id or not answer:
        return jsonify({"error": "patient_id and answer are required"}), 400

    conn = get_db_connection()
    vital = conn.execute(
        'SELECT * FROM vitals WHERE id = ? AND patient_id = ?',
        (vital_id, patient_id)
    ).fetchone()

    if vital is None:
        conn.close()
        return jsonify({"error": "Vitals record not found"}), 404

    current_triage = _load_json(vital["ai_recommendation"], {})
    conversation_history = _load_json(vital["conversation_history"], [])

    if current_triage.get("type") != "question":
        conn.close()
        return jsonify({"error": "Triage has already reached a final decision"}), 409

    conversation_history.append(
        {
            "question": current_triage.get("question", ""),
            "answer": answer,
        }
    )

    result = classify_vitals(
        vital["spo2"],
        vital["temperature"],
        vital["heart_rate"],
        conversation_history,
    )

    next_triage = result["triage_response"]
    classification = next_triage.get("risk_label", result["risk_level"])

    conn.execute(
        'UPDATE vitals SET classification = ?, ai_recommendation = ?, conversation_history = ? WHERE id = ?',
        (
            classification,
            json.dumps(next_triage),
            json.dumps(conversation_history),
            vital_id,
        )
    )
    conn.commit()
    conn.close()

    return jsonify(
        {
            "status": "success",
            "vital_id": vital_id,
            "conversation_history": conversation_history,
            **result,
        }
    )
