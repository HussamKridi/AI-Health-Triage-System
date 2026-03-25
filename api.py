from flask import Blueprint, request, jsonify
from models import get_db_connection
from classifier import classify_vitals

api_bp = Blueprint('api', __name__)

@api_bp.route('/vitals', methods=['POST'])
def receive_vitals():
    data = request.get_json()
    if not data or 'patient_id' not in data:
        return jsonify({"error": "Invalid input"}), 400

    patient_id = data['patient_id']
    spo2 = float(data.get('spo2', 0))
    temperature = float(data.get('temperature', 0))
    heart_rate = int(data.get('heart_rate', 0))

    classification = classify_vitals(spo2, temperature, heart_rate)

    conn = get_db_connection()
    conn.execute(
        'INSERT INTO vitals (patient_id, spo2, temperature, heart_rate, classification) VALUES (?, ?, ?, ?, ?)',
        (patient_id, spo2, temperature, heart_rate, classification)
    )
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "classification": classification}), 201