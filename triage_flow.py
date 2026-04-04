import json

from classifier import run_triage


def load_json(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def patient_payload_from_record(record):
    return {
        "age": record["age"],
        "gender": record["gender"],
        "weight": record["weight"],
        "height": record["height"],
        "spo2": record["spo2"],
        "temperature": record["temperature"],
        "heart_rate": record["heart_rate"],
    }


def run_initial_triage(patient_payload):
    result = run_triage(patient_payload, conversation_history=[])
    return build_storage_state(result, [])


def continue_triage(record, answer):
    current_triage = load_json(record["ai_recommendation"], {})
    conversation_history = load_json(record["conversation_history"], [])
    conversation_history.append(
        {
            "question": current_triage.get("question", ""),
            "answer": answer,
        }
    )
    result = run_triage(patient_payload_from_record(record), conversation_history=conversation_history)
    return build_storage_state(result, conversation_history)


def build_storage_state(result, conversation_history):
    triage_response = result["triage_response"]
    model_assessment = result["model_assessment"]
    is_final = triage_response.get("type") == "final"
    final_label = triage_response.get("risk_label") if is_final else None
    return {
        "result": result,
        "conversation_history": conversation_history,
        "classification": final_label,
        "triage_status": "completed" if is_final else "questioning",
        "final_source": "gemini" if is_final else "pending",
        "disagreement_logged": 1 if result["disagreement"] else 0,
        "ai_recommendation_json": json.dumps(triage_response),
        "conversation_history_json": json.dumps(conversation_history),
        "model_assessment_json": json.dumps(model_assessment),
    }
