import json
import os
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from model_features import build_feature_frame


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "triage_model.pkl"
MODEL_META_PATH = BASE_DIR / "triage_model_meta.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Local triage model not found at {MODEL_PATH}. Run train_model.py first."
    )

TRIAGE_MODEL = joblib.load(MODEL_PATH)
MODEL_META = {}
if MODEL_META_PATH.exists():
    MODEL_META = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment.")

GEMINI_CLIENT = genai.Client(api_key=api_key)


def _clean_text(value):
    return "".join(char for char in str(value) if char.isprintable()).strip()


def _normalize_text(value):
    cleaned = _clean_text(value).lower()
    return "".join(char for char in cleaned if char.isalnum() or char.isspace()).strip()


def _normalize_risk_label(label):
    normalized = _clean_text(label).lower()
    if normalized in {"low risk", "high risk"}:
        return normalized
    raise ValueError(f"Unsupported risk label: {label}")


def _normalize_patient_data(patient_data):
    return {
        "age": int(patient_data["age"]),
        "gender": _clean_text(patient_data["gender"]).title(),
        "weight": float(patient_data["weight"]),
        "height": float(patient_data["height"]),
        "spo2": float(patient_data["spo2"]),
        "temperature": float(patient_data["temperature"]),
        "heart_rate": int(patient_data["heart_rate"]),
    }


def _build_features(patient_data):
    return build_feature_frame(patient_data)


def _safety_override_triggered(patient_data):
    override = MODEL_META.get("safety_override", {})
    return (
        float(patient_data["spo2"]) <= float(override.get("spo2_le", -np.inf))
        or float(patient_data["temperature"]) >= float(override.get("temperature_ge", np.inf))
        or int(patient_data["heart_rate"]) >= int(override.get("heart_rate_ge", np.iinfo(np.int32).max))
        or int(patient_data["heart_rate"]) <= int(override.get("heart_rate_le", np.iinfo(np.int32).min))
    )


def _predict_high_risk_probability(features):
    if hasattr(TRIAGE_MODEL, "predict_proba"):
        probabilities = TRIAGE_MODEL.predict_proba(features)[0]
        classes = [str(label).strip().lower() for label in TRIAGE_MODEL.classes_]
        if "high risk" not in classes:
            raise ValueError(f"Model classes do not include high risk: {classes}")
        return float(probabilities[classes.index("high risk")])

    if hasattr(TRIAGE_MODEL, "decision_function"):
        decision = float(TRIAGE_MODEL.decision_function(features)[0])
        return 1.0 / (1.0 + np.exp(-decision))

    prediction = str(TRIAGE_MODEL.predict(features)[0]).strip().lower()
    return 1.0 if prediction == "high risk" else 0.0


def _local_risk_assessment(patient_data):
    normalized = _normalize_patient_data(patient_data)
    features = _build_features(normalized)
    threshold = float(MODEL_META.get("selected_threshold", 0.5))
    high_risk_probability = _predict_high_risk_probability(features)

    if _safety_override_triggered(normalized):
        return {
            "features": features,
            "risk_label": "high risk",
            "high_risk_probability": max(high_risk_probability, threshold),
            "used_safety_override": True,
            "is_crucial": True,
        }

    risk_label = "high risk" if high_risk_probability >= threshold else "low risk"
    return {
        "features": features,
        "risk_label": risk_label,
        "high_risk_probability": high_risk_probability,
        "used_safety_override": False,
        "is_crucial": risk_label == "high risk" and high_risk_probability >= max(0.75, threshold),
    }


def _asked_questions(conversation_history):
    return {
        _normalize_text(item.get("question", ""))
        for item in (conversation_history or [])
        if item.get("question")
    }


def _build_prompt(patient_data, model_assessment, conversation_history, repeat_guard=False):
    history_json = json.dumps(conversation_history or [], ensure_ascii=True)
    patient_json = json.dumps(patient_data, ensure_ascii=True)
    repeat_instruction = """
- Do not repeat, restate, or lightly rephrase a question that was already asked.
- If the next useful question would repeat prior questioning, return a final decision instead.
""".strip()
    return f"""
You are an AI medical triage assistant.

IMPORTANT:
- Final classification labels must be exactly one of:
  - "low risk"
  - "high risk"
- UI colors map as:
  - low risk -> green
  - high risk -> red
- You must ask follow-up questions one at a time.
- Do not reveal a final answer until you have enough information.
- If the case is clearly dangerous, you may stop immediately and return a final high-risk result.

AVAILABLE PATIENT INFORMATION:
{patient_json}

BACKGROUND MODEL OUTPUT:
- local model risk label: {model_assessment["risk_label"]}
- local high-risk probability: {model_assessment["high_risk_probability"]:.3f}
- local safety override used: {str(model_assessment["used_safety_override"]).lower()}

PREVIOUS QUESTIONS AND ANSWERS:
{history_json}

TASK:
1. Review the structured patient information.
2. Ask exactly one follow-up question if more information is needed.
3. Use the running Q/A history to avoid repetition.
4. Stop once you have enough information and return the final triage decision.
5. If severe symptoms or a dangerous pattern are already present, finalize immediately as high risk.

QUESTION RULES:
- Ask exactly one question at a time.
- The question must be short and clinically useful.
- Supported input types are: yes_no, text, multiple_choice.
- Prefer yes/no or multiple-choice when possible.
- Ask only questions that can change the final triage decision.
- Consider symptom severity, duration, breathing difficulty, chest pain, confusion, dizziness, known conditions, and infection symptoms.
{repeat_instruction if repeat_guard or conversation_history else ""}

OUTPUT RULES:
Return ONLY valid JSON.

If more questions are needed, return:
{{
  "type": "question",
  "question": "single next question",
  "input_type": "yes_no | text | multiple_choice",
  "options": [],
  "should_continue_questions": true
}}

If final output is ready, return:
{{
  "type": "final",
  "risk_label": "low risk" or "high risk",
  "ui_color": "green" or "red",
  "is_crucial": true or false,
  "reasoning": "brief explanation",
  "advice": "clear next-step advice",
  "recommended_next_action": "one concise action statement",
  "should_continue_questions": false
}}
""".strip()


def _is_repeated_question(question, conversation_history):
    normalized_question = _normalize_text(question)
    if not normalized_question:
        return False
    return normalized_question in _asked_questions(conversation_history)


def _fallback_question(patient_data, conversation_history):
    asked = _asked_questions(conversation_history)
    candidates = []

    if float(patient_data["spo2"]) <= 94:
        candidates.extend(
            [
                "Are you short of breath right now?",
                "Do you have chest pain right now?",
            ]
        )
    if float(patient_data["temperature"]) >= 37.8:
        candidates.extend(
            [
                "Have you had fever, chills, or recent infection symptoms?",
                "How long have these symptoms been present?",
            ]
        )
    if int(patient_data["heart_rate"]) >= 100 or int(patient_data["heart_rate"]) <= 55:
        candidates.extend(
            [
                "Do you feel faint, confused, or unusually weak?",
                "Have you had vomiting, diarrhea, or poor fluid intake today?",
            ]
        )
    if int(patient_data["age"]) >= 65:
        candidates.append("Do you have any chronic heart or lung disease?")

    candidates.extend(
        [
            "Are your symptoms getting worse quickly?",
            "Do you have severe shortness of breath, chest pain, or confusion right now?",
        ]
    )

    for candidate in candidates:
        if _normalize_text(candidate) in asked:
            continue
        is_yes_no = candidate.endswith("right now?") or candidate.startswith("Do you") or candidate.startswith("Are you") or candidate.startswith("Have you")
        return {
            "type": "question",
            "question": candidate,
            "input_type": "yes_no" if is_yes_no else "text",
            "options": ["yes", "no"] if is_yes_no else [],
            "should_continue_questions": True,
        }

    risk_label = "high risk" if _safety_override_triggered(patient_data) else "low risk"
    return {
        "type": "final",
        "risk_label": risk_label,
        "ui_color": "red" if risk_label == "high risk" else "green",
        "is_crucial": risk_label == "high risk",
        "reasoning": "The conversation stopped progressing because the next question would repeat prior questions.",
        "advice": "Seek medical care promptly if symptoms are worsening or severe. If you feel stable, continue monitoring closely.",
        "recommended_next_action": "Seek urgent care now." if risk_label == "high risk" else "Continue monitoring and arrange routine medical follow-up if symptoms persist.",
        "should_continue_questions": False,
    }


def _validate_question_response(payload):
    if payload.get("type") != "question":
        raise ValueError("Question response must have type='question'.")
    question = _clean_text(payload.get("question", ""))
    input_type = _clean_text(payload.get("input_type", "")).lower()
    if not question:
        raise ValueError("Question response is missing a question.")
    if input_type not in {"yes_no", "text", "multiple_choice"}:
        raise ValueError("Question response has unsupported input_type.")
    options = payload.get("options", [])
    if input_type == "yes_no":
        options = ["yes", "no"]
    elif input_type == "multiple_choice":
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("Multiple choice question requires at least 2 options.")
        options = [_clean_text(option) for option in options]
    else:
        options = []
    return {
        "type": "question",
        "question": question,
        "input_type": input_type,
        "options": options,
        "should_continue_questions": True,
    }


def _validate_final_response(payload):
    if payload.get("type") != "final":
        raise ValueError("Final response must have type='final'.")
    risk_label = _normalize_risk_label(payload.get("risk_label", ""))
    ui_color = _clean_text(payload.get("ui_color", "")).lower()
    expected_color = "green" if risk_label == "low risk" else "red"
    if ui_color != expected_color:
        ui_color = expected_color
    is_crucial = bool(payload.get("is_crucial", risk_label == "high risk"))
    return {
        "type": "final",
        "risk_label": risk_label,
        "ui_color": ui_color,
        "is_crucial": is_crucial,
        "reasoning": _clean_text(payload.get("reasoning", "")),
        "advice": _clean_text(payload.get("advice", "")),
        "recommended_next_action": _clean_text(payload.get("recommended_next_action", payload.get("advice", ""))),
        "should_continue_questions": False,
    }


def _validate_triage_response(payload):
    if not isinstance(payload, dict):
        raise ValueError("Gemini response was not a JSON object.")
    if payload.get("type") == "question":
        return _validate_question_response(payload)
    if payload.get("type") == "final":
        return _validate_final_response(payload)
    raise ValueError("Gemini response type must be 'question' or 'final'.")


def run_triage(patient_data, conversation_history=None):
    normalized = _normalize_patient_data(patient_data)
    model_assessment = _local_risk_assessment(normalized)
    conversation_history = conversation_history or []

    try:
        triage_response = None
        for attempt in range(2):
            prompt = _build_prompt(
                patient_data=normalized,
                model_assessment=model_assessment,
                conversation_history=conversation_history,
                repeat_guard=attempt > 0,
            )
            response = GEMINI_CLIENT.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            triage_response = getattr(response, "parsed", None)
            if triage_response is None:
                triage_response = json.loads((response.text or "").strip())
            triage_response = _validate_triage_response(triage_response)

            if triage_response["type"] != "question" or not _is_repeated_question(
                triage_response["question"], conversation_history
            ):
                break
            triage_response = None

        if triage_response is None:
            triage_response = _fallback_question(normalized, conversation_history)

    except Exception as exc:
        print(f"Gemini triage generation error: {exc}")
        triage_response = _fallback_question(normalized, conversation_history)

    disagreement = False
    if triage_response["type"] == "final":
        disagreement = triage_response["risk_label"] != model_assessment["risk_label"]

    return {
        "patient_data": normalized,
        "model_assessment": {
            "risk_label": model_assessment["risk_label"],
            "high_risk_probability": model_assessment["high_risk_probability"],
            "used_safety_override": model_assessment["used_safety_override"],
            "is_crucial": model_assessment["is_crucial"],
        },
        "triage_response": triage_response,
        "disagreement": disagreement,
    }


def classify_vitals(spo2, temperature, heart_rate, conversation_history=None):
    return run_triage(
        {
            "age": 35,
            "gender": "Unknown",
            "weight": 70,
            "height": 1.7,
            "spo2": spo2,
            "temperature": temperature,
            "heart_rate": heart_rate,
        },
        conversation_history=conversation_history,
    )
