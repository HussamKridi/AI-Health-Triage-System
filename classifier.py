import json
import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from google import genai
from google.genai import types
from model_features import build_feature_frame


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "triage_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Local triage model not found at {MODEL_PATH}. Run train_model.py first."
    )

TRIAGE_MODEL = joblib.load(MODEL_PATH)

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


def _build_features(spo2, temperature, heart_rate):
    return build_feature_frame(
        {
            "Oxygen Saturation": float(spo2),
            "Body Temperature": float(temperature),
            "Heart Rate": int(heart_rate),
        }
    )


def _asked_questions(conversation_history):
    return {
        _normalize_text(item.get("question", ""))
        for item in (conversation_history or [])
        if item.get("question")
    }


def _build_prompt(spo2, temperature, heart_rate, risk_level, conversation_history, repeat_guard=False):
    history_json = json.dumps(conversation_history or [], ensure_ascii=True)
    repeat_instruction = """
- Do not repeat, restate, or rephrase a question that was already asked in the previous questions and answers.
- If the next useful question would be repetitive, return a final decision instead.
""".strip()
    return f"""
You are an AI triage assistant integrated into a patient dashboard.

IMPORTANT:
- The final classification must follow the trained dataset labels ONLY:
  - "low risk"
  - "high risk"
- Do not invent other labels such as "green", "red", "medium", or anything else.
- Color is a UI property, not a class label:
  - low risk -> green
  - high risk -> red

You already have a deterministic local model risk estimate. Use it as an anchor unless the symptoms clearly justify high risk.

Your job is to:
1. Review the available vitals.
2. Ask follow-up questions one at a time.
3. Adapt each next question based on the current case, previous answers, and abnormal vitals.
4. Stop asking questions when enough information has been collected to confidently match the case to the trained dataset pattern.
5. Return either the next question or the final decision.

AVAILABLE INPUTS:
- SpO2: {float(spo2)}
- Temperature: {float(temperature)}
- Heart Rate: {int(heart_rate)}
- Local model risk label: {risk_level}
- Previous questions and answers: {history_json}

QUESTION BEHAVIOR:
- Ask only one question at a time.
- Never ask multiple questions in one response.
- Every question must be answerable in the UI with a text field, yes/no buttons, or multiple choice.
- Prefer short, clinically relevant questions.
- Ask follow-up questions only when they help distinguish between low risk and high risk according to the trained dataset.
- Do not ask unnecessary questions.
- If severe danger signs appear, stop and classify immediately as high risk.

QUESTION STRATEGY:
- Start with the most important symptom-discriminating question based on the vitals.
- Examples of useful areas:
  - shortness of breath
  - chest pain
  - unusual fatigue
  - dizziness
  - cough or fever symptoms
  - duration of symptoms
  - known heart/lung illness
  - recent infection
- If SpO2 is low, prioritize breathing-related questions.
- If temperature is elevated, prioritize infection-related questions.
- If heart rate is abnormal, prioritize circulation, dehydration, pain, or anxiety-related questions.
- If the patient reports severe symptoms, classify high risk without continuing.
{repeat_instruction if repeat_guard or conversation_history else ""}

STOP RULE:
Stop asking questions when one of these is true:
1. You have enough information to confidently assign "low risk" or "high risk" based on the trained dataset pattern.
2. A high-risk symptom pattern is already clear.
3. Additional questions would not meaningfully change the classification.

OUTPUT RULES:
You must return ONLY valid JSON.

If more information is needed, return:
{{
  "type": "question",
  "question": "single next question here",
  "input_type": "yes_no | text | multiple_choice",
  "options": ["option1", "option2"],
  "should_continue_questions": true
}}

If enough information is available, return:
{{
  "type": "final",
  "risk_label": "low risk" or "high risk",
  "ui_color": "green" or "red",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation based on symptoms, vitals, and answers",
  "advice": "clear next-step advice for the patient",
  "should_continue_questions": false
}}

IMPORTANT CONSTRAINTS:
- Final label must always be exactly one of:
  - low risk
  - high risk
- Never output "green" or "red" as the risk class itself.
- "ui_color" is only for frontend display.
- Ask questions one by one.
- Keep reasoning brief and clinically relevant.
- When enough information is collected, stop and decide.
""".strip()


def _is_repeated_question(question, conversation_history):
    normalized_question = _normalize_text(question)
    if not normalized_question:
        return False
    return normalized_question in _asked_questions(conversation_history)


def _fallback_question(spo2, temperature, heart_rate, conversation_history):
    asked = _asked_questions(conversation_history)
    candidates = []

    if float(spo2) < 95:
        candidates.extend(
            [
                "Are you having shortness of breath right now?",
                "Do you have chest pain when breathing or at rest?",
            ]
        )
    if float(temperature) >= 38.0:
        candidates.extend(
            [
                "Have you had fever, chills, or a recent infection?",
                "How long have the fever symptoms been present?",
            ]
        )
    if int(heart_rate) >= 100 or int(heart_rate) <= 50:
        candidates.extend(
            [
                "Are you feeling dizzy, faint, or unusually weak?",
                "Have you had vomiting, diarrhea, or poor fluid intake today?",
            ]
        )

    candidates.extend(
        [
            "Do you have a known heart or lung condition?",
            "Did these symptoms start suddenly today or build up over time?",
            "Are you having shortness of breath, chest pain, or severe dizziness right now?",
        ]
    )

    for candidate in candidates:
        if _normalize_text(candidate) not in asked:
            return {
                "type": "question",
                "question": candidate,
                "input_type": "yes_no" if candidate.endswith("right now?") or candidate.startswith("Do you") or candidate.startswith("Are you") or candidate.startswith("Have you") else "text",
                "options": ["yes", "no"] if candidate.endswith("right now?") or candidate.startswith("Do you") or candidate.startswith("Are you") or candidate.startswith("Have you") else [],
                "should_continue_questions": True,
            }

    return {
        "type": "final",
        "risk_label": "high risk" if float(spo2) < 95 or float(temperature) >= 38.0 or int(heart_rate) >= 100 else "low risk",
        "ui_color": "red" if float(spo2) < 95 or float(temperature) >= 38.0 or int(heart_rate) >= 100 else "green",
        "confidence": 0.65,
        "reasoning": "The conversation stopped progressing because the next question would repeat prior triage questions.",
        "advice": "Seek medical review if symptoms are worsening. If you feel stable, continue monitoring and follow standard care guidance.",
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

    confidence = payload.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "type": "final",
        "risk_label": risk_label,
        "ui_color": ui_color,
        "confidence": confidence,
        "reasoning": _clean_text(payload.get("reasoning", "")),
        "advice": _clean_text(payload.get("advice", "")),
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


def classify_vitals(spo2, temperature, heart_rate, conversation_history=None):
    features = _build_features(spo2, temperature, heart_rate)
    local_risk_level = _normalize_risk_label(TRIAGE_MODEL.predict(features)[0])
    conversation_history = conversation_history or []

    try:
        triage_response = None
        for attempt in range(2):
            prompt = _build_prompt(
                spo2=spo2,
                temperature=temperature,
                heart_rate=heart_rate,
                risk_level=local_risk_level,
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
            triage_response = _fallback_question(
                spo2=spo2,
                temperature=temperature,
                heart_rate=heart_rate,
                conversation_history=conversation_history,
            )

        return {
            "risk_level": local_risk_level,
            "triage_response": triage_response,
        }

    except Exception as exc:
        print(f"Gemini triage generation error: {exc}")
        return {
            "risk_level": local_risk_level,
            "triage_response": _fallback_question(
                spo2=spo2,
                temperature=temperature,
                heart_rate=heart_rate,
                conversation_history=conversation_history,
            ),
        }
