from flask import Blueprint, redirect, render_template, request, session, url_for

from models import get_db_connection
from triage_flow import continue_triage, load_json, run_initial_triage


routes_bp = Blueprint("routes", __name__)


def _display_input_type(input_type):
    mapping = {
        "yes_no": "Yes or No",
        "multiple_choice": "Multiple Choice",
        "text": "Short Text Response",
    }
    return mapping.get(str(input_type or "").strip().lower(), "Response")


def _format_delta(delta_value, suffix=""):
    if delta_value is None:
        return "No comparison"
    if delta_value > 0:
        return f"+{delta_value}{suffix}"
    return f"{delta_value}{suffix}"


def _vital_signal_summary(label, value, unit, severity, rationale, normal_range):
    return {
        "label": label,
        "value": value,
        "unit": unit,
        "severity": severity,
        "rationale": rationale,
        "normal_range": normal_range,
    }


def _build_vital_signals(vital):
    if not vital:
        return []

    spo2 = float(vital["spo2"])
    temperature = float(vital["temperature"])
    heart_rate = int(vital["heart_rate"])
    signals = []

    if spo2 <= 92:
        signals.append(_vital_signal_summary("SpO2", spo2, "%", "critical", "Oxygen saturation is severely low.", "95-100%"))
    elif spo2 <= 94:
        signals.append(_vital_signal_summary("SpO2", spo2, "%", "elevated", "Oxygen saturation is below the normal range.", "95-100%"))
    else:
        signals.append(_vital_signal_summary("SpO2", spo2, "%", "stable", "Oxygen saturation is within the normal range.", "95-100%"))

    if temperature >= 39.0:
        signals.append(_vital_signal_summary("Temperature", temperature, "°C", "critical", "High fever can indicate significant infection risk.", "36.1-37.5°C"))
    elif temperature >= 37.8:
        signals.append(_vital_signal_summary("Temperature", temperature, "°C", "elevated", "Temperature is mildly above the usual range.", "36.1-37.5°C"))
    elif temperature < 35.5:
        signals.append(_vital_signal_summary("Temperature", temperature, "°C", "critical", "Low body temperature can be clinically concerning.", "36.1-37.5°C"))
    else:
        signals.append(_vital_signal_summary("Temperature", temperature, "°C", "stable", "Temperature is within the expected range.", "36.1-37.5°C"))

    if heart_rate >= 120 or heart_rate <= 45:
        signals.append(_vital_signal_summary("Heart Rate", heart_rate, "BPM", "critical", "Heart rate is in a high-risk range.", "60-100 BPM"))
    elif heart_rate >= 100 or heart_rate <= 55:
        signals.append(_vital_signal_summary("Heart Rate", heart_rate, "BPM", "elevated", "Heart rate is outside the usual resting range.", "60-100 BPM"))
    else:
        signals.append(_vital_signal_summary("Heart Rate", heart_rate, "BPM", "stable", "Heart rate is within the expected range.", "60-100 BPM"))

    return signals


def _build_comparison(current_vital, previous_vital):
    if not current_vital:
        return {
            "has_previous": False,
            "headline": "No previous session available for comparison.",
            "rows": [],
        }

    if not previous_vital:
        return {
            "has_previous": False,
            "headline": "This is the first recorded triage session for this patient.",
            "rows": [],
        }

    current_spo2 = float(current_vital["spo2"])
    current_temp = float(current_vital["temperature"])
    current_hr = int(current_vital["heart_rate"])
    previous_spo2 = float(previous_vital["spo2"])
    previous_temp = float(previous_vital["temperature"])
    previous_hr = int(previous_vital["heart_rate"])

    comparison_items = [
        {
            "label": "SpO2",
            "current": f"{current_spo2:.1f}%",
            "previous": f"{previous_spo2:.1f}%",
            "delta": _format_delta(round(current_spo2 - previous_spo2, 1), "%"),
            "direction": "better" if current_spo2 > previous_spo2 else "worse" if current_spo2 < previous_spo2 else "same",
        },
        {
            "label": "Temperature",
            "current": f"{current_temp:.1f}°C",
            "previous": f"{previous_temp:.1f}°C",
            "delta": _format_delta(round(current_temp - previous_temp, 1), "°C"),
            "direction": "worse" if current_temp > previous_temp else "better" if current_temp < previous_temp else "same",
        },
        {
            "label": "Heart Rate",
            "current": f"{current_hr} BPM",
            "previous": f"{previous_hr} BPM",
            "delta": _format_delta(current_hr - previous_hr, " BPM"),
            "direction": "worse" if current_hr > previous_hr else "better" if current_hr < previous_hr else "same",
        },
    ]

    worsening_count = sum(1 for item in comparison_items if item["direction"] == "worse")
    if worsening_count >= 2:
        headline = "Patient status appears worse than the previous recorded session."
        trend = "worsening"
    elif all(item["direction"] == "same" for item in comparison_items):
        headline = "Vitals are largely unchanged compared with the previous session."
        trend = "stable"
    else:
        headline = "Some vitals improved, but clinical review is still recommended."
        trend = "mixed"

    return {
        "has_previous": True,
        "headline": headline,
        "trend": trend,
        "trend_tone": "changed" if trend == "worsening" else "stable" if trend == "stable" else "mixed",
        "previous_session_id": previous_vital["id"],
        "previous_timestamp": previous_vital["timestamp"],
        "rows": comparison_items,
    }


def _build_reasoning_summary(latest_vital, latest_triage, latest_history, model_assessment):
    if not latest_vital:
        return {}

    final_type = latest_triage.get("type")
    final_label = latest_triage.get("risk_label") if final_type == "final" else None
    model_label = model_assessment.get("risk_label")
    agreement = None
    if final_label and model_label:
        agreement = "aligned" if final_label == model_label else "changed"

    timeline = [
        {
            "title": "Intake recorded",
            "detail": f"Vitals captured for session #{latest_vital['id']}.",
            "tone": "neutral",
        },
        {
            "title": "Local model baseline",
            "detail": (
                f"Model marked the case as {model_label or 'pending'}"
                f" with {round(float(model_assessment.get('high_risk_probability', 0)) * 100)}% high-risk probability."
            ),
            "tone": "neutral",
        },
    ]

    if model_assessment.get("used_safety_override"):
        timeline.append(
            {
                "title": "Safety override triggered",
                "detail": "Critical vital thresholds forced the local model into a conservative high-risk posture.",
                "tone": "critical",
            }
        )

    for index, item in enumerate(latest_history, start=1):
        timeline.append(
            {
                "title": f"Follow-up {index}",
                "detail": f"Q: {item.get('question', '')} A: {item.get('answer', '')}",
                "tone": "neutral",
            }
        )

    if final_type == "final":
        timeline.append(
            {
                "title": "Final decision",
                "detail": latest_triage.get("recommended_next_action", "Triage finished."),
                "tone": "critical" if final_label == "high risk" else "positive",
            }
        )
    elif final_type == "question":
        timeline.append(
            {
                "title": "Waiting for next answer",
                "detail": latest_triage.get("question", "Next question pending."),
                "tone": "warning",
            }
        )

    handover_summary = (
        f"Patient {latest_vital['patient_id']} is in session #{latest_vital['id']} with SpO2 {latest_vital['spo2']}%, "
        f"temperature {latest_vital['temperature']}°C, and heart rate {latest_vital['heart_rate']} BPM. "
        f"The local model baseline is {model_label or 'pending'}."
    )

    if latest_history:
        latest_exchange = latest_history[-1]
        handover_summary += (
            f" Latest follow-up: {latest_exchange.get('question', '')} Answer: {latest_exchange.get('answer', '')}."
        )

    if final_type == "final":
        handover_summary += (
            f" Final triage decision is {final_label}. Reasoning: {latest_triage.get('reasoning', '')} "
            f"Recommended action: {latest_triage.get('recommended_next_action', '')}"
        )
    else:
        handover_summary += f" The triage is still in progress. Current question: {latest_triage.get('question', '')}"

    return {
        "signals": _build_vital_signals(latest_vital),
        "agreement": agreement,
        "timeline": timeline,
        "handover_summary": handover_summary.strip(),
    }


def _dashboard_redirect(triage_id=None):
    target = url_for("routes.dashboard", triage_id=triage_id) if triage_id else url_for("routes.dashboard")
    return redirect(f"{target}#active-triage")


def _parse_patient_form(form_data):
    return {
        "age": int(form_data["age"]),
        "gender": form_data["gender"],
        "weight": float(form_data["weight"]),
        "height": float(form_data["height"]),
        "spo2": float(form_data["spo2"]),
        "temperature": float(form_data["temperature"]),
        "heart_rate": int(form_data["heart_rate"]),
    }


@routes_bp.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["patient_id"] = request.form["patient_id"]
        return _dashboard_redirect()
    return render_template("login.html")


@routes_bp.route("/dashboard")
def dashboard():
    if "patient_id" not in session:
        return redirect(url_for("routes.login"))

    patient_id = session["patient_id"]
    conn = get_db_connection()
    vitals = conn.execute(
        "SELECT * FROM vitals WHERE patient_id = ? ORDER BY timestamp DESC",
        (patient_id,),
    ).fetchall()
    conn.close()

    selected_vital_id = request.args.get("triage_id", type=int)
    fresh_questioning = request.args.get("fresh", type=int) == 1
    active_vital = None
    if vitals and not fresh_questioning:
        if selected_vital_id is not None:
            active_vital = next((vital for vital in vitals if vital["id"] == selected_vital_id), None)
        if active_vital is None:
            active_vital = vitals[0]

    session_number_by_id = {}
    for index, vital in enumerate(reversed(vitals), start=1):
        session_number_by_id[vital["id"]] = index

    latest_vital = active_vital
    latest_triage = load_json(latest_vital["ai_recommendation"], {}) if latest_vital else {}
    latest_history = load_json(latest_vital["conversation_history"], []) if latest_vital else []
    model_assessment = load_json(latest_vital["model_assessment"], {}) if latest_vital else {}
    previous_vital = None
    if latest_vital:
        for vital in vitals:
            if vital["id"] == latest_vital["id"]:
                continue
            if vital["timestamp"] <= latest_vital["timestamp"]:
                previous_vital = vital
                break

    comparison = _build_comparison(latest_vital, previous_vital)
    reasoning_summary = _build_reasoning_summary(latest_vital, latest_triage, latest_history, model_assessment)
    intake_defaults = {
        "age": latest_vital["age"] if latest_vital and not fresh_questioning else "",
        "gender": latest_vital["gender"] if latest_vital and not fresh_questioning else "",
        "weight": latest_vital["weight"] if latest_vital and not fresh_questioning else "",
        "height": latest_vital["height"] if latest_vital and not fresh_questioning else "",
        "spo2": latest_vital["spo2"] if latest_vital and not fresh_questioning else "",
        "temperature": latest_vital["temperature"] if latest_vital and not fresh_questioning else "",
        "heart_rate": latest_vital["heart_rate"] if latest_vital and not fresh_questioning else "",
    }
    active_session_number = session_number_by_id.get(latest_vital["id"]) if latest_vital else None
    session_input_label = _display_input_type(latest_triage.get("input_type"))

    return render_template(
        "dashboard.html",
        vitals=vitals,
        latest_vital=latest_vital,
        latest_triage=latest_triage,
        latest_history=latest_history,
        model_assessment=model_assessment,
        comparison=comparison,
        reasoning_summary=reasoning_summary,
        intake_defaults=intake_defaults,
        session_number_by_id=session_number_by_id,
        active_session_number=active_session_number,
        session_input_label=session_input_label,
        fresh_questioning=fresh_questioning,
        selected_vital_id=latest_vital["id"] if latest_vital else None,
        patient_id=patient_id,
    )


@routes_bp.route("/triage/start", methods=["POST"])
def start_triage():
    if "patient_id" not in session:
        return redirect(url_for("routes.login"))

    patient_id = session["patient_id"]
    payload = _parse_patient_form(request.form)
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
    return _dashboard_redirect(vital_id)


@routes_bp.route("/triage/<int:vital_id>/answer", methods=["POST"])
def submit_triage_answer(vital_id):
    if "patient_id" not in session:
        return redirect(url_for("routes.login"))

    answer = request.form.get("answer", "").strip()
    if not answer:
        return _dashboard_redirect(vital_id)

    conn = get_db_connection()
    vital = conn.execute(
        "SELECT * FROM vitals WHERE id = ? AND patient_id = ?",
        (vital_id, session["patient_id"]),
    ).fetchone()
    if vital is None:
        conn.close()
        return _dashboard_redirect(vital_id)

    current_triage = load_json(vital["ai_recommendation"], {})
    if current_triage.get("type") != "question":
        conn.close()
        return _dashboard_redirect(vital_id)

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
    return _dashboard_redirect(vital_id)


@routes_bp.route("/logout")
def logout():
    session.pop("patient_id", None)
    return redirect(url_for("routes.login"))
