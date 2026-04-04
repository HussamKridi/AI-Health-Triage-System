from flask import Blueprint, redirect, render_template, request, session, url_for

from models import get_db_connection
from triage_flow import continue_triage, load_json, run_initial_triage


routes_bp = Blueprint("routes", __name__)


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
    active_vital = None
    if vitals:
        if selected_vital_id is not None:
            active_vital = next((vital for vital in vitals if vital["id"] == selected_vital_id), None)
        if active_vital is None:
            active_vital = vitals[0]

    latest_vital = active_vital
    latest_triage = load_json(latest_vital["ai_recommendation"], {}) if latest_vital else {}
    latest_history = load_json(latest_vital["conversation_history"], []) if latest_vital else []
    model_assessment = load_json(latest_vital["model_assessment"], {}) if latest_vital else {}

    return render_template(
        "dashboard.html",
        vitals=vitals,
        latest_vital=latest_vital,
        latest_triage=latest_triage,
        latest_history=latest_history,
        model_assessment=model_assessment,
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
