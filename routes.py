import json

from flask import Blueprint, render_template, request, redirect, url_for, session
from models import get_db_connection
from classifier import classify_vitals # <-- Added this import

routes_bp = Blueprint('routes', __name__)


def _load_json(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _dashboard_redirect(triage_id=None):
    target = url_for('routes.dashboard', triage_id=triage_id) if triage_id else url_for('routes.dashboard')
    return redirect(f"{target}#active-triage")

@routes_bp.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['patient_id'] = request.form['patient_id']
        return _dashboard_redirect()
    return render_template('login.html')

@routes_bp.route('/dashboard')
def dashboard():
    if 'patient_id' not in session:
        return redirect(url_for('routes.login'))

    patient_id = session['patient_id']
    conn = get_db_connection()
    vitals = conn.execute(
        'SELECT * FROM vitals WHERE patient_id = ? ORDER BY timestamp DESC',
        (patient_id,)
    ).fetchall()
    conn.close()

    selected_vital_id = request.args.get('triage_id', type=int)
    active_vital = None

    if vitals:
        if selected_vital_id is not None:
            active_vital = next((vital for vital in vitals if vital["id"] == selected_vital_id), None)
        if active_vital is None:
            active_vital = vitals[0]

    latest_vital = active_vital
    latest_triage = _load_json(latest_vital["ai_recommendation"], {}) if latest_vital else {}
    latest_history = _load_json(latest_vital["conversation_history"], []) if latest_vital else []

    return render_template(
        'dashboard.html',
        vitals=vitals,
        latest_vital=latest_vital,
        latest_triage=latest_triage,
        latest_history=latest_history,
        selected_vital_id=latest_vital["id"] if latest_vital else None,
        patient_id=patient_id,
    )

# --- NEW ROUTE FOR MANUAL WEB ENTRY ---
@routes_bp.route('/add_vitals', methods=['POST'])
def add_vitals():
    if 'patient_id' not in session:
        return redirect(url_for('routes.login'))

    patient_id = session['patient_id']
    
    # Grab the numbers from the web form
    spo2 = float(request.form['spo2'])
    temperature = float(request.form['temperature'])
    heart_rate = int(request.form['heart_rate'])

    # Run the hybrid classifier and store the structured output.
    result = classify_vitals(spo2, temperature, heart_rate)
    classification = result["triage_response"].get("risk_label", result["risk_level"])
    ai_recommendation = json.dumps(result["triage_response"])
    conversation_history = json.dumps([])

    # Save to database with the current triage state.
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO vitals (patient_id, spo2, temperature, heart_rate, classification, ai_recommendation, conversation_history) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (patient_id, spo2, temperature, heart_rate, classification, ai_recommendation, conversation_history)
    )
    vital_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    conn.commit()
    conn.close()
    
    # Refresh the dashboard
    return _dashboard_redirect(vital_id)


@routes_bp.route('/triage/<int:vital_id>/answer', methods=['POST'])
def submit_triage_answer(vital_id):
    if 'patient_id' not in session:
        return redirect(url_for('routes.login'))

    answer = request.form.get('answer', '').strip()
    if not answer:
        return _dashboard_redirect(vital_id)

    conn = get_db_connection()
    vital = conn.execute(
        'SELECT * FROM vitals WHERE id = ? AND patient_id = ?',
        (vital_id, session['patient_id'])
    ).fetchone()

    if vital is None:
        conn.close()
        return _dashboard_redirect(vital_id)

    current_triage = _load_json(vital["ai_recommendation"], {})
    conversation_history = _load_json(vital["conversation_history"], [])

    if current_triage.get("type") != "question":
        conn.close()
        return _dashboard_redirect(vital_id)

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

    return _dashboard_redirect(vital_id)

@routes_bp.route('/logout')
def logout():
    session.pop('patient_id', None)
    return redirect(url_for('routes.login'))
