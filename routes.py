from flask import Blueprint, render_template, request, redirect, url_for, session
from models import get_db_connection
from classifier import classify_vitals # <-- Added this import

routes_bp = Blueprint('routes', __name__)

@routes_bp.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['patient_id'] = request.form['patient_id']
        return redirect(url_for('routes.dashboard'))
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

    latest_vital = vitals[0] if vitals else None
    return render_template('dashboard.html', vitals=vitals, latest_vital=latest_vital, patient_id=patient_id)

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

    # Classify the vitals using Gemini
    classification, ai_recommendation = classify_vitals(spo2, temperature, heart_rate)

    # Save to database (Now including the AI recommendation)
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO vitals (patient_id, spo2, temperature, heart_rate, classification, ai_recommendation) VALUES (?, ?, ?, ?, ?, ?)',
        (patient_id, spo2, temperature, heart_rate, classification, ai_recommendation)
    )
    conn.commit()
    conn.close()
    
    # Refresh the dashboard
    return redirect(url_for('routes.dashboard'))

@routes_bp.route('/logout')
def logout():
    session.pop('patient_id', None)
    return redirect(url_for('routes.login'))