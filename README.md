# AI Health Triage System

A Flask-based medical triage dashboard that combines a local scikit-learn classifier with Gemini to analyze patient vitals, ask one follow-up question at a time, and stop with a final low-risk or high-risk decision.

## Features
- **Hybrid AI flow:** A local Random Forest model determines the baseline risk level from SpO2, temperature, and heart rate.
- **Multi-turn triage:** Gemini 2.5 Flash asks one clinically relevant question at a time and can stop early when the case is clearly low risk or high risk.
- **Strict labels:** Final triage labels are limited to `low risk` and `high risk`, while UI colors are mapped separately.
- **Historical tracking:** Patient readings are stored in a local SQLite database.
- **Credential isolation:** Gemini credentials are loaded from `.env`.

## How to Run Locally

1. Install Python 3.10+.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it on Windows: `.\.venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Add your key to `.env`:
   `GEMINI_API_KEY=your_actual_api_key_here`
6. Train the local model: `python train_model.py`
7. Start the app: `python app.py`

## Run Tests

Run the automated tests with:

`python -m unittest discover -s tests`
