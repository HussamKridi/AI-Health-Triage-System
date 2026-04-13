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
2. If your Windows username contains non-ASCII characters and `python app.py` fails from a virtualenv, create an ASCII alias first:
   `New-Item -ItemType Junction -Path .python-home -Target "$env:LOCALAPPDATA\Programs\Python\Python310"`
3. Create a virtual environment:
   `.\.python-home\python.exe -m venv .venv`
   If you do not hit the Unicode path issue, `python -m venv .venv` also works.
4. Install dependencies with the venv interpreter:
   `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
   If PowerShell allows script activation on your machine, `.\.venv\Scripts\Activate.ps1` is optional.
5. Add your key to `.env`:
   `GEMINI_API_KEY=your_actual_api_key_here`
6. Train the local model:
   `.\.venv\Scripts\python.exe train_model.py`
7. Start the app:
   `.\.venv\Scripts\python.exe app.py`
   On this machine, plain `python app.py` also works without activating the venv.

## Run Tests

Run the automated tests with:

`.\.venv\Scripts\python.exe -m unittest discover -s tests`
