# AI Health Triage System 🩺🤖

A real-time medical triage dashboard that uses AI to analyze patient vitals (SpO2, Temperature, Heart Rate) and output an instant priority classification.

## ✨ Features
* **AI Diagnostics:** Powered by Gemini 2.5 Flash for rapid triage decisions.
* **Bilingual UI:** Instantly switches between English and Turkish.
* **Historical Tracking:** Saves patient data to a local SQLite database.
* **Secure Architecture:** Uses environment variables to protect API credentials.

## 🚀 How to Run Locally

1. **Clone the repository**
2. **Create a virtual environment:** `python -m venv .venv`
3. **Activate the environment:** `.\.venv\Scripts\activate` (Windows)
4. **Install dependencies:** `pip install -r requirements.txt`
5. **Set up credentials:** Create a `.env` file in the root directory and add:
   `GEMINI_API_KEY=your_actual_api_key_here`
6. **Run the app:** `python app.py`