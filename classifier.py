import os
from dotenv import load_dotenv
from google import genai
import json

# 1. Load the hidden variables from the .env file
load_dotenv()

# 2. Pull the key into a variable
api_key = os.getenv("GEMINI_API_KEY")

# 3. Initialize the client securely
client = genai.Client(api_key=api_key)

def classify_vitals(spo2, temperature, heart_rate):
    prompt = f"""
    You are an expert AI triage assistant. A patient presents with these vitals:
    - Blood Oxygen (SpO2): {spo2}%
    - Body Temperature: {temperature}°C
    - Heart Rate: {heart_rate} BPM

    1. Classify this patient strictly into one of three categories: "Green", "Yellow", or "Red".
    2. Provide a brief 1-2 sentence recommendation on what should be done next.

    You MUST respond ONLY with a valid JSON object in this exact format:
    {{
        "classification": "Status Here",
        "recommendation": "Your advice here."
    }}
    """
    
    try:
        # Generate the response using the exact model name from your list
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        result_text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(result_text)
        
        return data.get("classification", "Yellow"), data.get("recommendation", "Please consult a medical professional.")
        
    except Exception as e:
        print(f"AI Connection Error: {e}")
        return "Yellow", "AI service unavailable. Please seek standard medical evaluation."