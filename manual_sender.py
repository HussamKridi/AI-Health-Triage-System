import requests
import json

API_URL = 'http://127.0.0.1:5000/api/vitals'

def manual_entry():
    print("--- Manual Vitals Entry Tool ---")
    print("Type 'quit' at any prompt to exit.\n")
    
    while True:
        try:
            # 1. Get inputs from the user
            patient_id = input("Patient ID (e.g., PAT-001): ")
            if patient_id.lower() == 'quit': break
            
            spo2_input = input("SpO2 (%) [e.g., 95]: ")
            if spo2_input.lower() == 'quit': break
            spo2 = float(spo2_input)
            
            temp_input = input("Temperature (°C) [e.g., 37.5]: ")
            if temp_input.lower() == 'quit': break
            temperature = float(temp_input)
            
            hr_input = input("Heart Rate (BPM) [e.g., 80]: ")
            if hr_input.lower() == 'quit': break
            heart_rate = int(hr_input)
            
            # 2. Package it into a JSON dictionary
            payload = {
                "patient_id": patient_id,
                "spo2": spo2,
                "temperature": temperature,
                "heart_rate": heart_rate
            }
            
            # 3. Send it to the Flask API
            print("\nSending data to server...")
            response = requests.post(API_URL, json=payload)
            
            # 4. Show the result
            if response.status_code == 201:
                result = response.json()
                print(f"✅ Success! The system classified this as: {result.get('classification')}\n")
                print("-" * 30 + "\n")
            else:
                print(f"❌ Failed. Server responded with status code: {response.status_code}")
                
        except ValueError:
            print("\n⚠️ Invalid input. Please make sure you are typing numbers for SpO2, Temp, and HR.\n")
        except requests.exceptions.ConnectionError:
            print("\n❌ Error: Could not connect to the Flask API. Make sure 'app.py' is running in another terminal!\n")

if __name__ == '__main__':
    manual_entry()