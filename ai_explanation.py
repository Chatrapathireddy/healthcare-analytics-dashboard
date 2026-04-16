import pandas as pd
import google.genai as genai
from sqlalchemy import create_engine, text
import time
import warnings
import os
warnings.filterwarnings("ignore")
 

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
 
DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     5432,
    "database": "postgres",
    "user":     "postgres",
 "password": os.getenv("DB_PASSWORD")   
}
 
engine = create_engine(
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)
 

client = genai.Client(api_key=GEMINI_API_KEY)
 

def fetch_risk_patients():
    query = """
        SELECT
            ROW_NUMBER() OVER () AS score_id,
            r.patient_id,
            r.admission_id,
            r.readmission_risk,
            r.deterioration_risk,
            r.mortality_risk,
            r.readmission_tier,
            r.deterioration_tier,
            r.mortality_tier,
            p.gender,
            EXTRACT(YEAR FROM AGE(p.date_of_birth)) AS age,
            a.department,
            a.admission_type,
            EXTRACT(DAY FROM (a.discharge_date - a.admission_date)) AS los_days,
            v.heart_rate,
            v.systolic_bp,
            v.spo2,
            v.temperature,
            d.diagnosis_name
        FROM risk_scores r
        JOIN patients p       ON r.patient_id = p.patient_id
        JOIN admissions a     ON r.admission_id = a.admission_id
        LEFT JOIN vitals v    ON r.patient_id = v.patient_id AND r.admission_id = v.admission_id
        LEFT JOIN diagnoses d ON r.admission_id = d.admission_id
        WHERE r.ai_explanation IS NULL
        AND (r.readmission_tier IN ('Medium', 'High')
         OR  r.deterioration_tier IN ('Medium', 'High'))
        LIMIT 10
    """
    df = pd.read_sql(query, engine)
    print(f"Found {len(df)} patients needing AI explanation.")
    return df

def build_prompt(row):
    return f"""You are a clinical AI assistant. Write a brief 2-3 sentence plain-language explanation of this patient's health risk for a doctor.
 
Patient Details:
- Age: {int(row.get('age', 0))} years, {row.get('gender', 'Unknown')}
- Diagnosis: {row.get('diagnosis_name', 'Unknown')}
- Department: {row.get('department', 'Unknown')}
- Admission Type: {row.get('admission_type', 'Unknown')}
- Length of Stay: {int(row.get('los_days', 0) or 0)} days
- Heart Rate: {row.get('heart_rate', 'N/A')} bpm
- Systolic BP: {row.get('systolic_bp', 'N/A')} mmHg
- SpO2: {row.get('spo2', 'N/A')}%
- Temperature: {row.get('temperature', 'N/A')}°C
 
Risk Scores:
- Readmission Risk: {row.get('readmission_tier', 'Low')} ({round(float(row.get('readmission_risk', 0)), 2)})
- Deterioration Risk: {row.get('deterioration_tier', 'Low')} ({round(float(row.get('deterioration_risk', 0)), 2)})
- Mortality Risk: {row.get('mortality_tier', 'Low')} ({round(float(row.get('mortality_risk', 0)), 2)})
 
Write a concise clinical summary explaining the key risk factors and what to monitor. Be direct and professional."""
 

def generate_and_save(df):
    success = 0
    failed  = 0
 
    for idx, row in df.iterrows():
        try:
            prompt = build_prompt(row)
 
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            explanation = response.text.strip()
 
            # Update the ai_explanation in SQL
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE risk_scores SET ai_explanation = :exp WHERE patient_id = :pid AND admission_id = :aid"),
                    {"exp": explanation, "pid": int(row['patient_id']), "aid": int(row['admission_id'])}
                )
 
            success += 1
            print(f"  [{success}] Patient {int(row['patient_id'])} — {row['readmission_tier']} readmission risk ✓")
 
           
            time.sleep(10)
 
        except Exception as e:
            failed += 1
            print(f"  [ERROR] Patient {int(row['patient_id'])}: {e}")
            time.sleep(10)
 
    print(f"\n✅ Done! {success} explanations generated, {failed} failed.")
 

def main():
    print("Starting AI Explanation Layer...")
    print("Using: Google Gemini 2.0 Flash (free tier)\n")
 
    df = fetch_risk_patients()
 
    if df.empty:
        print("No patients need explanations — all done already!")
        return
 
    print(f"Generating explanations for {len(df)} patients...\n")
    generate_and_save(df)
 
 
    with engine.connect() as conn:
        sample = pd.read_sql(
            "SELECT patient_id, readmission_tier, deterioration_tier, ai_explanation FROM risk_scores WHERE ai_explanation IS NOT NULL LIMIT 3",
            conn
        )
    print("\n── SAMPLE EXPLANATIONS ──")
    for _, r in sample.iterrows():
        print(f"\nPatient {r['patient_id']} | {r['readmission_tier']} readmission | {r['deterioration_tier']} deterioration")
        print(f"{r['ai_explanation']}")
        print("-" * 60)
 
if __name__ == "__main__":
    main()
