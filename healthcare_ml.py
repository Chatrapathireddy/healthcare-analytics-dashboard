import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import shap
import json
import warnings
import os

warnings.filterwarnings("ignore")

# ---------------- DATABASE (SAFE) ----------------
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
    "password": os.getenv("DB_PASSWORD")  # ✅ safe
}

engine = create_engine(
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# ---------------- EXTRACT ----------------
def extract_data():
    query = """
        SELECT
            p.patient_id,
            EXTRACT(YEAR FROM AGE(p.date_of_birth)) AS age,
            p.gender,
            p.blood_type,

            a.admission_id,
            a.department,
            a.admission_type,
            a.discharge_status,
            EXTRACT(DAY FROM (a.discharge_date - a.admission_date)) AS los_days,

            v.heart_rate,
            v.systolic_bp,
            v.diastolic_bp,
            v.temperature,
            v.spo2,
            v.respiratory_rate,

            COUNT(DISTINCT d.diagnosis_id) AS diagnosis_count,
            COUNT(DISTINCT m.medication_id) AS medication_count,

            MAX(CASE WHEN l.test_name = 'Creatinine' THEN l.test_value END) AS creatinine,
            MAX(CASE WHEN l.test_name = 'HbA1c' THEN l.test_value END) AS hba1c,
            MAX(CASE WHEN l.test_name = 'WBC' THEN l.test_value END) AS wbc,
            MAX(CASE WHEN l.test_name = 'Hemoglobin' THEN l.test_value END) AS hemoglobin,
            MAX(CASE WHEN l.test_name = 'Sodium' THEN l.test_value END) AS sodium,

            COUNT(l.test_name) AS abnormal_labs,

            CASE WHEN EXISTS (
                SELECT 1 FROM admissions a2
                WHERE a2.patient_id = p.patient_id
                AND a2.admission_date > a.discharge_date
                AND a2.admission_date <= a.discharge_date + INTERVAL '30 days'
            ) THEN 1 ELSE 0 END AS readmitted_30d,

            CASE WHEN a.discharge_status = 'Transferred' THEN 1 ELSE 0 END AS deteriorated,
            CASE WHEN a.discharge_status = 'Deceased' THEN 1 ELSE 0 END AS mortality

        FROM patients p
        JOIN admissions a ON p.patient_id = a.patient_id
        LEFT JOIN vitals v ON p.patient_id = v.patient_id AND v.admission_id = a.admission_id
        LEFT JOIN diagnoses d ON a.admission_id = d.admission_id
        LEFT JOIN medications m ON a.admission_id = m.admission_id
        LEFT JOIN lab_results l ON a.admission_id = l.admission_id
        WHERE a.discharge_date IS NOT NULL

        GROUP BY
            p.patient_id, p.date_of_birth, p.gender, p.blood_type,
            a.admission_id, a.department, a.admission_type,
            a.discharge_date, a.admission_date, a.discharge_status,
            v.heart_rate, v.systolic_bp, v.diastolic_bp,
            v.temperature, v.spo2, v.respiratory_rate
    """

    df = pd.read_sql(query, engine)
    print(f"Extracted {len(df)} records")
    return df

# ---------------- PREPROCESS ----------------
def preprocess(df):
    le = LabelEncoder()

    for col in ["gender", "blood_type", "department", "admission_type"]:
        df[col] = df[col].fillna("Unknown")
        df[col] = le.fit_transform(df[col].astype(str))

    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["shock_index"] = df["heart_rate"] / df["systolic_bp"].replace(0, np.nan)
    df["age_risk"] = (df["age"] > 65).astype(int)

    drop_cols = ["patient_id", "admission_id", "discharge_status",
                 "readmitted_30d", "deteriorated", "mortality"]

    features = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return features, df

# ---------------- MODEL ----------------
def safe_proba(pipe, X):
    p = pipe.predict_proba(X)
    return p[:, 1] if p.shape[1] > 1 else np.zeros(len(X))

def train_model(X, y, name):
    if y.nunique() < 2:
        print(f"\n{name} MODEL — SKIPPED")
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=42))
        ])
        X_dummy = pd.concat([X, X.iloc[:1]])
        y_dummy = pd.concat([y, pd.Series([1])])
        pipe.fit(X_dummy, y_dummy)
        return pipe

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = safe_proba(pipe, X_test)

    print(f"\n{name} MODEL")
    print(classification_report(y_test, y_pred))

    try:
        print("AUC:", roc_auc_score(y_test, y_prob))
    except:
        print("AUC: N/A")

    return pipe

# ---------------- SCORE ----------------
def score_patients(df, features, models):
    records = []

    for idx in range(len(df)):
        X_row = features.iloc[[idx]]

        r_prob = safe_proba(models["readmission"], X_row)[0]
        d_prob = safe_proba(models["deterioration"], X_row)[0]
        m_prob = safe_proba(models["mortality"], X_row)[0]

        def tier(prob):
            if prob >= 0.7: return "High"
            elif prob >= 0.4: return "Medium"
            else: return "Low"

        records.append({
            "patient_id": int(df.iloc[idx]["patient_id"]),
            "admission_id": int(df.iloc[idx]["admission_id"]),
            "model_version": "rf_v1.0",
            "readmission_risk": round(float(r_prob), 4),
            "deterioration_risk": round(float(d_prob), 4),
            "mortality_risk": round(float(m_prob), 4),
            "readmission_tier": tier(r_prob),
            "deterioration_tier": tier(d_prob),
            "mortality_tier": tier(m_prob),
            "top_features": json.dumps([]),
            "ai_explanation": None
        })

    scores_df = pd.DataFrame(records)
    scores_df.to_sql("risk_scores", engine, if_exists="append", index=False, method="multi")

    print(f"\nWrote {len(scores_df)} risk scores")
    return scores_df

# ---------------- MAIN ----------------
def main():
    df = extract_data()
    X, df_clean = preprocess(df)

    models = {
        "readmission": train_model(X, df_clean["readmitted_30d"], "READMISSION"),
        "deterioration": train_model(X, df_clean["deteriorated"], "DETERIORATION"),
        "mortality": train_model(X, df_clean["mortality"], "MORTALITY"),
    }

    score_patients(df_clean, X, models)

    print("\n✅ ML Pipeline Completed")

if __name__ == "__main__":
    main()