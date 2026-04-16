-- ============================================================
--  HEALTHCARE ANALYTICS & RISK PREDICTION SYSTEM
--  SQL Schema — General Setting | Multi-Risk Prediction
-- ============================================================

-- ────────────────────────────────────────────────────────────
-- 1. PATIENTS
-- ────────────────────────────────────────────────────────────
CREATE TABLE patients (
    patient_id        SERIAL PRIMARY KEY,
    full_name         VARCHAR(150)        NOT NULL,
    date_of_birth     DATE                NOT NULL,
    gender            VARCHAR(10)         CHECK (gender IN ('Male', 'Female', 'Other')),
    blood_type        VARCHAR(5),
    contact_number    VARCHAR(20),
    email             VARCHAR(150),
    address           TEXT,
    emergency_contact VARCHAR(150),
    created_at        TIMESTAMP           DEFAULT NOW()
);

-- ────────────────────────────────────────────────────────────
-- 2. ADMISSIONS
-- ────────────────────────────────────────────────────────────
CREATE TABLE admissions (
    admission_id      SERIAL PRIMARY KEY,
    patient_id        INT                 NOT NULL REFERENCES patients(patient_id),
    admission_date    TIMESTAMP           NOT NULL,
    discharge_date    TIMESTAMP,
    department        VARCHAR(100),        -- e.g. ICU, General Ward, Outpatient
    ward              VARCHAR(50),
    admission_type    VARCHAR(50)         CHECK (admission_type IN ('Emergency', 'Elective', 'Outpatient')),
    discharge_status  VARCHAR(50)         CHECK (discharge_status IN ('Recovered', 'Transferred', 'Deceased', 'LAMA', 'Ongoing')),
    attending_doctor  VARCHAR(150),
    notes             TEXT
);

-- ────────────────────────────────────────────────────────────
-- 3. DIAGNOSES
-- ────────────────────────────────────────────────────────────
CREATE TABLE diagnoses (
    diagnosis_id      SERIAL PRIMARY KEY,
    admission_id      INT                 NOT NULL REFERENCES admissions(admission_id),
    patient_id        INT                 NOT NULL REFERENCES patients(patient_id),
    icd10_code        VARCHAR(10),         -- e.g. I21.0 for STEMI
    diagnosis_name    VARCHAR(200)        NOT NULL,
    diagnosis_type    VARCHAR(20)         CHECK (diagnosis_type IN ('Primary', 'Secondary', 'Comorbidity')),
    diagnosed_at      TIMESTAMP           DEFAULT NOW()
);

-- ────────────────────────────────────────────────────────────
-- 4. VITALS
-- ────────────────────────────────────────────────────────────
CREATE TABLE vitals (
    vital_id          SERIAL PRIMARY KEY,
    patient_id        INT                 NOT NULL REFERENCES patients(patient_id),
    admission_id      INT                 REFERENCES admissions(admission_id),
    recorded_at       TIMESTAMP           NOT NULL DEFAULT NOW(),
    heart_rate        FLOAT,              -- bpm
    systolic_bp       FLOAT,              -- mmHg
    diastolic_bp      FLOAT,              -- mmHg
    temperature       FLOAT,              -- Celsius
    spo2              FLOAT,              -- % oxygen saturation
    respiratory_rate  FLOAT,              -- breaths per minute
    weight_kg         FLOAT,
    height_cm         FLOAT,
    bmi               FLOAT GENERATED ALWAYS AS (
                          CASE WHEN height_cm > 0
                          THEN ROUND((weight_kg / ((height_cm/100)^2))::NUMERIC, 2)
                          ELSE NULL END
                      ) STORED
);

-- ────────────────────────────────────────────────────────────
-- 5. LAB RESULTS
-- ────────────────────────────────────────────────────────────
CREATE TABLE lab_results (
    lab_id            SERIAL PRIMARY KEY,
    patient_id        INT                 NOT NULL REFERENCES patients(patient_id),
    admission_id      INT                 REFERENCES admissions(admission_id),
    test_name         VARCHAR(150)        NOT NULL,  -- e.g. HbA1c, Creatinine, WBC
    test_value        FLOAT               NOT NULL,
    unit              VARCHAR(30),                   -- e.g. mg/dL, mmol/L
    reference_min     FLOAT,
    reference_max     FLOAT,
    is_abnormal       BOOLEAN GENERATED ALWAYS AS (
                          test_value < reference_min OR test_value > reference_max
                      ) STORED,
    collected_at      TIMESTAMP           NOT NULL DEFAULT NOW()
);

-- ────────────────────────────────────────────────────────────
-- 6. MEDICATIONS
-- ────────────────────────────────────────────────────────────
CREATE TABLE medications (
    medication_id     SERIAL PRIMARY KEY,
    patient_id        INT                 NOT NULL REFERENCES patients(patient_id),
    admission_id      INT                 REFERENCES admissions(admission_id),
    drug_name         VARCHAR(150)        NOT NULL,
    dosage            VARCHAR(50),
    frequency         VARCHAR(50),        -- e.g. Once daily, TID
    route             VARCHAR(50),        -- e.g. Oral, IV, IM
    start_date        DATE,
    end_date          DATE,
    prescribed_by     VARCHAR(150)
);

-- ────────────────────────────────────────────────────────────
-- 7. RISK SCORES  (written by Python ML model)
-- ────────────────────────────────────────────────────────────
CREATE TABLE risk_scores (
    score_id              SERIAL PRIMARY KEY,
    patient_id            INT             NOT NULL REFERENCES patients(patient_id),
    admission_id          INT             REFERENCES admissions(admission_id),
    scored_at             TIMESTAMP       DEFAULT NOW(),
    model_version         VARCHAR(20),    -- e.g. rf_v1.2

    -- Risk probabilities (0.0 – 1.0)
    readmission_risk      FLOAT           CHECK (readmission_risk BETWEEN 0 AND 1),
    deterioration_risk    FLOAT           CHECK (deterioration_risk BETWEEN 0 AND 1),
    mortality_risk        FLOAT           CHECK (mortality_risk BETWEEN 0 AND 1),

    -- Risk tiers derived from probabilities
    readmission_tier      VARCHAR(10)     CHECK (readmission_tier IN ('Low', 'Medium', 'High')),
    deterioration_tier    VARCHAR(10)     CHECK (deterioration_tier IN ('Low', 'Medium', 'High')),
    mortality_tier        VARCHAR(10)     CHECK (mortality_tier IN ('Low', 'Medium', 'High')),

    -- Top features driving the prediction (stored as JSON)
    top_features          JSONB,          -- e.g. [{"feature": "creatinine", "shap": 0.32}, ...]
    ai_explanation        TEXT            -- Claude-generated plain-language explanation
);

-- ────────────────────────────────────────────────────────────
-- 8. ALERTS
-- ────────────────────────────────────────────────────────────
CREATE TABLE alerts (
    alert_id          SERIAL PRIMARY KEY,
    patient_id        INT                 NOT NULL REFERENCES patients(patient_id),
    score_id          INT                 REFERENCES risk_scores(score_id),
    alert_type        VARCHAR(50)         CHECK (alert_type IN ('Readmission', 'Deterioration', 'Mortality', 'Abnormal Lab')),
    severity          VARCHAR(10)         CHECK (severity IN ('Low', 'Medium', 'High', 'Critical')),
    message           TEXT,
    is_acknowledged   BOOLEAN             DEFAULT FALSE,
    acknowledged_by   VARCHAR(150),
    created_at        TIMESTAMP           DEFAULT NOW()
);

-- ────────────────────────────────────────────────────────────
-- INDEXES FOR QUERY PERFORMANCE
-- ────────────────────────────────────────────────────────────
CREATE INDEX idx_admissions_patient     ON admissions(patient_id);
CREATE INDEX idx_vitals_patient         ON vitals(patient_id);
CREATE INDEX idx_vitals_recorded        ON vitals(recorded_at);
CREATE INDEX idx_lab_patient            ON lab_results(patient_id);
CREATE INDEX idx_lab_collected          ON lab_results(collected_at);
CREATE INDEX idx_risk_patient           ON risk_scores(patient_id);
CREATE INDEX idx_risk_scored_at         ON risk_scores(scored_at);
CREATE INDEX idx_alerts_patient         ON alerts(patient_id);
CREATE INDEX idx_alerts_severity        ON alerts(severity);

-- ────────────────────────────────────────────────────────────
-- USEFUL VIEWS FOR PYTHON & POWER BI
-- ────────────────────────────────────────────────────────────

-- Latest vitals per patient
CREATE VIEW v_latest_vitals AS
SELECT DISTINCT ON (patient_id) *
FROM vitals
ORDER BY patient_id, recorded_at DESC;

-- Latest risk score per patient
CREATE VIEW v_latest_risk AS
SELECT DISTINCT ON (patient_id) *
FROM risk_scores
ORDER BY patient_id, scored_at DESC;

-- Full patient risk summary (used by Power BI & AI layer)
CREATE VIEW v_patient_risk_summary AS
SELECT
    p.patient_id,
    p.full_name,
    p.date_of_birth,
    p.gender,
    a.admission_id,
    a.department,
    a.admission_type,
    a.discharge_status,
    v.heart_rate,
    v.systolic_bp,
    v.spo2,
    v.bmi,
    r.readmission_risk,
    r.deterioration_risk,
    r.mortality_risk,
    r.readmission_tier,
    r.deterioration_tier,
    r.mortality_tier,
    r.top_features,
    r.ai_explanation,
    r.scored_at
FROM patients p
LEFT JOIN admissions a      ON p.patient_id = a.patient_id
LEFT JOIN v_latest_vitals v ON p.patient_id = v.patient_id
LEFT JOIN v