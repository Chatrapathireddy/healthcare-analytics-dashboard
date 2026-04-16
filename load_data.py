

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
import random
import warnings
import os
warnings.filterwarnings("ignore")


DB_CONFIG = {
    "host":      "127.0.0.1",
    "port":     5432,
    "database":  "postgres",
    "user":     "postgres",
    "password": os.getenv("DB_PASSWORD")
}

CSV_PATH = "healthcare_dataset.csv"

engine = create_engine(
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

random.seed(42)
np.random.seed(42)


print("Loengine ading CSV...")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date']    = pd.to_datetime(df['Discharge Date'])
print(f"  {len(df)} rows loaded.")

print("\nInserting patients...")

def estimate_dob(age):
    year = datetime.now().year - int(age)
    return datetime(year, random.randint(1,12), random.randint(1,28)).date()

patients_df = df[['Name','Age','Gender','Blood Type']].drop_duplicates(subset=['Name']).copy()
patients_df = patients_df.reset_index(drop=True)
patients_df['patient_id']        = patients_df.index + 1
patients_df['full_name']         = patients_df['Name'].str.title()
patients_df['date_of_birth']     = patients_df['Age'].apply(estimate_dob)
patients_df['gender']            = patients_df['Gender']
patients_df['blood_type']        = patients_df['Blood Type']
patients_df['contact_number']    = [f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}" for _ in range(len(patients_df))]
patients_df['email']             = patients_df['full_name'].str.replace(' ','').str.lower() + "@email.com"
patients_df['address']           = [f"{random.randint(1,9999)} Main St, City, State" for _ in range(len(patients_df))]
patients_df['emergency_contact'] = "Next of Kin"
patients_df['created_at']        = datetime.now()

patients_insert = patients_df[['patient_id','full_name','date_of_birth','gender',
                                'blood_type','contact_number','email','address',
                                'emergency_contact','created_at']]

patients_insert.to_sql('patients', engine, if_exists='append', index=False, method='multi')
print(f"  {len(patients_insert)} patients inserted.")

# Build name → patient_id map
name_to_id = dict(zip(patients_df['Name'].str.title(), patients_df['patient_id']))


print("\nInserting admissions...")

df['full_name'] = df['Name'].str.title()
df['patient_id'] = df['full_name'].map(name_to_id)

def map_discharge_status(test_result):
    mapping = {
        'Normal':        'Recovered',
        'Abnormal':      'Transferred',
        'Inconclusive':  'Ongoing'
    }
    return mapping.get(test_result, 'Recovered')

admissions_df = df.copy()
admissions_df['admission_id']     = admissions_df.index + 1
admissions_df['admission_date']   = admissions_df['Date of Admission']
admissions_df['discharge_date']   = admissions_df['Discharge Date']
admissions_df['department']       = admissions_df['Medical Condition'].map({
    'Cancer':       'Oncology',
    'Diabetes':     'Endocrinology',
    'Hypertension': 'Cardiology',
    'Obesity':      'General Ward',
    'Arthritis':    'Orthopedics',
    'Asthma':       'Pulmonology'
}).fillna('General Ward')
admissions_df['ward']             = admissions_df['Room Number'].apply(lambda x: f"Ward {chr(65 + (int(x) % 5))}")
admissions_df['admission_type']   = admissions_df['Admission Type'].replace('Urgent', 'Emergency')
admissions_df['discharge_status'] = admissions_df['Test Results'].apply(map_discharge_status)
admissions_df['attending_doctor'] = admissions_df['Doctor']
admissions_df['notes']            = admissions_df['Insurance Provider'] + " | Billing: $" + admissions_df['Billing Amount'].round(2).astype(str)

admissions_insert = admissions_df[['admission_id','patient_id','admission_date','discharge_date',
                                    'department','ward','admission_type','discharge_status',
                                    'attending_doctor','notes']]

admissions_insert.to_sql('admissions', engine, if_exists='append', index=False, method='multi')
print(f"  {len(admissions_insert)} admissions inserted.")


print("\nInserting diagnoses...")

icd10_map = {
    'Cancer':       'C80.1',
    'Diabetes':     'E11.9',
    'Hypertension': 'I10',
    'Obesity':      'E66.9',
    'Arthritis':    'M19.90',
    'Asthma':       'J45.50'
}

diagnoses_df = admissions_df.copy()
diagnoses_df['diagnosis_id']    = diagnoses_df.index + 1
diagnoses_df['icd10_code']      = diagnoses_df['Medical Condition'].map(icd10_map)
diagnoses_df['diagnosis_name']  = diagnoses_df['Medical Condition']
diagnoses_df['diagnosis_type']  = 'Primary'
diagnoses_df['diagnosed_at']    = diagnoses_df['admission_date']

diagnoses_insert = diagnoses_df[['diagnosis_id','admission_id','patient_id',
                                  'icd10_code','diagnosis_name','diagnosis_type','diagnosed_at']]

diagnoses_insert.to_sql('diagnoses', engine, if_exists='append', index=False, method='multi')
print(f"  {len(diagnoses_insert)} diagnoses inserted.")


print("\nInserting vitals...")

def generate_vitals(row):
    condition = row['Medical Condition']
    is_abnormal = row['Test Results'] == 'Abnormal'

    hr  = random.gauss(90, 15) if is_abnormal else random.gauss(75, 10)
    sbp = random.gauss(145, 15) if condition == 'Hypertension' else random.gauss(120, 12)
    dbp = sbp - random.gauss(40, 5)
    spo2 = random.gauss(93, 2) if condition == 'Asthma' and is_abnormal else random.gauss(97, 1.5)
    temp = random.gauss(37.8, 0.5) if is_abnormal else random.gauss(36.8, 0.3)
    rr   = random.gauss(20, 3) if is_abnormal else random.gauss(16, 2)
    wt   = random.gauss(90, 15) if condition == 'Obesity' else random.gauss(72, 12)
    ht   = random.gauss(168, 10)

    return pd.Series({
        'heart_rate':       round(max(40, hr), 1),
        'systolic_bp':      round(max(80, sbp), 1),
        'diastolic_bp':     round(max(50, dbp), 1),
        'temperature':      round(max(35, temp), 1),
        'spo2':             round(min(100, max(85, spo2)), 1),
        'respiratory_rate': round(max(8, rr), 1),
        'weight_kg':        round(max(40, wt), 1),
        'height_cm':        round(max(140, ht), 1)
    })

vitals_gen = admissions_df.apply(generate_vitals, axis=1)
vitals_df  = pd.concat([admissions_df[['patient_id','admission_id','admission_date']].reset_index(drop=True),
                         vitals_gen.reset_index(drop=True)], axis=1)
vitals_df['vital_id']    = vitals_df.index + 1
vitals_df['recorded_at'] = vitals_df['admission_date']

vitals_insert = vitals_df[['vital_id','patient_id','admission_id','recorded_at',
                             'heart_rate','systolic_bp','diastolic_bp','temperature',
                             'spo2','respiratory_rate','weight_kg','height_cm']]

vitals_insert.to_sql('vitals', engine, if_exists='append', index=False, method='multi')
print(f"  {len(vitals_insert)} vitals inserted.")


print("\nInserting lab results...")

lab_templates = {
    'HbA1c':      {'min': 4.0, 'max': 6.4, 'unit': '%',     'conditions': ['Diabetes']},
    'Creatinine': {'min': 0.6, 'max': 1.2, 'unit': 'mg/dL', 'conditions': None},
    'WBC':        {'min': 4.5, 'max': 11.0,'unit': 'K/uL',  'conditions': None},
    'Hemoglobin': {'min': 12.0,'max': 17.5,'unit': 'g/dL',  'conditions': None},
    'Sodium':     {'min': 136, 'max': 145, 'unit': 'mEq/L', 'conditions': None},
}

lab_records = []
lab_id = 1
for _, row in admissions_df.iterrows():
    is_abnormal = row['Test Results'] == 'Abnormal'
    for test_name, ref in lab_templates.items():
        if ref['conditions'] and row['Medical Condition'] not in ref['conditions']:
            continue
        if is_abnormal:
            val = round(random.uniform(ref['max'], ref['max'] * 1.4), 2)
        else:
            val = round(random.uniform(ref['min'], ref['max']), 2)

        lab_records.append({
            'lab_id':        lab_id,
            'patient_id':    row['patient_id'],
            'admission_id':  row['admission_id'],
            'test_name':     test_name,
            'test_value':    val,
            'unit':          ref['unit'],
            'reference_min': ref['min'],
            'reference_max': ref['max'],
            'collected_at':  row['admission_date']
        })
        lab_id += 1

labs_insert = pd.DataFrame(lab_records)
labs_insert.to_sql('lab_results', engine, if_exists='append', index=False, method='multi')
print(f"  {len(labs_insert)} lab results inserted.")


print("\nInserting medications...")

dosage_map = {
    'Lipitor':     '20mg', 'Ibuprofen': '400mg',
    'Aspirin':     '75mg', 'Paracetamol': '500mg',
    'Penicillin':  '500mg'
}
route_map = {
    'Lipitor':     'Oral',  'Ibuprofen': 'Oral',
    'Aspirin':     'Oral',  'Paracetamol': 'Oral',
    'Penicillin':  'IV'
}

meds_df = admissions_df.copy()
meds_df['medication_id']  = meds_df.index + 1
meds_df['drug_name']      = meds_df['Medication']
meds_df['dosage']         = meds_df['Medication'].map(dosage_map).fillna('Standard dose')
meds_df['frequency']      = 'Once daily'
meds_df['route']          = meds_df['Medication'].map(route_map).fillna('Oral')
meds_df['start_date']     = meds_df['admission_date'].dt.date
meds_df['end_date']       = meds_df['discharge_date'].dt.date
meds_df['prescribed_by']  = meds_df['Doctor']

meds_insert = meds_df[['medication_id','patient_id','admission_id','drug_name',
                         'dosage','frequency','route','start_date','end_date','prescribed_by']]

meds_insert.to_sql('medications', engine, if_exists='append', index=False, method='multi')
print(f"  {len(meds_insert)} medications inserted.")


print("\n All data loaded successfully into healthcare_db!")
print("   Tables populated: patients, admissions, diagnoses, vitals, lab_results, medications")
print("   Next step: run healthcare_ml.py to train models and generate risk scores.")
