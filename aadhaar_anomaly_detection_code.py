"""
Aadhaar Anomaly Detection - Full ML Pipeline
Generates synthetic dataset + runs Isolation Forest + evaluates results
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Generating Synthetic Aadhaar Transaction Dataset")
print("=" * 60)

N_NORMAL    = 10000
N_ANOMALY   =  1000
N_TOTAL     = N_NORMAL + N_ANOMALY

# ── Normal transactions ──────────────────────
normal = pd.DataFrame({
    'hour_of_day'        : np.random.choice(range(8, 22), N_NORMAL),         # business hours
    'auth_attempts'      : np.random.poisson(1.1, N_NORMAL).clip(1, 3),      # 1–3 tries
    'transaction_amount' : np.random.normal(2500, 800, N_NORMAL).clip(100, 8000),
    'geo_distance_km'    : np.random.exponential(15, N_NORMAL).clip(0, 60),   # local
    'session_duration_s' : np.random.normal(45, 15, N_NORMAL).clip(10, 120),
    'api_calls_per_min'  : np.random.poisson(3, N_NORMAL).clip(1, 10),
    'device_change_flag' : np.random.choice([0, 1], N_NORMAL, p=[0.95, 0.05]),
    'is_anomaly'         : 0
})

# ── Anomalous transactions ───────────────────
anomaly = pd.DataFrame({
    'hour_of_day'        : np.random.choice(range(0, 6), N_ANOMALY),          # odd hours
    'auth_attempts'      : np.random.randint(5, 15, N_ANOMALY),               # many retries
    'transaction_amount' : np.random.normal(18000, 4000, N_ANOMALY).clip(8000, 50000),
    'geo_distance_km'    : np.random.uniform(200, 2000, N_ANOMALY),           # far away
    'session_duration_s' : np.random.choice([2, 3, 4, 5], N_ANOMALY),        # very short
    'api_calls_per_min'  : np.random.randint(50, 200, N_ANOMALY),             # burst
    'device_change_flag' : np.random.choice([0, 1], N_ANOMALY, p=[0.1, 0.9]),
    'is_anomaly'         : 1
})

df = pd.concat([normal, anomaly], ignore_index=True).sample(frac=1, random_state=42)
print(f"  Total records   : {len(df):,}")
print(f"  Normal          : {(df.is_anomaly==0).sum():,}")
print(f"  Anomalous       : {(df.is_anomaly==1).sum():,}")
print(f"  Anomaly ratio   : {(df.is_anomaly==1).mean()*100:.1f}%\n")

features = ['hour_of_day','auth_attempts','transaction_amount',
            'geo_distance_km','session_duration_s',
            'api_calls_per_min','device_change_flag']

X = df[features].values
y_true = df['is_anomaly'].values          # 1 = anomaly, 0 = normal

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# 2. ISOLATION FOREST  (proposed model)
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Training Isolation Forest (Proposed Model)")
print("=" * 60)

iso = IsolationForest(n_estimators=200,
                      contamination=0.09,   # ~9% anomaly rate
                      random_state=42)
iso.fit(X_scaled)

# IsolationForest: -1 = anomaly, +1 = normal → convert to 0/1
y_iso = (iso.predict(X_scaled) == -1).astype(int)

acc_iso  = accuracy_score(y_true, y_iso)
prec_iso = precision_score(y_true, y_iso)
rec_iso  = recall_score(y_true, y_iso)
f1_iso   = f1_score(y_true, y_iso)

print(f"  Accuracy  : {acc_iso*100:.2f}%")
print(f"  Precision : {prec_iso*100:.2f}%")
print(f"  Recall    : {rec_iso*100:.2f}%")
print(f"  F1-Score  : {f1_iso*100:.2f}%\n")

# ─────────────────────────────────────────────
# 3. BASELINE — One-Class SVM
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Baseline — One-Class SVM")
print("=" * 60)

svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.09)
svm.fit(X_scaled)
y_svm = (svm.predict(X_scaled) == -1).astype(int)

acc_svm  = accuracy_score(y_true, y_svm)
prec_svm = precision_score(y_true, y_svm)
rec_svm  = recall_score(y_true, y_svm)
f1_svm   = f1_score(y_true, y_svm)

print(f"  Accuracy  : {acc_svm*100:.2f}%")
print(f"  Precision : {prec_svm*100:.2f}%")
print(f"  Recall    : {rec_svm*100:.2f}%")
print(f"  F1-Score  : {f1_svm*100:.2f}%\n")

# ─────────────────────────────────────────────
# 4. BASELINE — Threshold / Rule-Based
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Baseline — Rule-Based Threshold Detection")
print("=" * 60)

def rule_based(row):
    if (row['hour_of_day'] < 6 or
        row['auth_attempts'] > 4 or
        row['transaction_amount'] > 8000 or
        row['geo_distance_km'] > 150 or
        row['api_calls_per_min'] > 30):
        return 1
    return 0

y_rule = df.apply(rule_based, axis=1).values

acc_rule  = accuracy_score(y_true, y_rule)
prec_rule = precision_score(y_true, y_rule)
rec_rule  = recall_score(y_true, y_rule)
f1_rule   = f1_score(y_true, y_rule)

print(f"  Accuracy  : {acc_rule*100:.2f}%")
print(f"  Precision : {prec_rule*100:.2f}%")
print(f"  Recall    : {rec_rule*100:.2f}%")
print(f"  F1-Score  : {f1_rule*100:.2f}%\n")

# ─────────────────────────────────────────────
# 5. COMPARISON TABLE
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 5: Comparison Table (use directly in paper)")
print("=" * 60)

results = pd.DataFrame({
    'Model': ['Rule-Based (Baseline)', 'One-Class SVM (Baseline)',
              'Isolation Forest (Proposed)'],
    'Accuracy (%)' : [round(acc_rule*100,2), round(acc_svm*100,2), round(acc_iso*100,2)],
    'Precision (%)': [round(prec_rule*100,2), round(prec_svm*100,2), round(prec_iso*100,2)],
    'Recall (%)'   : [round(rec_rule*100,2), round(rec_svm*100,2), round(rec_iso*100,2)],
    'F1-Score (%)'  : [round(f1_rule*100,2), round(f1_svm*100,2), round(f1_iso*100,2)],
})
print(results.to_string(index=False))

# ─────────────────────────────────────────────
# 6. CONFUSION MATRIX for Isolation Forest
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Confusion Matrix — Isolation Forest")
print("=" * 60)
cm = confusion_matrix(y_true, y_iso)
print(f"  True Negative  (correctly flagged normal) : {cm[0][0]:,}")
print(f"  False Positive (normal flagged as anomaly): {cm[0][1]:,}")
print(f"  False Negative (missed anomaly)           : {cm[1][0]:,}")
print(f"  True Positive  (anomaly correctly caught) : {cm[1][1]:,}")

print("\n✅ All results generated successfully.")
