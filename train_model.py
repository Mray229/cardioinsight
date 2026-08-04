"""
CardioInsight - model training script.

Generates a synthetic but epidemiologically realistic dataset for cardiovascular
risk (relationships based on well-established risk-factor literature, e.g.
Framingham Risk Score direction/magnitude of effects), then trains an
interpretable Logistic Regression model.

This is an MVP/educational prototype, NOT a clinically validated tool.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

rng = np.random.default_rng(42)
N = 8000

age = rng.normal(55, 14, N).clip(30, 90)
sex = rng.integers(0, 2, N)  # 1 = male, 0 = female
sbp = rng.normal(128, 18, N).clip(90, 210)  # systolic blood pressure
resting_hr = rng.normal(72, 11, N).clip(40, 130)
activity_min = rng.gamma(2.0, 60, N).clip(0, 600)  # weekly moderate activity minutes
smoker = rng.binomial(1, 0.18, N)
diabetes = rng.binomial(1, 0.12, N)
family_history = rng.binomial(1, 0.25, N)
bmi = rng.normal(27, 5, N).clip(15, 50)
cholesterol = rng.normal(200, 35, N).clip(120, 350)

# Latent risk score built from known-direction effects (log-odds scale, loosely
# calibrated so overall prevalence and factor effect sizes are plausible).
z = (
    -3.0
    + 0.055 * (age - 50)
    + 0.55 * sex
    + 0.022 * (sbp - 120)
    + 0.010 * (resting_hr - 70)
    - 0.006 * activity_min
    + 0.70 * smoker
    + 0.75 * diabetes
    + 0.55 * family_history
    + 0.035 * (bmi - 25)
    + 0.006 * (cholesterol - 200)
)
prob = 1 / (1 + np.exp(-z))
label = rng.binomial(1, prob)

df = pd.DataFrame({
    "age": age.round(0),
    "sex": sex,
    "systolic_bp": sbp.round(0),
    "resting_hr": resting_hr.round(0),
    "activity_min_per_week": activity_min.round(0),
    "smoker": smoker,
    "diabetes": diabetes,
    "family_history": family_history,
    "bmi": bmi.round(1),
    "cholesterol": cholesterol.round(0),
    "cvd_risk_label": label,
})

df.to_csv("cardio_synthetic_dataset.csv", index=False)

feature_cols = [
    "age", "sex", "systolic_bp", "resting_hr", "activity_min_per_week",
    "smoker", "diabetes", "family_history", "bmi", "cholesterol",
]
X = df[feature_cols]
y = df["cvd_risk_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

preds = model.predict(X_test_s)
proba = model.predict_proba(X_test_s)[:, 1]
print("Accuracy:", accuracy_score(y_test, preds))
print("ROC AUC:", roc_auc_score(y_test, proba))

coef = pd.Series(model.coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)
print("\nFeature coefficients (standardized):")
print(coef)

joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, "cardio_model.joblib")
print("\nSaved model to cardio_model.joblib")
