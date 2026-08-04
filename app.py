import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CardioInsight", page_icon="❤️", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("cardio_model.joblib")

bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]

st.title("❤️ CardioInsight")
st.caption("An AI-powered healthy aging project — educational cardiovascular risk estimate")

st.info(
    "**This is an educational prototype, not a medical device.** It does not diagnose "
    "disease and should never replace advice from a qualified healthcare professional. "
    "If you have symptoms or concerns, please consult a clinician."
)

st.subheader("Enter your health information")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 30, 90, 55)
    sex = st.selectbox("Sex", ["Female", "Male"])
    systolic_bp = st.slider("Systolic blood pressure (mmHg)", 90, 210, 125)
    resting_hr = st.slider("Resting heart rate (bpm)", 40, 130, 72)
    activity = st.slider("Moderate activity (minutes/week)", 0, 600, 120)
with col2:
    smoker = st.selectbox("Do you currently smoke?", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes diagnosis?", ["No", "Yes"])
    family_history = st.selectbox("Family history of heart disease?", ["No", "Yes"])
    bmi = st.slider("BMI", 15.0, 50.0, 26.0)
    cholesterol = st.slider("Total cholesterol (mg/dL)", 120, 350, 200)

input_row = pd.DataFrame([{
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "systolic_bp": systolic_bp,
    "resting_hr": resting_hr,
    "activity_min_per_week": activity,
    "smoker": 1 if smoker == "Yes" else 0,
    "diabetes": 1 if diabetes == "Yes" else 0,
    "family_history": 1 if family_history == "Yes" else 0,
    "bmi": bmi,
    "cholesterol": cholesterol,
}])[features]

if st.button("Estimate my risk", type="primary"):
    X_s = scaler.transform(input_row)
    risk = model.predict_proba(X_s)[0, 1]

    st.subheader("Your estimated cardiovascular risk")
    st.metric("Estimated risk", f"{risk*100:.1f}%")
    st.progress(min(risk, 1.0))

    if risk < 0.10:
        st.success("Lower estimated risk based on the factors you entered.")
    elif risk < 0.25:
        st.warning("Moderate estimated risk — consider discussing these factors with a clinician.")
    else:
        st.error("Higher estimated risk — we recommend discussing these factors with a healthcare professional.")

    # Explainability: per-feature contribution = coefficient * standardized value
    contributions = model.coef_[0] * X_s[0]
    contrib_series = pd.Series(contributions, index=features).sort_values(key=abs, ascending=False)

    friendly_names = {
        "age": "Age",
        "sex": "Sex",
        "systolic_bp": "Blood pressure",
        "resting_hr": "Resting heart rate",
        "activity_min_per_week": "Physical activity",
        "smoker": "Smoking status",
        "diabetes": "Diabetes",
        "family_history": "Family history",
        "bmi": "BMI",
        "cholesterol": "Cholesterol",
    }

    st.subheader("What's contributing to this estimate")
    for feat, val in contrib_series.items():
        direction = "increasing" if val > 0 else "reducing"
        st.write(f"- **{friendly_names[feat]}** is {direction} your estimated risk.")

    st.caption(
        "Explanation is based on each factor's contribution to the model's estimate "
        "(a simplified, interpretable feature-importance view)."
    )

    st.markdown("---")
    st.markdown(
        "**Next steps:** Regular exercise, blood pressure management, not smoking, and "
        "regular check-ups are associated with lower cardiovascular risk. Please discuss "
        "your personal risk with a doctor — this tool is for educational purposes only."
    )

st.markdown("---")
st.caption(
    "CardioInsight is part of a broader concept for an AI-powered healthy aging platform "
    "(medication management, appointment reminders, caregiver dashboard, cognitive exercises, "
    "and this research-inspired cardiovascular risk module). Model trained on a synthetic "
    "dataset built from established cardiovascular risk-factor relationships; it is a "
    "prototype and has not been clinically validated."
)
