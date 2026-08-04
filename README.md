# ❤️ CardioInsight

**An educational cardiovascular risk estimator** — one module of a broader concept
for an AI-powered healthy aging platform, inspired by research into AI-powered
wearable devices for cardiovascular disease detection.

🔗 **Try it live:** _link added here once deployed_

> ⚠️ **This is a prototype for educational purposes only.** It is not a medical
> device and does not diagnose disease. It should never replace advice from a
> qualified healthcare professional — always consult a doctor about your personal
> health.

## What it does

You enter basic health information — age, sex, blood pressure, resting heart rate,
activity level, smoking status, diabetes status, family history, BMI, and
cholesterol — and a machine learning model estimates cardiovascular risk. It also
explains *why*, showing which factors are pushing the estimate up or down, so the
result isn't just a number but something you can actually understand and act on.

## How it works

- **Model:** Logistic Regression, trained on a synthetic dataset built from
  well-established cardiovascular risk-factor relationships (the same factors used
  in tools like the Framingham Risk Score — age, blood pressure, smoking, diabetes,
  family history, etc.)
- **Explainability:** each prediction is broken down by feature contribution, so
  users see which of their inputs is driving their result up or down
- **Interface:** a simple web app (Streamlit) — no installation needed, just open
  the link

## Project files

| File | Purpose |
|---|---|
| `app.py` | The web app — collects inputs and shows the risk estimate |
| `train_model.py` | Trains the model and saves it |
| `cardio_model.joblib` | The trained model + scaler |
| `cardio_synthetic_dataset.csv` | Training data |
| `requirements.txt` | Python dependencies |

## Why this project exists

This started as a passion project connected to EPQ research on the question: *to
what extent can AI-powered wearable devices provide clinically reliable early
detection of cardiovascular disease compared with traditional clinical screening
methods?* CardioInsight is a small, honest prototype exploring that idea in
practice — not a finished clinical product, but a demonstration of how machine
learning and accessible design can support healthier aging.

## Running it yourself

```bash
pip install -r requirements.txt
streamlit run app.py
```

