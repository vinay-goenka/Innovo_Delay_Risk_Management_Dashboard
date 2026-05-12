## Innovo Group's Delay Risk Dashboard 

An operational analytics dashboard that predicts delay risk for construction projects, explains the drivers behind each prediction, and simulates how risk would change under different management interventions.

Built during my internship at **Innovo Group, Dubai** as a prototype for proactive project oversight.

---

## What It Does

The dashboard takes live project signals (schedule progress, labor levels, material delays, inspection outcomes, and budget pressure) and produces:

- **A delay-risk score** (0–100%) classifying the project as Low, Medium, or High risk
- **A business impact estimate** showing predicted delay days and total cost exposure in dollars
- **Driver explanations** identifying which factors are pushing this specific project's risk up or down
- **Scenario simulations** showing how risk would drop if management addresses the top issues

Each prediction is project-specific, the dashboard reads the actual operational signals and calculates the insights, rather than applying generic rules or hardcoded weights.

---

## How It Works

The system combines three machine learning models with explainable AI:

- Random Forest **Classifier** -> Predicts the probability of delay
- Random Forest **Regressor** (×2) -> Predicts delay duration and cost exposure
- **SHAP** -> Explains which factors drive each prediction
- **Rule-based scenario engine** -> Uses SHAP to prioritize which drivers to fix first

The classifier outputs a risk percentage. SHAP then breaks that prediction down into per-feature contributions, showing whether each driver is currently increasing or decreasing risk. The scenario simulator uses those rankings to give progressive interventions (fix the top driver, fix the top two, fix the top three) and re-runs the classifier to estimate the resulting risk reduction.

---

## Tech Stack

- **Python** — core application logic
- **Streamlit** — interactive dashboard frontend
- **scikit-learn** — Random Forest classifier and regressor models
- **SHAP** — explainable AI for driver attribution
- **pandas / numpy** — data handling and feature engineering

---

## Project Structure

```
.
├── app.py                                      # Streamlit dashboard
├── generate_outcomes.py                        # Synthetic outcome data generator
├── construction_projects.csv                   # Source project data
├── construction_projects_with_outcomes.csv     # Training data (with synthetic outcomes)
└── assets/
    └── innovo-logo-bgremoved.png
```

---

## Running Locally

```bash
# Install dependencies
pip install streamlit pandas scikit-learn shap numpy

# Launch the dashboard
streamlit run app.py
```

The dashboard opens in your browser at `http://localhost:8501`.

---

## Note on Data

Training data uses real operational variables paired with synthetic outcome labels (`actual_delay_days`, `actual_cost_exposure`) generated for prototype purposes. The synthetic outcomes follow construction industry intuition about which signals drive delays. The architecture is designed to retrain on real historical project outcomes when available.

---

## About

Developed by **Vinay Goenka** during an internship at Innovo Group Dubai. The project combines machine learning, explainable AI, and operational analytics into a single tool for construction project oversight.
