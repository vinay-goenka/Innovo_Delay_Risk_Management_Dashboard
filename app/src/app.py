import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Construction Risk Predictor", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    max-width: 1300px;
}
div[data-testid="stImage"] {
    margin-bottom: -1.5rem;
}
div[data-testid="stImage"] img {
    margin: 0;
}
h1 {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
div[data-testid="stNumberInput"] {
    max-width: 180px;
}
div[data-testid="stNumberInput"] input {
    font-size: 16px;
    padding: 6px 8px;
}
.metric-card {
    border: 1px solid rgba(96, 165, 250, 0.22);
    padding: 28px;
    border-radius: 18px;
    background: radial-gradient(circle at top right, rgba(59, 130, 246, 0.18), transparent 35%), linear-gradient(135deg, #1f2937 0%, #111827 100%);
    color: white;
    min-height: 240px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
}
.big-risk {
    font-size: 58px;
    font-weight: 900;
    margin-top: 10px;
    margin-bottom: 10px;
}
.small-text {
    color: #d1d5db;
    font-size: 15px;
    font-weight: 500;
}
.risk-pill {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 700;
    margin-top: 8px;
    color: white;
}
.risk-pill-high {
    background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
}
.risk-pill-medium {
    background: linear-gradient(135deg, #eab308 0%, #b45309 100%);
}
.risk-pill-low {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
}
.big-risk-high {
    color: #ef4444;
}
.big-risk-medium {
    color: #eab308;
}
.big-risk-low {
    color: #22c55e;
}
.helper-card {
    border: 1px dashed rgba(255, 255, 255, 0.25);
    padding: 24px;
    border-radius: 16px;
    min-height: 220px;
    background: rgba(255, 255, 255, 0.03);
}
.insight-card {
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 10px;
}
.card-title {
    font-weight: 700;
    font-size: 16px;
    margin-bottom: 6px;
}
.card-detail {
    color: #9ca3af;
    font-size: 14px;
}
.section-card {
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 20px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.035);
    margin-bottom: 16px;
}
.scenario-card {
    border: 1px solid rgba(96, 165, 250, 0.16);
    padding: 20px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(31, 41, 55, 0.95), rgba(17, 24, 39, 0.95));
    text-align: center;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
}
.scenario-label {
    color: #9ca3af;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
}
.scenario-value {
    font-size: 38px;
    font-weight: 800;
}
.scenario-note {
    color: #9ca3af;
    font-size: 14px;
    margin-top: 12px;
}
.action-row {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 10px;
    background: rgba(255, 255, 255, 0.035);
}
.action-number {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    background: rgba(59, 130, 246, 0.2);
    color: #93c5fd;
    flex-shrink: 0;
}
.action-text {
    color: #d1d5db;
    font-size: 15px;
    line-height: 1.45;
}
.driver-card {
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 14px 16px;
    border-radius: 14px;
    margin-bottom: 8px;
    background: rgba(255, 255, 255, 0.03);
}
</style>
""", unsafe_allow_html=True)

st.image("assets/innovo-logo-bgremoved.png", width=180)

st.title("Construction Project Risk Predictor")
st.caption("Predict delay risk based on schedule, labor, materials, inspections, and budget pressure.")

@st.cache_data
def load_data():
    return pd.read_csv("construction_projects_with_outcomes.csv")

df = load_data()

df["schedule_gap"] = df["planned_progress"] - df["actual_progress"]
df["labor_shortage_pct"] = ((df["labor_planned"] - df["labor_actual"]) / df["labor_planned"]) * 100
df["cost_pressure"] = df["budget_used"] - df["actual_progress"]

features = [
    "schedule_gap",
    "labor_shortage_pct",
    "material_delay_days",
    "inspection_failures",
    "cost_pressure"
]

feature_labels = {
    "schedule_gap": "Schedule Gap",
    "labor_shortage_pct": "Labor Shortage %",
    "material_delay_days": "Material Delay Days",
    "inspection_failures": "Inspection Failures",
    "cost_pressure": "Cost Pressure"
}

from sklearn.ensemble import RandomForestRegressor

X = df[features]
y = df["delayed"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Regressor for delay days
y_delay = df["actual_delay_days"]
delay_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
delay_regressor.fit(X, y_delay)

# Regressor for cost exposure
y_cost = df["actual_cost_exposure"]
cost_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
cost_regressor.fit(X, y_cost)

st.divider()

left, right = st.columns([1, 1.2])

with left:
    st.subheader("Project Inputs")

    c1, c2 = st.columns(2)

    with c1:
        planned_progress = st.number_input("Planned Progress (%)", 0, 100, 55)
        actual_progress = st.number_input("Actual Progress (%)", 0, 100, 43)
        budget_used = st.number_input("Budget Used (%)", 0, 100, 58)
        material_delay_days = st.number_input("Material Delay Days", 0, 365, 7)

    with c2:
        labor_planned = st.number_input("Planned Labor", 1, 1000, 120)
        labor_actual = st.number_input("Actual Labor", 0, 1000, 92)
        inspection_failures = st.number_input("Inspection Failures", 0, 20, 1)

    predict_button = st.button("Predict Delay Risk", use_container_width=True)

    if predict_button:
        st.divider()
        st.subheader("Project Risk Summary")

        l1, l2 = st.columns(2)
        l1.metric("Schedule Gap", f"{planned_progress - actual_progress:.1f}%")
        l2.metric("Labor Shortage", f"{((labor_planned - labor_actual) / labor_planned) * 100:.1f}%")

        l3, l4 = st.columns(2)
        l3.metric("Material Delay", f"{material_delay_days} days")
        l4.metric("Inspection Failures", inspection_failures)

        st.metric("Cost Pressure", f"{budget_used - actual_progress:.1f}%")

schedule_gap = planned_progress - actual_progress
labor_shortage_pct = ((labor_planned - labor_actual) / labor_planned) * 100
cost_pressure = budget_used - actual_progress

new_project = pd.DataFrame([{
    "schedule_gap": schedule_gap,
    "labor_shortage_pct": labor_shortage_pct,
    "material_delay_days": material_delay_days,
    "inspection_failures": inspection_failures,
    "cost_pressure": cost_pressure
}])

risk = model.predict_proba(new_project)[0][1] * 100

estimated_delay_days = max(0, int(round(delay_regressor.predict(new_project)[0])))
estimated_cost_exposure = max(0, int(round(cost_regressor.predict(new_project)[0])))
estimated_daily_cost = estimated_cost_exposure / max(estimated_delay_days, 1)

improved_labor_actual = min(labor_planned, labor_actual + 15)
improved_material_delay_days = max(0, material_delay_days - 3)
improved_inspection_failures = max(0, inspection_failures - 1)
improved_labor_shortage_pct = ((labor_planned - improved_labor_actual) / labor_planned) * 100

scenario_project = pd.DataFrame([{
    "schedule_gap": max(0, schedule_gap - 4),
    "labor_shortage_pct": improved_labor_shortage_pct,
    "material_delay_days": improved_material_delay_days,
    "inspection_failures": improved_inspection_failures,
    "cost_pressure": max(0, cost_pressure - 3)
}])

scenario_risk = model.predict_proba(scenario_project)[0][1] * 100
risk_reduction = risk - scenario_risk

recommended_actions = []

if schedule_gap > 10:
    recommended_actions.append("Recover schedule by adding short-term labor support or extending shifts on critical activities.")

if labor_shortage_pct > 20:
    recommended_actions.append(f"Increase actual labor from {labor_actual} to at least {improved_labor_actual} workers for the next two weeks.")

if material_delay_days > 5:
    recommended_actions.append(f"Escalate procurement and reduce material delay from {material_delay_days} days to {improved_material_delay_days} days through supplier follow-up or backup vendors.")

if inspection_failures > 0:
    recommended_actions.append("Run a quality-control check before the next inspection to reduce rework and approval delays.")

if cost_pressure > 10:
    recommended_actions.append("Review cost categories with the largest overruns and freeze non-critical spending until progress catches up.")

if len(recommended_actions) == 0:
    recommended_actions.append("Continue regular monitoring. No major intervention is currently required.")

if risk >= 70:
    risk_label = "High Risk"
    risk_message = "This project is likely to be delayed."
    risk_class = "risk-pill-high"
    big_risk_class = "big-risk-high"
elif risk >= 40:
    risk_label = "Medium Risk"
    risk_message = "This project should be monitored closely."
    risk_class = "risk-pill-medium"
    big_risk_class = "big-risk-medium"
else:
    risk_label = "Low Risk"
    risk_message = "This project appears to be on track."
    risk_class = "risk-pill-low"
    big_risk_class = "big-risk-low"

with right:
    st.subheader("Prediction")

    if predict_button:
        st.markdown(f"""
        <div class="metric-card">
            <div class="small-text">Predicted Delay Risk</div>
            <div class="big-risk {big_risk_class}">{risk:.1f}%</div>
            <div class="risk-pill {risk_class}">{risk_label}</div>
            <div style="margin-top: 18px; padding: 12px 14px; border-radius: 12px; background: rgba(255, 255, 255, 0.06); font-size: 16px;">{risk_message}</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.subheader("Business Impact Estimate")

        b1, b2, b3 = st.columns(3)
        b1.metric("Estimated Delay", f"{estimated_delay_days} days")
        b2.metric("Predicted Daily Cost", f"${estimated_daily_cost:,.0f}")
        b3.metric("Estimated Cost Exposure", f"${estimated_cost_exposure:,.0f}")

        st.caption("Delay days and cost exposure are predicted by a Random Forest regressor trained on historical project outcomes. Daily cost is derived from predicted cost ÷ predicted days.")

        st.divider()

        try:
            import shap
            SHAP_AVAILABLE = True
        except ImportError:
            SHAP_AVAILABLE = False

        st.subheader("Risk Driver Contributions")

        shap_df = pd.DataFrame()

        if SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(new_project)

            if isinstance(shap_values, list):
                project_shap_values = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                project_shap_values = shap_values[0, :, 1]
            else:
                project_shap_values = shap_values[0]

            total_abs_shap = abs(project_shap_values).sum()

            if total_abs_shap > 0:
                contribution_pct = (100 * abs(project_shap_values) / total_abs_shap).round(1)
            else:
                contribution_pct = [0 for _ in project_shap_values]

            shap_df = pd.DataFrame({
                "Risk Driver": [feature_labels[feature] for feature in features],
                "Contribution %": contribution_pct,
                "Effect on Risk": ["Increases" if value > 0 else "Decreases" for value in project_shap_values]
            }).sort_values(by="Contribution %", ascending=False)

        fallback_importance_df = pd.DataFrame({
            "Risk Driver": [feature_labels[feature] for feature in features],
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        if SHAP_AVAILABLE and not shap_df.empty:
            for _, row in shap_df.iterrows():
                st.markdown(f"""
                <div class="driver-card">
                    <b>{row['Risk Driver']}</b> — {row['Contribution %']}% contribution — {row['Effect on Risk']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Showing model-level feature importance instead of project-specific SHAP values.")
            fallback_importance_df["Importance"] = (fallback_importance_df["Importance"] * 100).round(1)

            for _, row in fallback_importance_df.iterrows():
                st.markdown(f"""
                <div class="driver-card">
                    <b>{row['Risk Driver']}</b> — {row['Importance']}% model importance
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="helper-card">
            <h3>Ready to predict</h3>
            <p>Enter project details on the left, then click <b>Predict Delay Risk</b>.</p>
            <p>The model will return a delay-risk percentage and show the main warning signs.</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

if not predict_button:
    st.stop()

st.subheader("Recommended Intervention")

st.markdown("""
<div class="section-card">
    <div class="card-title">Priority Actions</div>
    <div class="card-detail">Recommended actions are generated from the project’s strongest operational risk signals.</div>
</div>
""", unsafe_allow_html=True)

for index, action in enumerate(recommended_actions, start=1):
    st.markdown(f"""
    <div class="action-row">
        <div class="action-number">{index}</div>
        <div class="action-text">{action}</div>
    </div>
    """, unsafe_allow_html=True)

st.subheader("Scenario Simulation")

st.markdown("""
<div class="section-card">
    <div class="card-title">What-if management takes corrective action?</div>
    <div class="card-detail">The simulation estimates how the delay-risk score could change after operational improvements.</div>
</div>
""", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(f"""
    <div class="scenario-card">
        <div class="scenario-label">Current Risk</div>
        <div class="scenario-value">{risk:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown(f"""
    <div class="scenario-card">
        <div class="scenario-label">Simulated Risk</div>
        <div class="scenario-value">{scenario_risk:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with s3:
    st.markdown(f"""
    <div class="scenario-card">
        <div class="scenario-label">Potential Reduction</div>
        <div class="scenario-value">{risk_reduction:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="section-card">
    <div class="card-title">Simulation Assumptions</div>
    <div class="card-detail">
        Labor increases to {improved_labor_actual}, material delay falls to {improved_material_delay_days} days, inspection failures fall to {improved_inspection_failures}, schedule gap improves by 4%, and cost pressure improves by 3%.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

st.subheader("Suggested Action Plan")

actions = []

if schedule_gap > 10:
    actions.append("Project is significantly behind schedule. Increase workforce, extend work shifts, or revise milestone planning.")

if labor_shortage_pct > 20:
    actions.append("Labor shortage is high. Reallocate workers from lower-risk projects or bring in subcontractor support.")

if material_delay_days > 5:
    actions.append("Material delay is a major issue. Escalate supplier communication and consider backup vendors.")

if inspection_failures > 0:
    actions.append("Inspection failures are slowing progress. Review quality control before the next inspection.")

if cost_pressure > 10:
    actions.append("Budget pressure is high. Review spending categories and check for inefficiencies.")

if len(actions) == 0:
    actions.append("No major warning signs. Continue regular monitoring.")

for action in actions:
    st.write("•", action)

st.divider()

with st.expander("View Training Data"):
    st.dataframe(df, use_container_width=True)
