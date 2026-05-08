"""
Generates synthetic actual_delay_days and actual_cost_exposure columns
for construction project outcomes.

Reads:  construction_projects.csv (untouched)
Writes: construction_projects_with_outcomes.csv (new file)

Run once: python generate_outcomes.py
"""

import pandas as pd
import numpy as np

np.random.seed(42)  # reproducibility

INPUT_FILE = "construction_projects.csv"
OUTPUT_FILE = "construction_projects_with_outcomes.csv"

df = pd.read_csv(INPUT_FILE)

# Recompute derived features
schedule_gap = df["planned_progress"] - df["actual_progress"]
labor_shortage_pct = ((df["labor_planned"] - df["labor_actual"]) / df["labor_planned"]) * 100
cost_pressure = df["budget_used"] - df["actual_progress"]

# --- DELAY DAYS ---
# Weighted combo of risk drivers + Gaussian noise
base_delay = (
    0.6 * schedule_gap.clip(lower=0)
    + 0.7 * df["material_delay_days"]
    + 1.5 * df["inspection_failures"]
    + 0.2 * labor_shortage_pct.clip(lower=0)
    + 0.3 * cost_pressure.clip(lower=0)
)

noise = np.random.normal(0, 3, size=len(df))
actual_delay_days = (base_delay + noise).clip(lower=0)

# Non-delayed projects: small noise (0-3 days)
actual_delay_days = np.where(
    df["delayed"] == 1,
    actual_delay_days,
    np.random.uniform(0, 3, size=len(df))
)

actual_delay_days = np.round(actual_delay_days).astype(int)

# --- COST EXPOSURE ---
# Daily cost scales with project size (labor_planned as proxy)
# Small projects ~$5K/day, mega-projects ~$150K/day
labor_min = df["labor_planned"].min()
labor_max = df["labor_planned"].max()
labor_normalized = (df["labor_planned"] - labor_min) / (labor_max - labor_min + 1e-9)

daily_cost_base = 5000 + (labor_normalized * 145000)
daily_cost_noise = np.random.normal(0, 8000, size=len(df))
daily_cost = (daily_cost_base + daily_cost_noise).clip(lower=3000)

actual_cost_exposure = (actual_delay_days * daily_cost).round(0).astype(int)

# --- WRITE TO NEW FILE ---
df["actual_delay_days"] = actual_delay_days
df["actual_cost_exposure"] = actual_cost_exposure

df.to_csv(OUTPUT_FILE, index=False)

# Sanity check
print(f"Read {len(df)} projects from {INPUT_FILE}")
print(f"Wrote outcomes to {OUTPUT_FILE}")
print(f"\nDelay days summary:")
print(df["actual_delay_days"].describe().round(1))
print(f"\nCost exposure summary:")
print(df["actual_cost_exposure"].describe().round(0))
print(f"\nDelayed vs non-delayed averages:")
print(df.groupby("delayed")[["actual_delay_days", "actual_cost_exposure"]].mean().round(0))