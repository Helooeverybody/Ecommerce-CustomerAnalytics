import numpy as np
import pandas as pd

from typing import List, Tuple, Dict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
import logging
import sys
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEGMENT_PARAMS = {'At Risk': {'k': np.float64(1.3),
  'c': np.float64(0.5960799026972933),
  'm': np.float64(0.07820241224389889)},
 'Loyal Customers': {'k': np.float64(0.6131388114528963),
  'c': np.float64(0.0),
  'm': np.float64(0.03688379545844122)},
 'Potential Loyalists': {'k': np.float64(1.1532779416581898),
  'c': np.float64(0.5821057378090623),
  'm': np.float64(0.06937624386565303)}}
DISCOUNT_DOMAIN = np.linspace(0.01, 0.20, 20)

# A function to produce discount-dependent lift ranges
def get_lift_ranges(segment, X):
    p = SEGMENT_PARAMS[segment]

    k = p["k"]
    c = p["c"]
    m = p["m"]

    # purchase lift
    purchase_min = k * (X ** 0.85)
    purchase_max = k * (X ** 1.05)
    
    #churn
    response_center = 1 / (1 + np.exp(-c * (X - m)))

    churn_min = response_center * 0.8
    churn_max = response_center * 1.2

    return (purchase_min, purchase_max), (churn_min, churn_max)


# Modify your policy simulation to accept ranges + discount X
def simulate_discount_policy(df, segment_name, X, n_simulations=1000, seed=42):
    purchase_range, churn_range = get_lift_ranges(segment_name, X)
    purchase_min, purchase_max = purchase_range
    churn_min, churn_max = churn_range
    results = []
    np.random.seed(seed)
    mask = df["business_label"] == segment_name
    for _ in range(n_simulations):
        temp = df.copy()
        purchase_lift = np.random.uniform(purchase_min, purchase_max)
        churn_reduction = np.random.uniform(churn_min, churn_max)
        temp.loc[mask, "new_no_purchase"] = (
            temp.loc[mask, "pred_no_purchase"] * (1 + purchase_lift)
        )
        temp.loc[~mask, "new_no_purchase"] = temp.loc[~mask, "pred_no_purchase"]
        churn_prob = 1 - temp.loc[mask, "survival_prob"]
        new_churn_prob = churn_prob * (1 - churn_reduction)
        temp.loc[mask, "new_survival_prob"] = 1 - new_churn_prob
        temp.loc[~mask, "new_survival_prob"] = temp.loc[~mask, "survival_prob"]
        temp["new_clv"] = (
            temp["new_no_purchase"] *
            temp["pred_money_each_purchase"] *
            temp["new_survival_prob"]
        )
        temp["discount_cost"] = (
            X *
            temp["new_no_purchase"] *
            temp["pred_money_each_purchase"] *
            mask
        )
        temp["net_gain"] = (
            temp["new_clv"]
            - temp["baseline_clv"]
            - temp["discount_cost"]
        )
        results.append(temp["net_gain"].sum())
    return np.array(results)

def run_discount_optimization(df, segment_name):

    results = []

    for X in DISCOUNT_DOMAIN:

        gains = simulate_discount_policy(df, segment_name, X)

        results.append({
            "discount": X,
            "mean_gain": gains.mean(),
            "median_gain": np.median(gains),    
            "prob_positive": np.mean(gains > 0),
            "worst_5pct": np.percentile(gains, 5),
            "best_95pct": np.percentile(gains, 95)
        })

    return pd.DataFrame(results)
