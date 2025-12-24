import os 
import gc 
import numpy as np 
import pandas as pd 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
import lightgbm as lgb
import shap
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.inspection import permutation_importance
from sksurv.util import Surv
from pathlib import Path
import logging
import sys

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define path
DATA_DIR = PROJECT_ROOT / "data" / "processed"
def compute_time_to_churn(
    df_tx,
    churn_window_days: int,
    data_end: pd.Timestamp
):
    churn_window = pd.Timedelta(days=churn_window_days)
    results = []
    df_tx = df_tx.sort_values(['visitorid', 'timestamp'])
    for vid, grp in df_tx.groupby('visitorid'):
        times = grp['timestamp']
        first_tx = times.iloc[0]
        last_tx  = times.iloc[-1]
        inactivity = data_end - last_tx
        if inactivity >= churn_window:
            churned = 1
            churn_time = last_tx 
        else:
            churned = 0
            churn_time = data_end

        time_to_event = (churn_time - first_tx).total_seconds() / (3600 * 24)

        results.append({
            'visitorid': vid,
            'origin': first_tx,
            'time_to_event': time_to_event,
            'churned': churned
        })

    return pd.DataFrame(results)
