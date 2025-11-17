"""
MLflow-ready training script for credit card fraud detection (RandomForest)
Features:
- Sets tracking URI and experiment
- Train/val/test split (stratified)
- StandardScaler, SMOTE handling
- Logs params, metrics (ROC-AUC, PR-AUC), artifacts (model, scaler, SMOTE by joblib)
- Logs model with signature and input_example using mlflow.models.signature.infer_signature
- Saves ROC and PR curves as artifacts
- Provides run_experiment() and run_grid() helpers

Usage:
    python mlflow_fraud_experiment.py

Customize at top (DATAFRAME `df`) or import this file as a module and call run_grid / run_experiment.
"""

import os
import tempfile
from typing import Dict, Any, Iterable

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# -------------------------------
# CONFIG
# -------------------------------
# If running this as a standalone script, set df beforehand, e.g. load from CSV:
# import pandas as pd
# df = pd.read_csv("creditcard.csv")

# Tracking & experiment settings
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "credit_fraud_rf")
RANDOM_STATE = 42

# Columns to scale (adjust to your dataset)
COLS_SCALE = ["time", "amount"]
TARGET_COL = "class"

# -------------------------------
# Helper functions
# -------------------------------

def prepare_data(df):
    """Train/val/test split and scaling. Returns dict of X_train, X_val, X_test, y_* arrays, scaler object, and SMOTE-resampled training set."""
    # split
    full_train, X_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET_COL])
    X_train, X_val = train_test_split(full_train, test_size=0.25, random_state=RANDOM_STATE, stratify=full_train[TARGET_COL])

    y_train = X_train[TARGET_COL].values
    y_val = X_val[TARGET_COL].values
    y_test = X_test[TARGET_COL].values

    # drop target
    X_train = X_train.drop(columns=[TARGET_COL])
    X_val = X_val.drop(columns=[TARGET_COL])
    X_test = X_test.drop(columns=[TARGET_COL])

    # scale
    scaler = StandardScaler()
    # fit using training data only
    X_train.loc[:, COLS_SCALE] = scaler.fit_transform(X_train[COLS_SCALE])
    X_val.loc[:, COLS_SCALE] = scaler.transform(X_val[COLS_SCALE])
    X_test.loc[:, COLS_SCALE] = scaler.transform(X_test[COLS_SCALE])

    # smote on training set (optional)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "smote": sm,
        "X_train_sm": X_train_sm,
        "y_train_sm": y_train_sm,
    }


def _plot_and_log_curves(y_true, y_score, run_artifact_path="."):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_path = os.path.join(run_artifact_path, "roc_curve.png")
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.savefig(roc_path)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_path = os.path.join(run_artifact_path, "pr_curve.png")
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.savefig(pr_path)
    plt.close()

    return roc_path, pr_path


# -------------------------------
# Core experiment functions
# -------------------------------

def run_experiment(df,
                   params: Dict[str, Any] = None,
                   log_artifacts: bool = True,
                   use_smote: bool = True):
    """Run a single experiment and log to MLflow.

    Args:
        df: pandas DataFrame with feature columns + TARGET_COL.
        params: RandomForest params to override defaults.
        log_artifacts: whether to save scaler/SMOTE to artifacts.
        use_smote: whether to train on SMOTE-resampled data.

    Returns:
        dict with metrics and run info
    """
    if params is None:
        params = {}

    # set tracking
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = prepare_data(df)

    X_train_sm = data["X_train_sm"]
    y_train_sm = data["y_train_sm"]
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    scaler = data["scaler"]
    smote = data["smote"]

    # default RF params, can be overridden
    rf_params = {
        "n_estimators": 200,
        "max_depth": None,
        "max_features": "auto",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    rf_params.update(params)

    with mlflow.start_run() as run:
        # log params
        for k, v in rf_params.items():
            mlflow.log_param(k, v)

        model = RandomForestClassifier(**rf_params)

        X_fit = X_train_sm if use_smote else X_train
        y_fit = y_train_sm if use_smote else y_train

        model.fit(X_fit, y_fit)

        y_score = model.predict_proba(X_val)[:, 1]

        auc_roc = roc_auc_score(y_val, y_score)
        prec, rec, _ = precision_recall_curve(y_val, y_score)
        pr_auc = auc(rec, prec)

        mlflow.log_metric("AUC_ROC", float(auc_roc))
        mlflow.log_metric("AUPRC", float(pr_auc))

        # signature + input example
        input_example = X_val.head(1)
        signature = infer_signature(X_val, model.predict_proba(X_val))

        # log model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            signature=signature,
            input_example=input_example,
        )

        # save and log scaler / smote objects
        if log_artifacts:
            with tempfile.TemporaryDirectory() as td:
                scaler_path = os.path.join(td, "scaler.joblib")
                joblib.dump(scaler, scaler_path)
                mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

                # SMOTE object
                smote_path = os.path.join(td, "smote.joblib")
                joblib.dump(smote, smote_path)
                mlflow.log_artifact(smote_path, artifact_path="preprocessing")

        # save curves and log as artifacts
        with tempfile.TemporaryDirectory() as td:
            roc_path, pr_path = _plot_and_log_curves(y_val, y_score, run_artifact_path=td)
            mlflow.log_artifact(roc_path, artifact_path="plots")
            mlflow.log_artifact(pr_path, artifact_path="plots")

        run_id = run.info.run_id

    return {
        "run_id": run_id,
        "auc_roc": float(auc_roc),
        "pr_auc": float(pr_auc),
        "params": rf_params,
    }


def run_grid(df, param_grid: Dict[str, Iterable], max_runs: int = None):
    """Run multiple experiments over a parameter grid and log all runs to MLflow.

    Args:
        df: pandas DataFrame
        param_grid: dict compatible with sklearn.model_selection.ParameterGrid
        max_runs: optional limit on number of runs

    Returns:
        list of run result dicts
    """
    grid = list(ParameterGrid(param_grid))
    results = []
    for i, params in enumerate(grid):
        if max_runs is not None and i >= max_runs:
            break
        print(f"Running grid {i+1}/{len(grid)}: {params}")
        res = run_experiment(df, params=params)
        results.append(res)
    return results


# -------------------------------
# If run as script
# -------------------------------
if __name__ == "__main__":
    import pandas as pd

    # Example: load dataset from CSV in current folder
    # Replace with your path or inject df from external code
    csv_path = "creditcard.csv"
    if not os.path.exists(csv_path):
        raise SystemExit(f"Put your creditcard.csv in the working directory, or set csv_path variable: {csv_path}")

    df = pd.read_csv(csv_path)

    print("MLflow tracking URI:", TRACKING_URI)
    print("Experiment:", EXPERIMENT_NAME)

    # single run example
    result = run_experiment(df)
    print("Done. Run ID:", result["run_id"]) 

    # example grid (uncomment to run)
    # grid = {"n_estimators": [100, 200], "max_depth": [8, 16], "max_features": ["sqrt", "log2"]}
    # run_grid(df, grid)
