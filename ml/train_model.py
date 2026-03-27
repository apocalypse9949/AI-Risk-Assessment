"""
train_model.py — Train XGBoost DRI regressor + tier classifier.

Artifacts saved to ml/artifacts/
  dri_model.pkl       XGBoost regressor   (predicts DRI score 0-100)
  tier_model.pkl      XGBoost classifier  (predicts tier index 0-4)
  scaler.pkl          StandardScaler      (fitted on training features)
  feature_names.json  Ordered feature list
  training_report.txt Human-readable evaluation summary
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # headless — no display required
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, accuracy_score,
)
import xgboost as xgb

from config         import DRI_MODEL_PATH, TIER_MODEL_PATH, SCALER_PATH, \
                           FEATURE_NAMES_PATH, RANDOM_SEED, PREMIUM_TIERS
from synthetic_data import build_dataset, dri_to_tier_index
from feature_engine import FEATURE_NAMES

os.makedirs("artifacts", exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════

def load_or_generate_data() -> pd.DataFrame:
    csv_path = "artifacts/synthetic_dataset.csv"
    if os.path.exists(csv_path):
        print(f"[Train] Loading cached dataset from {csv_path}")
        return pd.read_csv(csv_path)
    df = build_dataset()
    df.to_csv(csv_path, index=False)
    return df


def prepare_splits(df: pd.DataFrame):
    meta_cols = [c for c in df.columns if c.startswith("_")]
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y_dri  = df["dri_score"].values.astype(np.float32)
    y_tier = df["tier_label"].values.astype(np.int32)

    X_tr, X_te, y_dri_tr, y_dri_te, y_tier_tr, y_tier_te = train_test_split(
        X, y_dri, y_tier,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_tier,
    )
    return X_tr, X_te, y_dri_tr, y_dri_te, y_tier_tr, y_tier_te


# ══════════════════════════════════════════════════════════════════
# 2. DRI REGRESSOR
# ══════════════════════════════════════════════════════════════════

DRI_PARAMS = {
    "objective":        "reg:squarederror",
    "n_estimators":      600,
    "max_depth":           5,
    "learning_rate":    0.05,
    "subsample":        0.80,
    "colsample_bytree": 0.80,
    "min_child_weight":    3,
    "gamma":            0.10,
    "reg_alpha":        0.05,
    "reg_lambda":       1.00,
    "random_state":     RANDOM_SEED,
    "n_jobs":             -1,
}

def train_dri_model(X_tr, y_tr, X_te, y_te, scaler: StandardScaler):
    print("\n[Train] ── DRI Regressor ─────────────────────────────")
    Xs_tr = scaler.transform(X_tr)
    Xs_te = scaler.transform(X_te)

    model = xgb.XGBRegressor(**DRI_PARAMS)
    t0 = time.time()
    model.fit(
        Xs_tr, y_tr,
        eval_set=[(Xs_te, y_te)],
        verbose=100,
    )
    elapsed = time.time() - t0

    y_pred = model.predict(Xs_te)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)

    print(f"\n  MAE  = {mae:.3f}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  R²   = {r2:.4f}")
    print(f"  Time = {elapsed:.1f}s")

    return model, {"mae": mae, "rmse": rmse, "r2": r2, "time_s": elapsed}, y_pred


# ══════════════════════════════════════════════════════════════════
# 3. TIER CLASSIFIER
# ══════════════════════════════════════════════════════════════════

TIER_PARAMS = {
    "objective":          "multi:softmax",
    "num_class":                        5,
    "n_estimators":                   500,
    "max_depth":                        4,
    "learning_rate":                 0.06,
    "subsample":                     0.80,
    "colsample_bytree":              0.80,
    "min_child_weight":                 2,
    "random_state":           RANDOM_SEED,
    "n_jobs":                          -1,
    "use_label_encoder":            False,
    "eval_metric":              "mlogloss",
}

TIER_NAMES = [t["tier"] for t in PREMIUM_TIERS]

def train_tier_model(X_tr, y_tr, X_te, y_te, scaler: StandardScaler):
    print("\n[Train] ── Tier Classifier ───────────────────────────")
    Xs_tr = scaler.transform(X_tr)
    Xs_te = scaler.transform(X_te)

    model = xgb.XGBClassifier(**TIER_PARAMS)
    t0 = time.time()
    model.fit(
        Xs_tr, y_tr,
        eval_set=[(Xs_te, y_te)],
        verbose=100,
    )
    elapsed = time.time() - t0

    y_pred = model.predict(Xs_te)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=TIER_NAMES)
    cm = confusion_matrix(y_te, y_pred)

    print(f"\n  Accuracy = {acc:.4f}")
    print(f"  Time     = {elapsed:.1f}s")
    print("\n" + report)

    return model, {
        "accuracy": acc, "time_s": elapsed,
        "classification_report": report
    }, y_pred, cm


# ══════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════

def plot_feature_importance(model, title: str, out_path: str):
    scores = model.feature_importances_
    idx    = np.argsort(scores)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(FEATURE_NAMES)))
    ax.barh([FEATURE_NAMES[i] for i in idx[::-1]],
            [scores[i]        for i in idx[::-1]],
            color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_dri_residuals(y_true, y_pred, out_path: str):
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=8, c="steelblue")
    axes[0].axhline(0, color="red", lw=1.5)
    axes[0].set_xlabel("Predicted DRI")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual Plot — DRI Regressor")

    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_confusion_matrix(cm, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=TIER_NAMES, yticklabels=TIER_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted Tier")
    ax.set_ylabel("True Tier")
    ax.set_title("Tier Classifier — Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_dri_vs_actual(y_true, y_pred, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.2, s=6, c="steelblue")
    lims = [0, 100]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect Fit")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True DRI")
    ax.set_ylabel("Predicted DRI")
    ax.set_title("Predicted vs Actual DRI Score")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Insurance Pricing — XGBoost Training Pipeline")
    print("=" * 60)

    # ── Data ───────────────────────────────────────────────────
    df = load_or_generate_data()
    X_tr, X_te, y_dri_tr, y_dri_te, y_tier_tr, y_tier_te = prepare_splits(df)
    print(f"\n  Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}")

    # ── Scaler ─────────────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_tr)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Scaler saved → {SCALER_PATH}")

    # ── Feature names ──────────────────────────────────────────
    with open(FEATURE_NAMES_PATH, "w") as fh:
        json.dump(FEATURE_NAMES, fh, indent=2)
    print(f"  Feature names saved → {FEATURE_NAMES_PATH}")

    # ── DRI Regressor ──────────────────────────────────────────
    dri_model, dri_metrics, dri_preds = train_dri_model(
        X_tr, y_dri_tr, X_te, y_dri_te, scaler
    )
    joblib.dump(dri_model, DRI_MODEL_PATH)
    print(f"\n  DRI model saved → {DRI_MODEL_PATH}")

    # ── Tier Classifier ────────────────────────────────────────
    tier_model, tier_metrics, tier_preds, cm = train_tier_model(
        X_tr, y_tier_tr, X_te, y_tier_te, scaler
    )
    joblib.dump(tier_model, TIER_MODEL_PATH)
    print(f"\n  Tier model saved → {TIER_MODEL_PATH}")

    # ── Plots ──────────────────────────────────────────────────
    print("\n[Train] Generating diagnostic plots …")
    plot_feature_importance(dri_model,  "Feature Importance — DRI Regressor",   "artifacts/dri_feature_importance.png")
    plot_feature_importance(tier_model, "Feature Importance — Tier Classifier",  "artifacts/tier_feature_importance.png")
    plot_dri_residuals(y_dri_te, dri_preds,  "artifacts/dri_residuals.png")
    plot_dri_vs_actual(y_dri_te, dri_preds,  "artifacts/dri_actual_vs_pred.png")
    plot_confusion_matrix(cm,              "artifacts/tier_confusion_matrix.png")

    # ── Training report ────────────────────────────────────────
    report = (
        f"Insurance Pricing — XGBoost Training Report\n"
        f"{'=' * 55}\n\n"
        f"Dataset\n"
        f"  Total samples : {len(df)}\n"
        f"  Train / Test  : {X_tr.shape[0]} / {X_te.shape[0]}\n"
        f"  Features      : {len(FEATURE_NAMES)}\n\n"
        f"DRI Regressor (XGBoost)\n"
        f"  MAE    : {dri_metrics['mae']:.4f}\n"
        f"  RMSE   : {dri_metrics['rmse']:.4f}\n"
        f"  R²     : {dri_metrics['r2']:.4f}\n"
        f"  Train time : {dri_metrics['time_s']:.1f}s\n\n"
        f"Tier Classifier (XGBoost)\n"
        f"  Accuracy   : {tier_metrics['accuracy']:.4f}\n"
        f"  Train time : {tier_metrics['time_s']:.1f}s\n\n"
        f"Classification Report\n"
        f"{tier_metrics['classification_report']}\n"
    )
    with open("artifacts/training_report.txt", "w") as fh:
        fh.write(report)
    print("\n  Training report → artifacts/training_report.txt")

    print("\n" + "=" * 60)
    print("  Training complete. All artifacts in ml/artifacts/")
    print("=" * 60)


if __name__ == "__main__":
    main()
