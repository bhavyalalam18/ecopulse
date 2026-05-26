"""
EcoPulse ML Training Pipeline
Models: Linear Regression, Random Forest, XGBoost
Evaluation: RMSE, MAE, R² with TimeSeriesSplit
Extras: SHAP explainability, feature importance, model serialization
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
DATA_DIR   = os.path.join(BASE_DIR, '..', 'data')
DB_PATH    = os.path.abspath(os.path.join(DATA_DIR, 'ecopulse.db'))

FEATURE_COLS = [
    'hour', 'month', 'dayofweek', 'quarter', 'is_weekend',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'dow_sin', 'dow_cos',
    'lag_1h', 'lag_24h', 'lag_168h',
    'rolling_mean_24h', 'rolling_std_24h', 'rolling_mean_168h'
]
TARGET_COL = 'PJME_MW'


def load_data() -> pd.DataFrame:
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql('SELECT * FROM energy_hourly ORDER BY datetime', conn)
    conn.close()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def compute_metrics(y_true, y_pred, label='') -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    if label:
        print(f"  {label}: RMSE={rmse:.1f} | MAE={mae:.1f} | R²={r2:.4f} | MAPE={mape:.2f}%")
    return {'model': label, 'RMSE': round(rmse, 2), 'MAE': round(mae, 2),
            'R2': round(r2, 4), 'MAPE_pct': round(mape, 2)}


def time_series_cv(model, X, y, n_splits=5, label='') -> dict:
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes, r2s = [], [], []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        rmses.append(np.sqrt(mean_squared_error(y[val_idx], preds)))
        maes.append(mean_absolute_error(y[val_idx], preds))
        r2s.append(r2_score(y[val_idx], preds))
    print(f"  CV {label}: RMSE={np.mean(rmses):.1f}±{np.std(rmses):.1f} "
          f"| R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}")
    return {'cv_rmse_mean': round(np.mean(rmses), 2),
            'cv_rmse_std':  round(np.std(rmses), 2),
            'cv_r2_mean':   round(np.mean(r2s), 4)}


def plot_predictions(y_test, predictions_dict, out_path):
    fig, axes = plt.subplots(len(predictions_dict), 1,
                             figsize=(14, 4 * len(predictions_dict)))
    if len(predictions_dict) == 1:
        axes = [axes]
    sample = min(720, len(y_test))
    for ax, (name, preds) in zip(axes, predictions_dict.items()):
        ax.plot(y_test[:sample].values, label='Actual', color='#2E86AB', linewidth=1)
        ax.plot(preds[:sample],         label='Predicted', color='#E84855',
                linewidth=1, alpha=0.8)
        ax.set_title(f'{name} — Actual vs Predicted (first 720h)')
        ax.set_ylabel('MW')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_residuals(y_test, y_pred, model_name, out_path):
    residuals = y_test.values - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(residuals, bins=60, color='#7B2D8B', edgecolor='white', alpha=0.8)
    ax1.axvline(0, color='red', linestyle='--')
    ax1.set_title(f'{model_name} — Residuals Distribution')
    ax1.set_xlabel('Residual (MW)')
    ax1.set_ylabel('Count')
    ax2.scatter(y_pred[:2000], residuals[:2000], alpha=0.3, s=5, color='#2E86AB')
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_title('Residuals vs Fitted')
    ax2.set_xlabel('Predicted MW')
    ax2.set_ylabel('Residual')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_feature_importance(model, feature_names, out_path):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind='barh', ax=ax, color='#2E86AB')
    ax.set_title('XGBoost — Feature Importance')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_shap(model, X_sample, feature_names, out_path):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample,
                          feature_names=feature_names,
                          show=False, plot_size=(10, 6))
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  SHAP plot skipped: {e}")


def plot_correlation_heatmap(df, out_path):
    cols = FEATURE_COLS + [TARGET_COL]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                linewidths=0.3, ax=ax, fmt='.2f')
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    plots_dir = os.path.join(MODELS_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("\n=== EcoPulse ML Training ===\n")

    # ------------------------------------------------------------------
    print("[1/6] Loading data from SQLite...")
    df = load_data()
    print(f"  Rows: {len(df)} | Features: {len(FEATURE_COLS)}")

    # ------------------------------------------------------------------
    print("\n[2/6] Preparing train/test split (time-based, 80/20)...")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL]
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[TARGET_COL]
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # ------------------------------------------------------------------
    print("\n[3/6] Generating EDA plots...")
    plot_correlation_heatmap(df, os.path.join(plots_dir, 'correlation_heatmap.png'))

    # ------------------------------------------------------------------
    print("\n[4/6] Training models...\n")
    results = []
    predictions = {}

    # --- Linear Regression ---
    print("Linear Regression:")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    results.append(compute_metrics(y_test, lr_pred, 'LinearRegression'))
    time_series_cv(LinearRegression(), X_train_sc, y_train.values, label='LinearRegression')
    predictions['Linear Regression'] = lr_pred

    # --- Random Forest ---
    print("\nRandom Forest:")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12,
                               n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results.append(compute_metrics(y_test, rf_pred, 'RandomForest'))
    time_series_cv(RandomForestRegressor(n_estimators=50, max_depth=10,
                                         n_jobs=-1, random_state=42),
                   X_train, y_train.values, label='RandomForest')
    predictions['Random Forest'] = rf_pred

    # --- XGBoost ---
    print("\nXGBoost:")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        eval_metric='rmse'
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    xgb_pred = xgb_model.predict(X_test)
    results.append(compute_metrics(y_test, xgb_pred, 'XGBoost'))
    time_series_cv(xgb.XGBRegressor(n_estimators=100, max_depth=6,
                                     learning_rate=0.05, random_state=42,
                                     verbosity=0),
                   X_train, y_train.values, label='XGBoost')
    predictions['XGBoost'] = xgb_pred

    # ------------------------------------------------------------------
    print("\n[5/6] Generating plots...")
    plot_predictions(y_test, predictions,
                     os.path.join(plots_dir, 'predictions_comparison.png'))
    plot_residuals(y_test, xgb_pred, 'XGBoost',
                   os.path.join(plots_dir, 'xgb_residuals.png'))
    plot_feature_importance(xgb_model, FEATURE_COLS,
                            os.path.join(plots_dir, 'feature_importance.png'))
    shap_sample = X_test[:500]
    plot_shap(xgb_model, shap_sample, FEATURE_COLS,
              os.path.join(plots_dir, 'shap_summary.png'))

    # ------------------------------------------------------------------
    print("\n[6/6] Saving model artifacts...")
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model.pkl'))
    joblib.dump(rf,        os.path.join(MODELS_DIR, 'rf_model.pkl'))
    joblib.dump(scaler,    os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, 'feature_cols.pkl'))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODELS_DIR, 'model_results.csv'), index=False)

    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))
    print("\nBest model (XGBoost) saved to models/xgb_model.pkl")
    return xgb_model, results_df


if __name__ == '__main__':
    train()
