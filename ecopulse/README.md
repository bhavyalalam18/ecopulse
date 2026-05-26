# ⚡ EcoPulse: Smart City Energy Consumption Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)](https://xgboost.readthedocs.io)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey)](https://sqlite.org)

A full-stack data analytics project predicting hourly energy consumption for the PJME grid using machine learning, SQL analytics, and an interactive Streamlit dashboard.

---

## 🏗 Architecture

```
PJME_hourly.csv
      │
      ▼
[Excel Audit]          ← Data quality, pivot analysis, YoY charts
      │
      ▼
[Python Pipeline]      ← Feature engineering (lags, rolling stats, cyclical encoding)
      │
      ▼
[SQLite Database]      ← 145k+ rows · Window functions · CTEs · Views
      │
      ▼
[ML Models]            ← Linear Regression → Random Forest → XGBoost
      │
      ▼
[Streamlit App]        ← Live predictions · Time-series charts · SQL insights
      │
      ▼
[Streamlit Cloud]      ← Public deployment
```

---

## 📁 Project Structure

```
ecopulse/
├── app.py                    # Streamlit application
├── setup.py                  # One-command setup (pipeline + training + Excel)
├── requirements.txt
├── data/
│   ├── PJME_hourly.csv       # Source dataset
│   ├── ecopulse.db           # SQLite database
│   └── EcoPulse_Audit.xlsx   # Excel audit workbook
├── models/
│   ├── xgb_model.pkl         # Trained XGBoost model
│   ├── rf_model.pkl          # Random Forest model
│   ├── scaler.pkl            # StandardScaler for Linear Regression
│   ├── feature_cols.pkl      # Feature column list
│   ├── model_results.csv     # RMSE/MAE/R² comparison table
│   └── plots/                # Feature importance, SHAP, residuals
├── utils/
│   ├── data_pipeline.py      # CSV → feature engineering → SQLite
│   ├── train_model.py        # Model training, evaluation, serialization
│   └── excel_audit.py        # Excel audit workbook generator
├── sql/
│   └── queries.sql           # Window functions, CTEs, Views
└── notebooks/
    └── (add your EDA notebooks here)
```

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/ecopulse.git
cd ecopulse
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run full setup (pipeline + training + Excel audit)
python setup.py

# 3. Launch the app
streamlit run app.py
```

---

## 🤖 ML Models & Results

| Model             | RMSE (MW) | MAE (MW) | R²     | MAPE   |
|-------------------|-----------|----------|--------|--------|
| Linear Regression | ~1,800    | ~1,400   | ~0.87  | ~5%    |
| Random Forest     | ~450      | ~310     | ~0.98  | ~1.4%  |
| **XGBoost** ✅   | **~320**  | **~220** | **~0.99** | **~1%** |

### Features Used
- **Time features**: hour, month, dayofweek, quarter, is_weekend
- **Cyclical encoding**: hour_sin/cos, month_sin/cos, dow_sin/cos
- **Lag features**: lag_1h, lag_24h, lag_168h (1 week)
- **Rolling stats**: rolling_mean_24h, rolling_std_24h, rolling_mean_168h

---

## 🗄 SQL Highlights

- `RANK() OVER (PARTITION BY month ORDER BY avg_mw DESC)` — hour rankings per month
- `ROW_NUMBER() OVER (PARTITION BY year ORDER BY PJME_MW DESC)` — yearly peaks
- `AVG() OVER (ROWS BETWEEN 167 PRECEDING AND CURRENT ROW)` — 7-day rolling average
- CTEs for year-over-year % change analysis
- Self-join to compare same-hour demand across consecutive years
- SQL `VIEW` (`vw_hourly_features`) consumed by the ML pipeline

---

## ☁️ Deployment

Deployed on **Streamlit Community Cloud**:
1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → main file: `app.py` → Deploy

---

## 📊 Dataset

**PJME Hourly Energy Consumption** — PJM Interconnection LLC  
Source: [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)  
~145,000 hourly records from 2002–2018

---

## 🛠 Tech Stack

| Layer        | Tool                           |
|--------------|--------------------------------|
| Data Audit   | Excel (openpyxl)               |
| Storage      | SQLite + SQLAlchemy            |
| Processing   | Python, Pandas, NumPy          |
| ML           | Scikit-learn, XGBoost          |
| Explainability | SHAP                         |
| Visualization | Plotly, Matplotlib, Seaborn  |
| Dashboard    | Streamlit                      |
| Cloud        | Streamlit Community Cloud      |
| IDE          | VS Code                        |
