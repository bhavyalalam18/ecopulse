"""
EcoPulse: Smart City Energy Consumption Predictor
Streamlit Application — Main Entry Point
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit as st
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
DB_PATH    = os.path.abspath(os.path.join(DATA_DIR, 'ecopulse.db'))

st.set_page_config(
    page_title="EcoPulse — Smart City Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; padding: 6px 18px; }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 14px 18px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML model...")
def load_model():
    path = os.path.join(MODELS_DIR, 'xgb_model.pkl')
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource
def load_feature_cols():
    path = os.path.join(MODELS_DIR, 'feature_cols.pkl')
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data(show_spinner="Loading data...", ttl=3600)
def load_data(limit: int = 26_280) -> pd.DataFrame:
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    query = f'''SELECT datetime, PJME_MW, year, month, day, hour, dayofweek,
               is_weekend, lag_1h, lag_24h, lag_168h,
               rolling_mean_24h, rolling_std_24h, rolling_mean_168h,
               hour_sin, hour_cos, month_sin, month_cos,
               dow_sin, dow_cos, quarter
        FROM energy_hourly ORDER BY datetime DESC LIMIT {limit}'''
    df = pd.read_sql(query, conn)
    conn.close()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3600)
def load_model_results() -> pd.DataFrame | None:
    path = os.path.join(MODELS_DIR, 'model_results.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── Helper ────────────────────────────────────────────────────────────────────

def make_prediction(model, feature_cols, hour, month, dayofweek, is_weekend,
                    lag_1h, lag_24h, lag_168h,
                    rolling_mean_24h, rolling_std_24h, rolling_mean_168h):
    quarter = (month - 1) // 3 + 1
    row = {
        'hour':              hour,
        'month':             month,
        'dayofweek':         dayofweek,
        'quarter':           quarter,
        'is_weekend':        int(is_weekend),
        'hour_sin':          np.sin(2 * np.pi * hour / 24),
        'hour_cos':          np.cos(2 * np.pi * hour / 24),
        'month_sin':         np.sin(2 * np.pi * month / 12),
        'month_cos':         np.cos(2 * np.pi * month / 12),
        'dow_sin':           np.sin(2 * np.pi * dayofweek / 7),
        'dow_cos':           np.cos(2 * np.pi * dayofweek / 7),
        'lag_1h':            lag_1h,
        'lag_24h':           lag_24h,
        'lag_168h':          lag_168h,
        'rolling_mean_24h':  rolling_mean_24h,
        'rolling_std_24h':   rolling_std_24h,
        'rolling_mean_168h': rolling_mean_168h,
    }
    X = pd.DataFrame([row])[feature_cols]
    return float(model.predict(X)[0])


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
    st.title("EcoPulse")
    st.caption("Smart City Energy Predictor")
    st.divider()

    st.subheader("⚡ Predict Demand")
    pred_hour     = st.slider("Hour of Day", 0, 23, 12)
    pred_month    = st.selectbox("Month", range(1, 13),
                                 format_func=lambda m: pd.Timestamp(2024, m, 1).strftime('%B'))
    pred_dow      = st.selectbox("Day of Week",
                                 range(7),
                                 format_func=lambda d: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d])
    pred_weekend  = pred_dow >= 5

    st.caption("Reference values (auto-filled from recent data)")
    pred_lag1     = st.number_input("Lag 1h (MW)",   value=32000.0, step=100.0)
    pred_lag24    = st.number_input("Lag 24h (MW)",  value=31500.0, step=100.0)
    pred_lag168   = st.number_input("Lag 168h (MW)", value=30800.0, step=100.0)
    pred_rmean24  = st.number_input("Rolling Mean 24h", value=31000.0, step=100.0)
    pred_rstd24   = st.number_input("Rolling Std 24h",  value=2500.0,  step=100.0)
    pred_rmean168 = st.number_input("Rolling Mean 168h", value=30500.0, step=100.0)

    predict_btn = st.button("🔮 Predict", use_container_width=True, type="primary")
    st.divider()
    st.caption("Tech Stack: Python · XGBoost · SQLite · Streamlit")


# ── Load assets ───────────────────────────────────────────────────────────────

model        = load_model()
feature_cols = load_feature_cols()
df           = load_data()
model_results = load_model_results()

has_model = (model is not None and feature_cols is not None)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("⚡ EcoPulse: Smart City Energy Consumption Predictor")
st.markdown(
    "**PJME Grid** · Hourly Energy Demand Analysis · XGBoost ML Model · "
    f"{len(df):,} records loaded"
)

# ── Prediction result ─────────────────────────────────────────────────────────

if predict_btn:
    if has_model:
        pred_mw = make_prediction(
            model, feature_cols,
            pred_hour, pred_month, pred_dow, pred_weekend,
            pred_lag1, pred_lag24, pred_lag168,
            pred_rmean24, pred_rstd24, pred_rmean168
        )
        avg_mw = df['PJME_MW'].mean()
        delta  = pred_mw - avg_mw
        st.success(
            f"🔮 **Predicted Demand: {pred_mw:,.1f} MW** "
            f"({'↑' if delta > 0 else '↓'} {abs(delta):,.0f} MW vs historical avg)"
        )
    else:
        st.warning("Model not found. Run `python utils/train_model.py` first.")

st.divider()

# ── KPI Cards ─────────────────────────────────────────────────────────────────

last_24h  = df[df['datetime'] >= df['datetime'].max() - pd.Timedelta(hours=24)]
last_7d   = df[df['datetime'] >= df['datetime'].max() - pd.Timedelta(days=7)]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📊 Avg MW (All time)",  f"{df['PJME_MW'].mean():,.0f}")
k2.metric("📈 Peak MW (7d)",       f"{last_7d['PJME_MW'].max():,.0f}")
k3.metric("📉 Min MW (7d)",        f"{last_7d['PJME_MW'].min():,.0f}")
k4.metric("⚡ Avg MW (24h)",       f"{last_24h['PJME_MW'].mean():,.0f}")
k5.metric("🗓 Records Loaded",     f"{len(df):,}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Time Series", "🗓 Patterns", "🤖 Model Results",
    "🔍 SQL Insights", "📋 Raw Data"
])

# ── Tab 1: Time Series ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Energy Consumption Over Time")
    col_a, col_b = st.columns([3, 1])
    with col_b:
        lookback = st.selectbox("Show last", ["30 days", "90 days",
                                               "180 days", "1 year", "All"], index=0)
    lookup = {"30 days": 30, "90 days": 90, "180 days": 180,
              "1 year": 365, "All": None}
    days_n = lookup[lookback]
    plot_df = (df[df['datetime'] >= df['datetime'].max() - pd.Timedelta(days=days_n)]
               if days_n else df)

    fig_ts = px.line(plot_df, x='datetime', y='PJME_MW',
                     title=f"Hourly Energy Demand — {lookback}",
                     labels={'PJME_MW': 'MW', 'datetime': 'Date'},
                     color_discrete_sequence=['#2E86AB'])
    fig_ts.add_hline(y=df['PJME_MW'].mean(), line_dash='dash',
                     line_color='#E84855', annotation_text='Historical avg')
    fig_ts.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig_ts, use_container_width=True)

    # Rolling average overlay
    st.subheader("Rolling 7-Day Average vs Actual")
    roll_df = plot_df.copy()
    roll_df['rolling_7d'] = roll_df['PJME_MW'].rolling(168).mean()
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=roll_df['datetime'], y=roll_df['PJME_MW'],
                                  name='Actual', line=dict(color='#2E86AB', width=1),
                                  opacity=0.5))
    fig_roll.add_trace(go.Scatter(x=roll_df['datetime'], y=roll_df['rolling_7d'],
                                  name='7-Day Avg', line=dict(color='#E84855', width=2)))
    fig_roll.update_layout(height=350, hovermode='x unified',
                            title="Actual vs 7-Day Rolling Average")
    st.plotly_chart(fig_roll, use_container_width=True)

# ── Tab 2: Patterns ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Demand Patterns")
    c1, c2 = st.columns(2)

    with c1:
        hourly_avg = df.groupby('hour')['PJME_MW'].mean().reset_index()
        fig_h = px.bar(hourly_avg, x='hour', y='PJME_MW',
                       title="Average Demand by Hour of Day",
                       color='PJME_MW', color_continuous_scale='Blues',
                       labels={'PJME_MW': 'Avg MW', 'hour': 'Hour'})
        fig_h.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_h, use_container_width=True)

    with c2:
        monthly_avg = df.groupby('month')['PJME_MW'].mean().reset_index()
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        monthly_avg['month_name'] = monthly_avg['month'].apply(
            lambda m: month_names[m - 1])
        fig_m = px.bar(monthly_avg, x='month_name', y='PJME_MW',
                       title="Average Demand by Month",
                       color='PJME_MW', color_continuous_scale='Oranges',
                       labels={'PJME_MW': 'Avg MW', 'month_name': 'Month'})
        fig_m.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_m, use_container_width=True)

    # Heatmap: Hour vs Day of Week
    st.subheader("Demand Heatmap — Hour × Day of Week")
    heatmap_data = df.groupby(['dayofweek', 'hour'])['PJME_MW'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='dayofweek', columns='hour', values='PJME_MW')
    dow_labels = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    fig_hm = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg MW"),
        y=dow_labels,
        color_continuous_scale='RdYlBu_r',
        title="Average Energy Demand Heatmap (Hour × Weekday)",
        aspect='auto'
    )
    fig_hm.update_layout(height=380)
    st.plotly_chart(fig_hm, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Year-over-year
        yoy = df.groupby('year')['PJME_MW'].mean().reset_index()
        fig_yoy = px.line(yoy, x='year', y='PJME_MW', markers=True,
                          title="Year-over-Year Average Demand",
                          labels={'PJME_MW': 'Avg MW', 'year': 'Year'},
                          color_discrete_sequence=['#1D9E75'])
        fig_yoy.update_layout(height=300)
        st.plotly_chart(fig_yoy, use_container_width=True)

    with c4:
        # Weekday vs Weekend
        wkd = df.groupby('is_weekend')['PJME_MW'].mean().reset_index()
        wkd['type'] = wkd['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
        fig_wkd = px.bar(wkd, x='type', y='PJME_MW',
                         title="Weekday vs Weekend Average Demand",
                         color='type',
                         color_discrete_map={'Weekday': '#2E86AB', 'Weekend': '#E84855'},
                         labels={'PJME_MW': 'Avg MW', 'type': ''})
        fig_wkd.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_wkd, use_container_width=True)

# ── Tab 3: Model Results ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Machine Learning Model Comparison")

    if model_results is not None:
        # Metric cards
        xgb_row = model_results[model_results['model'] == 'XGBoost'].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("XGBoost RMSE", f"{xgb_row['RMSE']:,.1f} MW")
        m2.metric("XGBoost MAE",  f"{xgb_row['MAE']:,.1f} MW")
        m3.metric("XGBoost R²",   f"{xgb_row['R2']:.4f}")
        m4.metric("XGBoost MAPE", f"{xgb_row['MAPE_pct']:.2f}%")

        st.dataframe(
            model_results.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE_pct'],
                                               color='#d4edda')
                               .highlight_max(subset=['R2'], color='#d4edda')
                               .format({'RMSE': '{:.1f}', 'MAE': '{:.1f}',
                                        'R2': '{:.4f}', 'MAPE_pct': '{:.2f}%'}),
            use_container_width=True, hide_index=True
        )

        # Metrics bar chart
        metrics_melt = model_results.melt(id_vars='model',
                                           value_vars=['RMSE', 'MAE'],
                                           var_name='Metric', value_name='Value')
        fig_metrics = px.bar(metrics_melt, x='model', y='Value', color='Metric',
                             barmode='group', title="RMSE & MAE Comparison",
                             color_discrete_sequence=['#2E86AB', '#E84855'])
        fig_metrics.update_layout(height=350)
        st.plotly_chart(fig_metrics, use_container_width=True)
    else:
        st.info("Train the model first: `cd ecopulse && python utils/train_model.py`")

    # Feature importance plot
    plots_dir = os.path.join(MODELS_DIR, 'plots')
    fi_path   = os.path.join(plots_dir, 'feature_importance.png')
    shap_path = os.path.join(plots_dir, 'shap_summary.png')
    c5, c6 = st.columns(2)
    with c5:
        if os.path.exists(fi_path):
            st.image(fi_path, caption="XGBoost Feature Importance", use_container_width=True)
    with c6:
        if os.path.exists(shap_path):
            st.image(shap_path, caption="SHAP Summary Plot", use_container_width=True)

    res_path = os.path.join(plots_dir, 'xgb_residuals.png')
    pred_path = os.path.join(plots_dir, 'predictions_comparison.png')
    if os.path.exists(res_path):
        st.image(res_path, caption="XGBoost Residuals", use_container_width=True)
    if os.path.exists(pred_path):
        st.image(pred_path, caption="Predictions vs Actual", use_container_width=True)

# ── Tab 4: SQL Insights ───────────────────────────────────────────────────────
with tab4:
    st.subheader("SQL Window Function Results")
    import sqlite3

    st.markdown("**Peak hour per year (ROW_NUMBER window function)**")
    with sqlite3.connect(DB_PATH) as conn:
        peak_q = """
            WITH yearly_peaks AS (
                SELECT year, datetime, PJME_MW,
                       ROW_NUMBER() OVER (PARTITION BY year ORDER BY PJME_MW DESC) AS rn
                FROM energy_hourly
            )
            SELECT year,
                   datetime AS peak_datetime,
                   ROUND(PJME_MW, 1) AS peak_mw
            FROM yearly_peaks WHERE rn = 1 ORDER BY year
        """
        peak_df = pd.read_sql(peak_q, conn)
    st.dataframe(peak_df, use_container_width=True, hide_index=True)

    st.markdown("**Monthly rank by demand within year (RANK window function)**")
    with sqlite3.connect(DB_PATH) as conn:
        rank_q = """
            SELECT year, month,
                   ROUND(AVG(PJME_MW), 1) AS avg_mw,
                   RANK() OVER (PARTITION BY year ORDER BY AVG(PJME_MW) DESC) AS rank_in_year
            FROM energy_hourly
            GROUP BY year, month
            ORDER BY year, rank_in_year
            LIMIT 60
        """
        rank_df = pd.read_sql(rank_q, conn)
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    st.markdown("**Weekday vs Weekend summary (CTE)**")
    with sqlite3.connect(DB_PATH) as conn:
        wkd_q = """
            WITH summary AS (
                SELECT year,
                       CASE WHEN is_weekend=1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
                       ROUND(AVG(PJME_MW), 1) AS avg_mw,
                       ROUND(MAX(PJME_MW), 1) AS peak_mw,
                       COUNT(*) AS hours
                FROM energy_hourly
                GROUP BY year, is_weekend
            )
            SELECT * FROM summary ORDER BY year, day_type
        """
        wkd_sql_df = pd.read_sql(wkd_q, conn)
    st.dataframe(wkd_sql_df, use_container_width=True, hide_index=True)

# ── Tab 5: Raw Data ───────────────────────────────────────────────────────────
with tab5:
    st.subheader("Raw Data Explorer")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_year = st.multiselect("Filter by Year",
                                     sorted(df['year'].unique()),
                                     default=[])
    with col_f2:
        filter_month = st.multiselect("Filter by Month", range(1, 13), default=[])

    show_df = df.copy()
    if filter_year:
        show_df = show_df[show_df['year'].isin(filter_year)]
    if filter_month:
        show_df = show_df[show_df['month'].isin(filter_month)]

    st.dataframe(
        show_df[['datetime', 'PJME_MW', 'year', 'month', 'hour',
                  'dayofweek', 'is_weekend', 'rolling_mean_24h']].head(500),
        use_container_width=True, hide_index=True
    )
    csv_bytes = show_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download filtered data as CSV",
                       data=csv_bytes, file_name='ecopulse_filtered.csv',
                       mime='text/csv')
