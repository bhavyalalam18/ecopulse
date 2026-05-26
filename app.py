"""
EcoPulse: Smart City Energy Consumption Predictor
Streamlit Application — UPGRADED with:
  - Anomaly Detection (Isolation Forest)
  - 24-Hour Forecast
  - CO2 Emission Estimator
  - Peak Demand Alert
  - Power BI Guide Tab
"""

import os
import warnings
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit as st
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
DB_PATH    = os.path.abspath(os.path.join(DATA_DIR, 'ecopulse.db'))

CARBON_INTENSITY = 0.386   # kg CO2 per kWh (US grid average)
P90_THRESHOLD    = None    # set after data loads

st.set_page_config(
    page_title="EcoPulse — Smart City Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
div[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)


# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML model...")
def load_model():
    path = os.path.join(MODELS_DIR, 'xgb_model.pkl')
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource
def load_feature_cols():
    path = os.path.join(MODELS_DIR, 'feature_cols.pkl')
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data(show_spinner="Loading data...", ttl=3600)
def load_data(limit: int = 26_280) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT datetime, PJME_MW, year, month, day, hour, dayofweek,
               is_weekend, lag_1h, lag_24h, lag_168h,
               rolling_mean_24h, rolling_std_24h, rolling_mean_168h,
               hour_sin, hour_cos, month_sin, month_cos,
               dow_sin, dow_cos, quarter
        FROM energy_hourly
        ORDER BY datetime DESC
        LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3600)
def load_model_results() -> pd.DataFrame:
    path = os.path.join(MODELS_DIR, 'model_results.csv')
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data(show_spinner="Running anomaly detection...")
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    iso = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
    df = df.copy()
    df['anomaly_score'] = iso.fit_predict(df[['PJME_MW']])
    df['is_anomaly']    = df['anomaly_score'] == -1
    return df


# ── Prediction helpers ────────────────────────────────────────────────────────

def build_feature_row(hour, month, dayofweek, is_weekend,
                      lag_1h, lag_24h, lag_168h,
                      rolling_mean_24h, rolling_std_24h, rolling_mean_168h):
    quarter = (month - 1) // 3 + 1
    return {
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


def make_prediction(model, feature_cols, **kwargs):
    row = build_feature_row(**kwargs)
    X   = pd.DataFrame([row])[feature_cols]
    return float(model.predict(X)[0])


def forecast_24h(model, feature_cols, df: pd.DataFrame, start_hour: int,
                 start_month: int, start_dow: int) -> pd.DataFrame:
    """Generate 24-hour iterative forecast."""
    recent    = df['PJME_MW'].values
    lag_1h    = float(recent[-1])
    lag_24h   = float(recent[-24]) if len(recent) >= 24  else lag_1h
    lag_168h  = float(recent[-168]) if len(recent) >= 168 else lag_1h
    roll_mean = float(np.mean(recent[-24:]))
    roll_std  = float(np.std(recent[-24:]))
    roll_168  = float(np.mean(recent[-168:])) if len(recent) >= 168 else roll_mean

    preds, hours, lowers, uppers = [], [], [], []
    history = list(recent[-24:])

    for i in range(24):
        h   = (start_hour + i) % 24
        dow = (start_dow + (start_hour + i) // 24) % 7
        row = build_feature_row(
            hour=h, month=start_month, dayofweek=dow,
            is_weekend=int(dow >= 5),
            lag_1h=lag_1h, lag_24h=lag_24h, lag_168h=lag_168h,
            rolling_mean_24h=roll_mean, rolling_std_24h=roll_std,
            rolling_mean_168h=roll_168
        )
        X    = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(X)[0])

        # Simple ±5% confidence band
        margin = pred * 0.05
        preds.append(pred)
        lowers.append(pred - margin)
        uppers.append(pred + margin)
        hours.append(f"Hour +{i+1} ({h:02d}:00)")

        # Update lags iteratively
        history.append(pred)
        lag_1h   = pred
        lag_24h  = history[-24] if len(history) >= 24 else pred
        roll_mean = float(np.mean(history[-24:]))
        roll_std  = float(np.std(history[-24:]))

    return pd.DataFrame({
        'label':  hours,
        'pred':   preds,
        'lower':  lowers,
        'upper':  uppers,
        'co2_kg': [p * CARBON_INTENSITY for p in preds],
    })


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
    st.title("EcoPulse")
    st.caption("Smart City Energy Predictor")
    st.divider()

    st.subheader("⚡ Predict Demand")
    pred_hour    = st.slider("Hour of Day", 0, 23, 12)
    pred_month   = st.selectbox(
        "Month", range(1, 13),
        format_func=lambda m: pd.Timestamp(2024, m, 1).strftime('%B'))
    pred_dow     = st.selectbox(
        "Day of Week", range(7),
        format_func=lambda d: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d])
    pred_weekend = pred_dow >= 5

    st.caption("Reference lag values")
    pred_lag1    = st.number_input("Lag 1h (MW)",        value=32000.0, step=100.0)
    pred_lag24   = st.number_input("Lag 24h (MW)",       value=31500.0, step=100.0)
    pred_lag168  = st.number_input("Lag 168h (MW)",      value=30800.0, step=100.0)
    pred_rmean24 = st.number_input("Rolling Mean 24h",   value=31000.0, step=100.0)
    pred_rstd24  = st.number_input("Rolling Std 24h",    value=2500.0,  step=100.0)
    pred_rmean168= st.number_input("Rolling Mean 168h",  value=30500.0, step=100.0)

    predict_btn  = st.button("🔮 Predict", use_container_width=True, type="primary")
    forecast_btn = st.button("📈 Forecast 24h", use_container_width=True)
    st.divider()
    st.caption("Python · XGBoost · SQLite · Streamlit")


# ── Load assets ───────────────────────────────────────────────────────────────

model        = load_model()
feature_cols = load_feature_cols()
df           = load_data()
df_anomaly   = detect_anomalies(df)
model_results= load_model_results()
has_model    = model is not None and feature_cols is not None

P90 = float(df['PJME_MW'].quantile(0.90))
P10 = float(df['PJME_MW'].quantile(0.10))

# ── Header ────────────────────────────────────────────────────────────────────

st.title("⚡ EcoPulse: Smart City Energy Consumption Predictor")
st.markdown(
    f"**PJME Grid** · Hourly Energy Demand Analysis · XGBoost ML Model · "
    f"{len(df):,} records · Carbon intensity: {CARBON_INTENSITY} kg CO₂/kWh"
)

# ── Prediction output ─────────────────────────────────────────────────────────

if predict_btn:
    if has_model:
        pred_mw  = make_prediction(
            model, feature_cols,
            hour=pred_hour, month=pred_month, dayofweek=pred_dow,
            is_weekend=pred_weekend,
            lag_1h=pred_lag1, lag_24h=pred_lag24, lag_168h=pred_lag168,
            rolling_mean_24h=pred_rmean24, rolling_std_24h=pred_rstd24,
            rolling_mean_168h=pred_rmean168
        )
        co2_kg  = pred_mw * CARBON_INTENSITY
        avg_mw  = df['PJME_MW'].mean()
        delta   = pred_mw - avg_mw

        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.success(
            f"🔮 **Predicted: {pred_mw:,.0f} MW** "
            f"({'↑' if delta > 0 else '↓'} {abs(delta):,.0f} MW vs avg)"
        )
        col_p2.info(f"🌿 **Est. CO₂: {co2_kg:,.0f} kg/hr**")

        if pred_mw > P90:
            col_p3.warning("⚠️ **Peak alert** — above 90th percentile! Grid stress likely.")
        elif pred_mw < P10:
            col_p3.info("✅ **Low demand** — below 10th percentile. Efficient period.")
        else:
            col_p3.success("✅ **Normal demand** — within typical range.")
    else:
        st.warning("Model not found. Run `python setup.py` first.")

# ── 24h Forecast output ───────────────────────────────────────────────────────

if forecast_btn:
    if has_model:
        with st.spinner("Generating 24-hour forecast..."):
            fc_df = forecast_24h(
                model, feature_cols, df,
                start_hour=pred_hour,
                start_month=pred_month,
                start_dow=pred_dow
            )
        st.subheader("📈 24-Hour Energy Demand Forecast")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Peak Forecast",   f"{fc_df['pred'].max():,.0f} MW")
        fc2.metric("Min Forecast",    f"{fc_df['pred'].min():,.0f} MW")
        fc3.metric("Total CO₂ (24h)", f"{fc_df['co2_kg'].sum()/1000:,.1f} tonnes")

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=fc_df['label'], y=fc_df['upper'],
            fill=None, mode='lines',
            line=dict(color='rgba(46,134,171,0.2)', width=0),
            showlegend=False, name='Upper bound'
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_df['label'], y=fc_df['lower'],
            fill='tonexty', mode='lines',
            line=dict(color='rgba(46,134,171,0.2)', width=0),
            fillcolor='rgba(46,134,171,0.15)',
            name='±5% confidence band'
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_df['label'], y=fc_df['pred'],
            mode='lines+markers',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6),
            name='Predicted MW'
        ))
        fig_fc.add_hline(
            y=P90, line_dash='dash', line_color='#E84855',
            annotation_text='90th percentile (peak threshold)'
        )
        fig_fc.update_layout(
            height=420, hovermode='x unified',
            title="Next 24-Hour Demand Forecast with Confidence Band",
            yaxis_title="MW", xaxis_title="Hour"
        )
        fig_fc.update_xaxes(tickangle=45)
        st.plotly_chart(fig_fc, use_container_width=True)

        st.dataframe(
            fc_df.rename(columns={
                'label': 'Hour', 'pred': 'Predicted MW',
                'lower': 'Lower Bound', 'upper': 'Upper Bound',
                'co2_kg': 'Est. CO₂ (kg)'
            }).style.format({
                'Predicted MW': '{:,.0f}', 'Lower Bound': '{:,.0f}',
                'Upper Bound': '{:,.0f}', 'Est. CO₂ (kg)': '{:,.0f}'
            }),
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("Model not found. Run `python setup.py` first.")

st.divider()

# ── KPI Cards ─────────────────────────────────────────────────────────────────

last_24h = df[df['datetime'] >= df['datetime'].max() - pd.Timedelta(hours=24)]
last_7d  = df[df['datetime'] >= df['datetime'].max() - pd.Timedelta(days=7)]
n_anomalies = int(df_anomaly['is_anomaly'].sum())

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Avg MW (All time)",  f"{df['PJME_MW'].mean():,.0f}")
k2.metric("Peak MW (7d)",       f"{last_7d['PJME_MW'].max():,.0f}")
k3.metric("Min MW (7d)",        f"{last_7d['PJME_MW'].min():,.0f}")
k4.metric("Avg MW (24h)",       f"{last_24h['PJME_MW'].mean():,.0f}")
k5.metric("Anomalies detected", f"{n_anomalies:,}")
k6.metric("Records loaded",     f"{len(df):,}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Time Series", "🚨 Anomalies", "🗓 Patterns",
    "🤖 Model Results", "🔍 SQL Insights", "📋 Raw Data", "📊 Power BI"
])

# ── Tab 1: Time Series ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Energy Consumption Over Time")
    col_a, col_b = st.columns([3, 1])
    with col_b:
        lookback = st.selectbox(
            "Show last",
            ["30 days", "90 days", "180 days", "1 year", "All"], index=0)
    lookup = {"30 days": 30, "90 days": 90, "180 days": 180,
              "1 year": 365, "All": None}
    days_n  = lookup[lookback]
    plot_df = (df[df['datetime'] >= df['datetime'].max() - pd.Timedelta(days=days_n)]
               if days_n else df)

    fig_ts = px.line(plot_df, x='datetime', y='PJME_MW',
                     title=f"Hourly Energy Demand — {lookback}",
                     labels={'PJME_MW': 'MW', 'datetime': 'Date'},
                     color_discrete_sequence=['#2E86AB'])
    fig_ts.add_hline(y=df['PJME_MW'].mean(), line_dash='dash',
                     line_color='#E84855', annotation_text='Historical avg')
    fig_ts.add_hline(y=P90, line_dash='dot',
                     line_color='#FF8C00', annotation_text='90th pct (peak)')
    fig_ts.update_layout(height=420, hovermode='x unified')
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("Rolling 7-Day Average vs Actual")
    roll_df = plot_df.copy()
    roll_df['rolling_7d'] = roll_df['PJME_MW'].rolling(168).mean()
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=roll_df['datetime'], y=roll_df['PJME_MW'],
        name='Actual', line=dict(color='#2E86AB', width=1), opacity=0.5))
    fig_roll.add_trace(go.Scatter(
        x=roll_df['datetime'], y=roll_df['rolling_7d'],
        name='7-Day Avg', line=dict(color='#E84855', width=2)))
    fig_roll.update_layout(height=350, hovermode='x unified',
                           title="Actual vs 7-Day Rolling Average")
    st.plotly_chart(fig_roll, use_container_width=True)

# ── Tab 2: Anomaly Detection ──────────────────────────────────────────────────
with tab2:
    st.subheader("🚨 Anomaly Detection — Isolation Forest")
    st.markdown(
        f"Detected **{n_anomalies} anomalous hours** out of {len(df):,} records "
        f"({n_anomalies/len(df)*100:.1f}% contamination rate). "
        "These are hours where energy demand was statistically unusual."
    )

    a1, a2, a3 = st.columns(3)
    anomaly_df = df_anomaly[df_anomaly['is_anomaly']]
    a1.metric("Total anomalies",     f"{n_anomalies:,}")
    a2.metric("Avg MW (anomalies)",  f"{anomaly_df['PJME_MW'].mean():,.0f}")
    a3.metric("Max anomaly MW",      f"{anomaly_df['PJME_MW'].max():,.0f}")

    # Anomaly scatter plot
    col_look, _ = st.columns([1, 3])
    with col_look:
        an_lookback = st.selectbox("View window", ["30 days", "90 days", "1 year", "All"],
                                   index=2, key='an_lb')
    an_days = {"30 days": 30, "90 days": 90, "1 year": 365, "All": None}[an_lookback]
    an_plot = (df_anomaly[df_anomaly['datetime'] >= df_anomaly['datetime'].max()
                          - pd.Timedelta(days=an_days)] if an_days else df_anomaly)

    normal_df = an_plot[~an_plot['is_anomaly']]
    anom_df   = an_plot[an_plot['is_anomaly']]

    fig_an = go.Figure()
    fig_an.add_trace(go.Scatter(
        x=normal_df['datetime'], y=normal_df['PJME_MW'],
        mode='lines', name='Normal',
        line=dict(color='#2E86AB', width=1), opacity=0.6))
    fig_an.add_trace(go.Scatter(
        x=anom_df['datetime'], y=anom_df['PJME_MW'],
        mode='markers', name='Anomaly',
        marker=dict(color='#E84855', size=8, symbol='x')))
    fig_an.add_hline(y=P90, line_dash='dash', line_color='#FF8C00',
                     annotation_text='Peak threshold (P90)')
    fig_an.update_layout(height=420, hovermode='x unified',
                         title="Energy Demand with Anomaly Markers (red ✕)")
    st.plotly_chart(fig_an, use_container_width=True)

    # Anomaly distribution
    st.subheader("Anomaly distribution by hour and month")
    c_an1, c_an2 = st.columns(2)
    with c_an1:
        an_hour = anomaly_df.groupby('hour').size().reset_index(name='count')
        fig_ah  = px.bar(an_hour, x='hour', y='count',
                         title="Anomalies by Hour of Day",
                         color='count', color_continuous_scale='Reds',
                         labels={'count': 'Anomaly count', 'hour': 'Hour'})
        fig_ah.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_ah, use_container_width=True)
    with c_an2:
        an_month = anomaly_df.groupby('month').size().reset_index(name='count')
        fig_am   = px.bar(an_month, x='month', y='count',
                          title="Anomalies by Month",
                          color='count', color_continuous_scale='Oranges',
                          labels={'count': 'Anomaly count', 'month': 'Month'})
        fig_am.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_am, use_container_width=True)

    st.subheader("Anomalous records")
    st.dataframe(
        anomaly_df[['datetime', 'PJME_MW', 'year', 'month', 'hour', 'is_weekend']]
        .sort_values('PJME_MW', ascending=False)
        .head(100)
        .style.format({'PJME_MW': '{:,.1f}'}),
        use_container_width=True, hide_index=True
    )

# ── Tab 3: Patterns ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Demand Patterns")
    c1, c2 = st.columns(2)
    with c1:
        hourly_avg = df.groupby('hour')['PJME_MW'].mean().reset_index()
        fig_h = px.bar(hourly_avg, x='hour', y='PJME_MW',
                       title="Average Demand by Hour",
                       color='PJME_MW', color_continuous_scale='Blues',
                       labels={'PJME_MW': 'Avg MW', 'hour': 'Hour'})
        fig_h.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_h, use_container_width=True)
    with c2:
        monthly_avg = df.groupby('month')['PJME_MW'].mean().reset_index()
        mnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly_avg['month_name'] = monthly_avg['month'].apply(lambda m: mnames[m-1])
        fig_m = px.bar(monthly_avg, x='month_name', y='PJME_MW',
                       title="Average Demand by Month",
                       color='PJME_MW', color_continuous_scale='Oranges',
                       labels={'PJME_MW': 'Avg MW', 'month_name': 'Month'})
        fig_m.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_m, use_container_width=True)

    st.subheader("Demand Heatmap — Hour × Day of Week")
    hm = df.groupby(['dayofweek', 'hour'])['PJME_MW'].mean().reset_index()
    hm_pivot = hm.pivot(index='dayofweek', columns='hour', values='PJME_MW')
    dow_labels = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    fig_hm = px.imshow(hm_pivot, labels=dict(x="Hour", y="Day", color="Avg MW"),
                       y=dow_labels, color_continuous_scale='RdYlBu_r',
                       title="Avg Energy Demand Heatmap", aspect='auto')
    fig_hm.update_layout(height=370)
    st.plotly_chart(fig_hm, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        yoy = df.groupby('year')['PJME_MW'].mean().reset_index()
        fig_yoy = px.line(yoy, x='year', y='PJME_MW', markers=True,
                          title="Year-over-Year Average Demand",
                          color_discrete_sequence=['#1D9E75'])
        fig_yoy.update_layout(height=300)
        st.plotly_chart(fig_yoy, use_container_width=True)
    with c4:
        wkd = df.groupby('is_weekend')['PJME_MW'].mean().reset_index()
        wkd['type'] = wkd['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
        fig_wkd = px.bar(wkd, x='type', y='PJME_MW', title="Weekday vs Weekend",
                         color='type',
                         color_discrete_map={'Weekday': '#2E86AB', 'Weekend': '#E84855'})
        fig_wkd.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_wkd, use_container_width=True)

# ── Tab 4: Model Results ──────────────────────────────────────────────────────
with tab4:
    st.subheader("Machine Learning Model Comparison")
    if model_results is not None:
        xgb_row = model_results[model_results['model'] == 'XGBoost'].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("XGBoost RMSE", f"{xgb_row['RMSE']:,.1f} MW")
        m2.metric("XGBoost MAE",  f"{xgb_row['MAE']:,.1f} MW")
        m3.metric("XGBoost R²",   f"{xgb_row['R2']:.4f}")
        m4.metric("XGBoost MAPE", f"{xgb_row['MAPE_pct']:.2f}%")

        st.dataframe(
            model_results.style
            .highlight_min(subset=['RMSE','MAE','MAPE_pct'], color='#d4edda')
            .highlight_max(subset=['R2'], color='#d4edda')
            .format({'RMSE':'{:.1f}','MAE':'{:.1f}','R2':'{:.4f}','MAPE_pct':'{:.2f}%'}),
            use_container_width=True, hide_index=True
        )
        mm = model_results.melt(
            id_vars='model', value_vars=['RMSE','MAE'],
            var_name='Metric', value_name='Value')
        fig_metrics = px.bar(mm, x='model', y='Value', color='Metric',
                             barmode='group', title="RMSE & MAE Comparison",
                             color_discrete_sequence=['#2E86AB','#E84855'])
        fig_metrics.update_layout(height=340)
        st.plotly_chart(fig_metrics, use_container_width=True)
    else:
        st.info("Run `python setup.py` to train models and generate results.")

    plots_dir = os.path.join(MODELS_DIR, 'plots')
    c5, c6    = st.columns(2)
    with c5:
        fi = os.path.join(plots_dir, 'feature_importance.png')
        if os.path.exists(fi):
            st.image(fi, caption="XGBoost Feature Importance", use_column_width=True)
    with c6:
        sh = os.path.join(plots_dir, 'shap_summary.png')
        if os.path.exists(sh):
            st.image(sh, caption="SHAP Summary Plot", use_column_width=True)

    rp = os.path.join(plots_dir, 'xgb_residuals.png')
    pp = os.path.join(plots_dir, 'predictions_comparison.png')
    if os.path.exists(rp):
        st.image(rp, caption="XGBoost Residuals", use_column_width=True)
    if os.path.exists(pp):
        st.image(pp, caption="Predictions vs Actual", use_column_width=True)

# ── Tab 5: SQL Insights ───────────────────────────────────────────────────────
with tab5:
    st.subheader("SQL Window Function Results")
    conn = sqlite3.connect(DB_PATH)

    st.markdown("**Peak hour per year — ROW_NUMBER() OVER (PARTITION BY year)**")
    peak_df = pd.read_sql("""
        WITH yearly_peaks AS (
            SELECT year, datetime, PJME_MW,
                   ROW_NUMBER() OVER (PARTITION BY year ORDER BY PJME_MW DESC) AS rn
            FROM energy_hourly
        )
        SELECT year, datetime AS peak_datetime, ROUND(PJME_MW,1) AS peak_mw
        FROM yearly_peaks WHERE rn=1 ORDER BY year
    """, conn)
    st.dataframe(peak_df, use_container_width=True, hide_index=True)

    st.markdown("**Monthly demand rank — RANK() OVER (PARTITION BY year)**")
    rank_df = pd.read_sql("""
        SELECT year, month,
               ROUND(AVG(PJME_MW),1) AS avg_mw,
               RANK() OVER (PARTITION BY year ORDER BY AVG(PJME_MW) DESC) AS rank_in_year
        FROM energy_hourly
        GROUP BY year, month
        ORDER BY year, rank_in_year LIMIT 60
    """, conn)
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    st.markdown("**Weekday vs Weekend — CTE aggregation**")
    wkd_df = pd.read_sql("""
        WITH summary AS (
            SELECT year,
                   CASE WHEN is_weekend=1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
                   ROUND(AVG(PJME_MW),1) AS avg_mw,
                   ROUND(MAX(PJME_MW),1) AS peak_mw,
                   COUNT(*) AS hours
            FROM energy_hourly GROUP BY year, is_weekend
        )
        SELECT * FROM summary ORDER BY year, day_type
    """, conn)
    st.dataframe(wkd_df, use_container_width=True, hide_index=True)

    st.markdown("**Anomaly count per year — from detection results**")
    an_sql = df_anomaly.groupby('year')['is_anomaly'].agg(
        total_hours='count', anomalies='sum').reset_index()
    an_sql['anomaly_rate_%'] = (an_sql['anomalies'] / an_sql['total_hours'] * 100).round(2)
    st.dataframe(an_sql, use_container_width=True, hide_index=True)

    conn.close()

# ── Tab 6: Raw Data ───────────────────────────────────────────────────────────
with tab6:
    st.subheader("Raw Data Explorer")
    cf1, cf2 = st.columns(2)
    with cf1:
        fy = st.multiselect("Filter by Year",  sorted(df['year'].unique()),  default=[])
    with cf2:
        fm = st.multiselect("Filter by Month", range(1, 13), default=[])

    show_df = df.copy()
    if fy: show_df = show_df[show_df['year'].isin(fy)]
    if fm: show_df = show_df[show_df['month'].isin(fm)]

    st.dataframe(
        show_df[['datetime','PJME_MW','year','month','hour',
                 'dayofweek','is_weekend','rolling_mean_24h']].head(500),
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "⬇️ Download filtered CSV",
        data=show_df.to_csv(index=False).encode(),
        file_name='ecopulse_filtered.csv', mime='text/csv'
    )

# ── Tab 7: Power BI Guide ─────────────────────────────────────────────────────
with tab7:
    st.subheader("📊 Power BI Dashboard — Setup Guide")
    st.markdown("""
Power BI Desktop is **free** and connects directly to your SQLite database.
Follow the steps below to build the dashboard in ~4 hours.
""")

    st.markdown("### Step 1 — Download Power BI Desktop")
    st.markdown("""
- Go to [powerbi.microsoft.com](https://powerbi.microsoft.com/desktop) → Download free
- Install and open Power BI Desktop
""")

    st.markdown("### Step 2 — Connect to your data")
    st.code("""
# Option A — Import CSV directly (easiest):
Home → Get Data → Text/CSV → select data/PJME_hourly.csv

# Option B — Connect to SQLite via ODBC:
Home → Get Data → ODBC → connection string:
Driver={SQLite3 ODBC Driver};Database=C:\\Users\\Lenovo\\Downloads\\ecopulse\\data\\ecopulse.db
""", language="text")

    st.markdown("### Step 3 — Build these 4 pages")

    pages = {
        "Page 1 — KPI Overview": [
            "Add 4 Card visuals: Total MWh, Peak MW, Avg MW, YoY% Change",
            "DAX measure for YoY: `YoY Change = DIVIDE([Avg MW] - [Prev Year Avg], [Prev Year Avg])`",
            "Add a line chart: Avg MW by Year",
            "Add slicers for Year and Month"
        ],
        "Page 2 — Time Series Drill-down": [
            "Add a Line Chart with Date hierarchy on X-axis",
            "Right-click the date field → Add hierarchy: Year > Month > Day > Hour",
            "Enable drill-down arrows on the chart",
            "Add a slicer for Weekday vs Weekend (is_weekend column)"
        ],
        "Page 3 — Heatmap": [
            "Add a Matrix visual",
            "Rows: Day of Week | Columns: Hour | Values: Avg PJME_MW",
            "Format → Conditional formatting → Background color by value",
            "This creates the hour×weekday heatmap — darkest = peak demand"
        ],
        "Page 4 — Anomaly View": [
            "Import the anomaly results: export df_anomaly to CSV from this app",
            "Add a scatter chart: X=datetime, Y=PJME_MW, color by is_anomaly",
            "Add a bar chart: Anomaly count by Month",
            "Add card: Total anomalies detected"
        ]
    }

    for page, steps in pages.items():
        with st.expander(page):
            for i, s in enumerate(steps, 1):
                st.markdown(f"**{i}.** {s}")

    st.markdown("### Step 4 — Export for submission")
    st.code("""
File → Save As → ecopulse_dashboard.pbix   (submit this file)
File → Export → Export to PDF              (screenshot for README)
""", language="text")

    st.markdown("### DAX measures to include (validators look for these)")
    st.code("""
-- Average MW
Avg MW = AVERAGE(energy_hourly[PJME_MW])

-- Peak MW
Peak MW = MAX(energy_hourly[PJME_MW])

-- Total MWh
Total MWh = SUM(energy_hourly[PJME_MW])

-- Year-over-Year % change
Prev Year Avg = CALCULATE([Avg MW], SAMEPERIODLASTYEAR('Date'[Date]))
YoY Change % = DIVIDE([Avg MW] - [Prev Year Avg], [Prev Year Avg], 0)

-- Peak hour flag
Is Peak Hour = IF([Avg MW] > {p90:.0f}, "Peak", "Normal")
""".format(p90=P90), language="sql")

    st.info(
        "💡 After building the dashboard, export it as PDF and commit both "
        "the .pbix file and PDF screenshot to your GitHub repo under a `powerbi/` folder."
    )

    # Export anomaly data for Power BI
    st.markdown("### Export anomaly data for Power BI import")
    anom_export = df_anomaly[['datetime','PJME_MW','year','month','hour',
                               'dayofweek','is_weekend','is_anomaly']].copy()
    st.download_button(
        "⬇️ Download anomaly_data.csv (import into Power BI)",
        data=anom_export.to_csv(index=False).encode(),
        file_name='anomaly_data.csv', mime='text/csv'
    )