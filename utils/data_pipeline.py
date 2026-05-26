"""
EcoPulse Data Pipeline
Loads PJME CSV, engineers time-series features, and stores to SQLite.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ecopulse.db')
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'PJME_hourly.csv')


def get_engine():
    db_abs = os.path.abspath(DB_PATH)
    return create_engine(f'sqlite:///{db_abs}')


def load_and_clean_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Rename to standard names
    rename_map = {}
    for col in df.columns:
        if 'datetime' in col.lower() or 'date' in col.lower():
            rename_map[col] = 'datetime'
        elif 'mw' in col.lower() or 'pjme' in col.lower():
            rename_map[col] = 'PJME_MW'
    df = df.rename(columns=rename_map)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.dropna(subset=['PJME_MW'])
    df = df.drop_duplicates(subset=['datetime'])
    df = df[df['PJME_MW'] > 0]
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['year']        = df['datetime'].dt.year
    df['month']       = df['datetime'].dt.month
    df['day']         = df['datetime'].dt.day
    df['hour']        = df['datetime'].dt.hour
    df['dayofweek']   = df['datetime'].dt.dayofweek
    df['quarter']     = df['datetime'].dt.quarter
    df['is_weekend']  = (df['dayofweek'] >= 5).astype(int)
    df['dayofyear']   = df['datetime'].dt.dayofyear

    # Cyclical encoding (prevents model from thinking 23h and 0h are far apart)
    df['hour_sin']    = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']    = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin']   = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']   = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin']     = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos']     = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Lag features (requires sorted data)
    df['lag_1h']      = df['PJME_MW'].shift(1)
    df['lag_24h']     = df['PJME_MW'].shift(24)
    df['lag_168h']    = df['PJME_MW'].shift(168)   # 1 week

    # Rolling statistics
    df['rolling_mean_24h']  = df['PJME_MW'].shift(1).rolling(24).mean()
    df['rolling_std_24h']   = df['PJME_MW'].shift(1).rolling(24).std()
    df['rolling_mean_168h'] = df['PJME_MW'].shift(1).rolling(168).mean()

    df = df.dropna().reset_index(drop=True)
    return df


def save_to_sqlite(df: pd.DataFrame, engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS energy_hourly"))

    df.to_sql('energy_hourly', engine, index=False, if_exists='replace')

    with engine.begin() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_datetime ON energy_hourly (datetime)"
        ))
    print(f"Saved {len(df)} rows to SQLite.")


def run_pipeline() -> pd.DataFrame:
    print("Loading CSV...")
    raw = load_and_clean_csv(CSV_PATH)
    print(f"  Raw rows: {len(raw)}")

    print("Engineering features...")
    df = engineer_features(raw)
    print(f"  Feature rows (after lag dropna): {len(df)}")

    print("Saving to SQLite...")
    engine = get_engine()
    save_to_sqlite(df, engine)

    return df


if __name__ == '__main__':
    run_pipeline()
