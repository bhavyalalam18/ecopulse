"""
EcoPulse Setup — Run this ONCE before launching the Streamlit app.
Executes: data pipeline → SQLite load → model training → Excel audit
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 55)
    print("  EcoPulse — Full Pipeline Setup")
    print("=" * 55)

    # Step 1: Data pipeline
    print("\n[STEP 1/3] Running data pipeline...")
    from utils.data_pipeline import run_pipeline
    df = run_pipeline()
    print(f"  ✅ Data pipeline complete. {len(df):,} rows in SQLite.")

    # Step 2: Train models
    print("\n[STEP 2/3] Training ML models (this takes 2-5 minutes)...")
    from utils.train_model import train
    _, results = train()
    print(f"  ✅ Models trained and saved to models/")

    # Step 3: Excel audit
    print("\n[STEP 3/3] Generating Excel audit workbook...")
    from utils.excel_audit import generate_excel_audit
    path = generate_excel_audit()
    print(f"  ✅ Excel audit saved: {path}")

    print("\n" + "=" * 55)
    print("  ✅ Setup complete!")
    print("  To launch the app, run:")
    print("      streamlit run app.py")
    print("=" * 55)


if __name__ == '__main__':
    main()
