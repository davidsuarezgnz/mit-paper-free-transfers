from __future__ import annotations

import sys
import pandas as pd

from pathlib import Path
from src.utils.config import Config
from src.models.oip_model import OIPCalculator

# ===============================
# Preprocess auxiliar functions
# ===============================
def _dir_is_empty(p: Path) -> bool:
    """Return True if path doesn't exist or exists and has no files"""
    if not p.exists():
        return True
    return next(p.iterdir(), None) is None


def _run_preprocess() -> None:
    """
    Run preprocessing pipeline to generate the processed features file.
    It calls DataPreprocessor.prepare_features(), which is expected to
    write 'player_valuations_features.csv' into data/processed
    """
    # Import here to avoid importing heavy deps if not needed
    from src.data.preprocess import DataPreprocessor

    print(">> /data/processed is empty: running preprocessing pipeline...")
    pre = DataPreprocessor()
    df_processed = pre.prepare_features()
    print(f">> Preprocessing finished. Produced dataset with shape: {df_processed.shape}")

# ===============================
# Main definition
# ===============================
def main():
    cfg = Config()
    cfg.ensure_dirs()

    # 1) Check if /data/processed is empty
    if _dir_is_empty(cfg.data_processed):
        try:
            _run_preprocess()
        except Exception as e:
            # Fail fast with a clear message if preprocessing fails
            print("Preprocessing failed:", repr(e))
            sys.exit(1)
    else:
        print(">> /data/processed is not empty: skipping preprocessing.")
    
    # 2) Read the features CSV (expected to exist after preprocessing)
    print(f">> Reading dataset from: {cfg.features_csv}")
    if not cfg.features_csv.exists():
        print(f"Expected features file not found: {cfg.features_csv}")
        print("Check preprocessing saved 'player_contract_valuations.csv' in data/processed/")
        sys.exit(1)

    df = pd.read_csv(cfg.features_csv)
    print(f"Dataset ready: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 3) Train segmented models
    oipCal = OIPCalculator(cfg)
    print(">> Calculating OIP")
    final_df  = oipCal.calculate_oip_team(df=df, team_id= 241.0, initial_date= "2021-01-01", t_window=18,  macro_position="Goalkeeper", min_value=20000000, min_age=23)
    
    print("\nSummary and models saved at:")
    print(" -", cfg.models_dir)
    print(final_df)

# ===============================
# Main execution
# ===============================
if __name__ == "__main__":
    main()
