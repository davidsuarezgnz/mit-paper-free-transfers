# src/utils/config.py

from __future__ import annotations

import os

from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field, asdict


def _project_root() -> Path:
    """Infer project root as two levels up from this file."""
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Centralized configuration for paths, features and training parameters."""

    # --- Paths ---
    project_root: Path = field(default_factory=_project_root)
    data_processed: Path = field(init=False)
    models_dir: Path = field(init=False)
    features_csv: Path = field(init=False)
    last_vals_csv: Path = field(init=False)
    evaluation_dir: Path = field(init=False)

    # --- Columns / features ---

    base_features_columns = ["market_value", "player_age", "gross_salary_per_year"]
    horizons = [6, 12, 24, 36]


    # --- Filenames (relative to data_processed by default) ---
    features_filename: str = "player_contract_valuations.csv"
    last_vals_filename: str = "latest_valuations.csv"

    def __post_init__(self) -> None:
        """Finalize derived paths after base fields are set."""
        self.data_processed = self.project_root / "data" / "processed"
        self.models_dir = self.project_root / "models_saved"
        self.evaluation_dir = self.project_root  / "evaluation" / "outputs"
        self.features_csv = self.data_processed / self.features_filename
        self.last_vals_csv = self.data_processed / self.last_vals_filename


    @classmethod
    def from_env(cls) -> "Config":
        """
        Optional: create Config reading overrides from environment variables.
        Useful for CI or different machines without changing code.
        Supported env vars:
          - WINNING_PROJECT_ROOT
          - WINNING_FEATURES_FILE
          - WINNING_TEST_SIZE
          - WINNING_RANDOM_STATE
        """
        cfg = cls()
        # Root
        env_root = os.getenv("WINNING_PROJECT_ROOT")
        if env_root:
            cfg.project_root = Path(env_root).expanduser().resolve()
            cfg.data_processed = cfg.project_root / "data" / "processed"
            cfg.models_dir = cfg.project_root / "models_saved"
            cfg.features_csv = cfg.data_processed / cfg.features_filename
            cfg.last_vals_csv = cfg.data_processed / cfg.last_vals_filename

        # Features file
        env_features = os.getenv("WINNING_FEATURES_FILE")
        if env_features:
            p = Path(env_features).expanduser()
            cfg.features_csv = p if p.is_absolute() else (cfg.project_root / env_features)
            
        # Last Vals file
        env_lastvals = os.getenv("WINNING_LASTVALS_FILE")
        if env_lastvals:
            p = Path(env_lastvals).expanduser()
            cfg.last_vals_csv = p if p.is_absolute() else (cfg.project_root / env_lastvals)

        # Basic training params
        if (ts := os.getenv("WINNING_TEST_SIZE")) is not None:
            try:
                cfg.test_size = float(ts)
            except ValueError:
                pass

        return cfg

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.data_processed.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        d = asdict(self)
        # Convert Paths to strings for JSON-friendliness
        for k in ("project_root", "data_processed", "models_dir", "features_csv", "last_vals_csv"):
            d[k] = str(d[k])
        return d
