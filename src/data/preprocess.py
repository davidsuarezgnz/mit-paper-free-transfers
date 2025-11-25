# src/data/preprocess.py

import numpy as np
import pandas as pd

from pathlib import Path
from src.data.clean import DataCleaner
from sklearn.linear_model import LinearRegression

class DataPreprocessor:
    """Class to get the final features for the model"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.cleaner = DataCleaner()        
    
    def prepare_features(self) -> pd.DataFrame:
        """
        Get the final features for the VM model
        
        Returns:
            DataFrame with the final features for the model
        """

        # ===============================
        # 1. Data Getter 
        # ===============================
        print("\nPreprocess 1. Loading and cleaning data...")
        df_players = self.cleaner.clean_players()
        
        # ===============================
        # 2. # Split positions
        # ===============================

        print("\nPreprocess 2. Split positions")
        # df_players["positions_list"] = (
        #     df_players["player_positions"]
        #     .astype(str)
        #     .apply(lambda s: [p.strip() for p in s.split(",")] )
        # )


        
        # # ===============================
        # # 3. Convert contract year to date
        # # ===============================
        print("\nPreprocess 3. Convert contract year to date ")
        df_players = self.convert_contract_year_to_date(df_players)
        
 
        # # ===============================
        # # 4. Predict next year valuation
        # # ===============================
        print("\nPreprocess 4. Predict next year valuation")
        df_final = self.predict_next_year_value(df_players)
            
            
        # ===============================
        # 6. Save the final dataset into /data/processed
        # ===============================
        print("\nPreprocess 6. Saving preprocessed dataset...")
        self.save_processed(
            df                      = df_final,
            base_name               = "player_contract_valuations",
            save_parquet            = False,
            save_csv                = True,
            versioned_parquet       = False
        )

        # ===============================
        # 7. Return the final dataset
        # ===============================
        return df_final

    
    # ===============================
    # Auxiliar methods 
    # ===============================   

    def save_processed(
        self,
        df: pd.DataFrame,
        base_name: str = "player_contracts_valuations",
        save_parquet: bool = False,
        save_csv: bool = True,
        versioned_parquet: bool = False,
    ) -> None:
        """
        Save the processed DataFrame into /data/processed directory.
        - Saves as Parquet ('latest' + optional timestamped version)
        - Saves as CSV ('latest')
        """
        # Get project root: src/data/preprocess.py -> src -> project root
        project_root = Path(__file__).resolve().parents[2]
        processed_dir = project_root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        if save_parquet:
            latest_parquet = processed_dir / f"{base_name}.parquet"
            try:
                df.to_parquet(latest_parquet, index=False)
                print(f"[OK] Saved Parquet: {latest_parquet}")
                if versioned_parquet:
                    versioned = processed_dir / f"{base_name}_{ts}.parquet"
                    df.to_parquet(versioned, index=False)
                    print(f"[OK] Saved versioned Parquet: {versioned}")
            except Exception as e:
                print(f"[WARN] Failed to save Parquet ({e}). "
                    f"Install pyarrow or fastparquet if needed.")

        if save_csv:
            latest_csv = processed_dir / f"{base_name}.csv"
            df.to_csv(latest_csv, index=False)
            print(f"[OK] Saved CSV: {latest_csv}")
    
    def convert_contract_year_to_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a contract year column ('club_contract_valid_until')
        into a real expiration date ('contract_expiration') using the format YYYY-06-30.

        Args:
            df_players (pd.DataFrame): Player dataframe containing 'club_contract_valid_until'.

        Returns:
            pd.DataFrame: Updated dataframe with a new 'contract_expiration' datetime column.

        Raises:
            ValueError: If the required column is missing.
        """
        
        df_players = df.copy()

        if "club_contract_valid_until" not in df_players.columns:
            raise ValueError("Column 'club_contract_valid_until' not found in dataframe")

        df_players = df_players.copy()

        # Convert year column to numeric
        df_players["club_contract_valid_until"] = pd.to_numeric(
            df_players["club_contract_valid_until"], 
            errors="coerce"
        )

        # Build actual date (June 30th of the given year)
        df_players["contract_expiration"] = pd.to_datetime(
            df_players["club_contract_valid_until"].astype("Int64").astype(str) + "-06-30",
            errors="coerce"
        )

        return df_players



    def predict_next_year_value(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """
        Para cada jugador y para cada año disponible,
        estima el valor del año siguiente usando SOLO datos hasta ese año.

        Devuelve el dataframe original + columna future_market_value.
        """

        df = df_all.copy()
        df["year"] = df["year"].astype(int)

        df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")

        # Añadir columna vacía donde guardaremos las predicciones
        df["future_market_value"] = np.nan

        # Procesar por jugador
        for pid, grp in df.groupby("player_id"):

            # Ordenar por año ascendentemente
            grp = grp.sort_values("year")

            for i, row in grp.iterrows():

                year_current = row["year"]
                year_next = year_current + 1

                # Datos históricos hasta el año actual (inclusive)
                grp_hist = grp[grp["year"] <= year_current]

                # Si solo hay un dato → fallback (no se puede ajustar regresión)
                if len(grp_hist) == 1:
                    predicted = grp_hist["market_value"].iloc[0]

                else:
                    # Regresión lineal sobre datos históricos
                    X = grp_hist["year"].values.reshape(-1, 1)
                    y = grp_hist["market_value"].values

                    model = LinearRegression()
                    model.fit(X, y)

                    predicted = model.predict(np.array([[year_next]]))[0]

                predicted = max(predicted, 0)  # no permitir valores negativos

                # Guardar predicción en la fila correspondiente del DF original
                df.loc[(df["player_id"] == pid) & (df["year"] == year_current),
                    "future_market_value"] = predicted

        return df
