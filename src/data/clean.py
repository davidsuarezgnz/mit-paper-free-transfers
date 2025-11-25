# src/data/clean_contracts.py

import pandas as pd
from pathlib import Path
from src.data.ingest import DataIngestor

class DataCleaner:
    """Clean and validate player contract data."""

    def __init__(self):
        """Initialize the data cleaner"""
        self.ingestor = DataIngestor()
        self.data_path = Path("data/clean")
        self.today = pd.Timestamp.today()

    # ===============================
    # Main Method
    # ===============================
    def clean_players(self) -> pd.DataFrame:
        """
        Full cleaning pipeline for players data.
        """
        players_dfs = {}

        for year in range(15, 23):   # años 15 a 22
            df = self.ingestor.get_players(year)
            players_dfs[year] = df

        # List to keep dataframes
        df_list = []

        for year, df in players_dfs.items():
            df_temp = df.copy()
            df_temp["year"] = 2000 + year
            df_list.append(df_temp)

        # Concat all data
        df_players_all = pd.concat(df_list, ignore_index=True)
        
        df_players_all = self.drop_innecesary_columns(df_players_all)
        
        # Rename columns
        df_players_all = df_players_all.rename(columns={
            "sofifa_id": "player_id",
            "value_eur": "market_value",
            "age": "player_age",
            "wage_eur": "gross_salary_per_year"
        })
        
        
        # Drop NaN values
        df_players_all = df_players_all.dropna(subset=["market_value", "gross_salary_per_year", "club_contract_valid_until"])
        
        
        return df_players_all


    
    # ===============================
    # Auxiliar methods 
    # ===============================

    def drop_innecesary_columns(self, df:  pd.DataFrame):
        
        df_to_drop = df.copy()
        
        df_to_drop = df_to_drop.drop(columns=["player_face_url","club_logo_url",
                                                "club_flag_url","nation_logo_url",
                                                "nation_flag_url", "international_reputation",
                                                "work_rate","body_type","real_face",
                                                "release_clause_eur","player_tags",
                                                "player_traits", "nationality_id",
                                                "nationality_name","nation_team_id",
                                                "nation_position","nation_jersey_number"])
        df_to_drop = df_to_drop.drop(columns=["ls","st","rs","lw","lf","cf","rf","rw","lam",
                                                "cam","ram","lm","lcm","cm","rcm","rm","lwb",
                                                "ldm","cdm","rdm","rwb","lb","lcb","cb","rcb","rb","gk"])
        
        df = df_to_drop.drop(columns=["player_url", "club_jersey_number", "club_loaned_from"])
        
        return df
        
        
