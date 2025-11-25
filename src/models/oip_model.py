
from __future__ import annotations
from logging import config

import numpy as np
import pandas as pd
from dataclasses import dataclass
from ..utils.config import Config
from typing import Optional, Literal, Dict, Any
from numpy.linalg import pinv
from sklearn.preprocessing import StandardScaler

class OIPCalculator:
    
    """
    """     
    
    FEATURES_BY_MACRO_POS = {
        "Goalkeeper": [
            "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
            "goalkeeping_positioning", "goalkeeping_reflexes", "goalkeeping_speed",
            "height_cm", "weight_kg", "power_jumping"
        ],
        "Centre-Back": [
            "defending_marking_awareness", "defending_standing_tackle",
            "defending_sliding_tackle", "mentality_interceptions", "power_strength",
            "power_jumping", "movement_acceleration", "movement_sprint_speed",
            "movement_reactions", "movement_balance"
        ],
        "Midfielder": [
            "passing", "attacking_short_passing", "skill_long_passing",
            "mentality_vision", "mentality_composure", "dribbling",
            "skill_ball_control", "movement_agility", "movement_balance"
        ],
        "Forward": [
            "attacking_finishing", "attacking_volleys", "shooting", "pace", 
            "dribbling", "mentality_positioning", "mentality_composure",
            "attacking_heading_accuracy"
        ]
    }
    
      
    def __init__(
        self,
        config: Config | None = None,
        logger=None
    ):
        self.scaler = None
        self.mu = None
        self.cov_inv = None
        self.cfg = Config()
        self.cfg.ensure_dirs()
        self.logger = logger

    
    def map_macro_position(self, positions_list):
        """
        Dada una lista de posiciones (['LM', 'CAM', 'LW']), devuelve la macro-posición:
        goalkeeper / defense / midfield / attack
        siguiendo una jerarquía futbolística.
        """
        pos_list = [p.strip() for p in positions_list]

        # 1) Porteros
        if "GK" in pos_list:
            return "goalkeeper"

        # 2) Defensas
        defense_pos = {"CB", "LB", "RB", "LWB", "RWB"}
        if any(p in defense_pos for p in pos_list):
            return "defense"

        # 3) Atacantes (tienen prioridad sobre mediocentros híbridos)
        attack_pos = {"ST", "CF", "LW", "RW", "LF", "RF"}
        if any(p in attack_pos for p in pos_list):
            return "attack"

        # 4) Mediocentros
        midfield_pos = {"CDM", "CM", "CAM", "RM", "LM"}
        if any(p in midfield_pos for p in pos_list):
            return "midfield"

        # Fallback
        return "midfield"
    
    def map_fifa_positions_to_role(self, positions_str: str) -> str | None:
        """
        Map FIFA 'player_positions' string (e.g. 'ST, CF, RW') to a high-level role:
        Goalkeeper, Centre-Back, Full-Back, Midfielder, Winger, Forward
        """
        if not isinstance(positions_str, str) or positions_str.strip() == "":
            return None

        # Normalize
        positions = [p.strip().upper() for p in positions_str.split(",")]

        # Priority order: GK -> CB -> FB -> MF -> W -> FW
        # (para evitar ambigüedades, miramos en orden)
        
        # 1) Goalkeeper
        if "GK" in positions:
            return "Goalkeeper"

        # 2) Centre-Back (central defenders)
        cb_set = {"CB", "LCB", "RCB"}
        if any(p in cb_set for p in positions):
            return "Centre-Back"

        # 3) Full-Back (laterales, carrileros)
        fb_set = {"LB", "RB", "LWB", "RWB"}
        if any(p in fb_set for p in positions):
            return "Full-Back"

        # 4) Winger (extremos y wide AM)
        winger_set = {"LW", "RW", "LM", "RM", "LAM", "RAM"}
        if any(p in winger_set for p in positions):
            return "Winger"

        # 5) Forward (delanteros centro / segunda punta)
        fw_set = {"ST", "CF", "LS", "RS", "LF", "RF"}
        if any(p in fw_set for p in positions):
            return "Forward"

        # 6) Midfielder (resto de mediocampistas)
        mf_set = {"CDM", "LDM", "RDM", "CM", "LCM", "RCM", "CAM"}
        if any(p in mf_set for p in positions):
            return "Midfielder"

        # Si algo raro, devolvemos None y luego podemos decidir qué hacer
        return None


    
    def clean_contract_dates(self, df: pd.DataFrame, initial_date: str) -> pd.DataFrame:
        """
        Clean and transform contract data.

        Steps:
        - Ensure 'contract_expiration' is converted to a proper datetime.
        - Remove players whose contracts expired before `initial_date`.
        - Compute months_to_expiry from `initial_date`.
        - Return a cleaned dataframe.

        Args:
            df (pd.DataFrame): Input dataframe containing 'contract_expiration'.
            initial_date (str): Reference date (e.g., "2024-06-30").

        Returns:
            pd.DataFrame: Cleaned dataframe with 'months_to_expiry' added.
        """

        df = df.copy()

        # 1) Convert initial_date to Timestamp
        start = pd.to_datetime(initial_date, errors="coerce")

        if start is None:
            raise ValueError("Invalid initial_date format. Must be a valid date string.")

        # 2) Ensure contract_expiration is a datetime
        df["contract_expiration"] = pd.to_datetime(df["contract_expiration"], errors="coerce")

        # Drop rows with invalid or missing contract_expiration
        df = df[df["contract_expiration"].notna()].copy()

        # 3) Remove contracts expired before initial_date
        df = df[df["contract_expiration"] >= start].copy()

        # 4) Compute remaining months
        df["months_to_expiry"] = (df["contract_expiration"] - start).dt.days / 30.44

        df.reset_index(drop=True, inplace=True)
        return df




    def player_matches_positions(positions_list, target_positions):
        return any(pos in target_positions for pos in positions_list)

    # -----------------------------
    # 1) PERFIL DEL EQUIPO
    # -----------------------------
    
    def fit_team_profile(self, df_team):

        df = df_team.copy()

        # # asegurar macro_pos
        # if "macro_position" not in df.columns:
        #     # Usar positions_list si ya existe (preprocessed)
        #     if "positions_list" not in df.columns:
        #         df["positions_list"] = df[self.role_column].astype(str).apply(
        #             lambda s: [p.strip() for p in s.split(",")]
        #         )
        #     df["macro_position"] = df["positions_list"].apply(self.map_macro_position)
        df["macro_position"] = df["player_positions"].apply(self.map_fifa_positions_to_role)


        # filtrar por macro_position
        if self.macro_position is not None:
            df = df[df["macro_position"] == self.macro_position].copy()
            if df.empty:
                raise ValueError(f"No hay jugadores del equipo en macro-posición {self.macro_position}")

        # guardar macro para el dataframe de candidatos
        #self.macro_position = self.macro_position

        # Selección dinámica de features
        self.feature_cols = self.FEATURES_BY_MACRO_POS.get(self.macro_position)
        if self.feature_cols is None:
            raise ValueError("macro_position inválido.")

        # Filtrar solo columnas disponibles en df
        self.feature_cols = [c for c in self.feature_cols if c in df.columns]

        df_num = df[self.feature_cols].copy()
        df_num = df_num.fillna(df_num.mean())

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df_num.values)

        self.mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        self.cov_inv = pinv(cov)


    # -----------------------------
    # 2) VECTORES Y DISTANCIAS
    # -----------------------------
    def _player_vector(self, row: pd.Series) -> np.ndarray:
        """
        Convierte una fila de jugador en vector numérico en el mismo espacio que el equipo.
        """
        if self.feature_cols is None or self.scaler is None:
            raise RuntimeError("Debes llamar antes a fit_team_profile().")

        vals = row[self.feature_cols].copy()

        if "market_value" in self.feature_cols:
            vals = vals.copy()
            vals.loc["market_value"] = np.log1p(vals["market_value"])

        x_scaled = self.scaler.transform([vals.values])[0]  # shape (n_features,)

        return x_scaled

    def mahalanobis_distance(self, x: np.ndarray) -> float:
        diff = x - self.mu
        return float(np.sqrt(diff.T @ self.cov_inv @ diff))
    
    def similarity_from_mahalanobis(self, d_M):
        # d_norm = d_M / (1 + d_M)
        d_norm = d_M / (1.0 + d_M)
        # similarity = 1 - d_norm = 1 / (1 + d_M)
        similarity = 1- d_norm
        
        return similarity

    # -----------------------------
    # 3) FÓRMULA BASE DE OIP
    # -----------------------------
    import numpy as np

    def growth_factor(self, v_current: float, v_future: float) -> float:
        """
        growth piecewise:
        - jóvenes: (Vf - Vc) / Vc
        - seniors: log((1+Vf)/(1+Vc))
        """

        if v_current <= 0:
            # por seguridad, si el valor es 0 o raro, asumimos crecimiento neutro
            return 0.0

        if v_future is None or np.isnan(v_future):
            return 0.0
        
        return float(np.log1p(v_future) - np.log1p(v_current))

        # if age <= self.age_cutoff:
        #     # fórmula original (porcentaje de subida)
        #     return (v_future - v_current) / v_current
        # else:
        #     # crecimiento logarítmico (más suave)
        #     return float(np.log1p(v_future) - np.log1p(v_current))
    
    def calculate_oip_single(self, v_current: float, v_future: float, theta: float) -> float:
        growth = self.growth_factor(v_current, v_future)
        urgency = max(0.0, 1.0 - theta / self.t_window)
        return growth * urgency

    

    # -----------------------------
    # 4) OIP POR JUGADOR (MULTI-HORIZONTE)
    # -----------------------------
    
    def calculate_oip_player(self, row: pd.Series) -> dict:
        x = self._player_vector(row)
        d = self.mahalanobis_distance(x)
        similarity = self.similarity_from_mahalanobis(d)

        v_current = float(row["market_value"])
        theta = float(row["months_to_expiry"])
        #age = float(row[self.age_column])

        col = "future_market_value"
        if col not in row or pd.isna(row[col]):
            return {"oip": None}

        v_future = float(row[col])

        base_oip = self.calculate_oip_single(v_current, v_future, theta)
        oip_final = base_oip * similarity *10

        return {"oip": oip_final}


    # -----------------------------
    # 5) OIP PARA UN DATAFRAME DE CANDIDATOS
    # -----------------------------
    
    def calculate_oip_dataframe(self, df_players):
        df = df_players.copy()

        # ---------------------------------------------------------
        # 1. Asegurar posiciones como lista
        # ---------------------------------------------------------
        # Usar positions_list si ya existe (preprocessed)
        # if "positions_list" not in df.columns:
        #     df["positions_list"] = df[self.role_column].astype(str).apply(
        #         lambda s: [p.strip() for p in s.split(",")]
        #     )

        # ---------------------------------------------------------
        # 2. Mapear macro-posición automáticamente
        # ---------------------------------------------------------
        # if "macro_position" not in df.columns:
        #     df["macro_position"] = df["positions_list"].apply(self.map_macro_position)
        df["macro_position"] = df["player_positions"].apply(self.map_fifa_positions_to_role)


        # ---------------------------------------------------------
        # 3. Filtrar por la macro-posición adecuada
        # ---------------------------------------------------------
        if hasattr(self, "macro_position") and self.macro_position is not None:
            df = df[df["macro_position"] == self.macro_position].copy()

            if df.empty:
                raise ValueError(f"No hay jugadores candidatos en macro-posición {self.macro_position}")

        # ---------------------------------------------------------
        # 4. Calcular OIP para cada jugador filtrado
        # ---------------------------------------------------------
        result_rows = []
        for _, row in df.iterrows():
            result_rows.append(self.calculate_oip_player(row))

        df_oip = pd.DataFrame(result_rows)

        # ---------------------------------------------------------
        # 5. Unir resultados
        # ---------------------------------------------------------
        df_final = pd.concat([df.reset_index(drop=True), df_oip], axis=1)


        # ---------------------------------------------------------
        # 5. OIP >0
        # ---------------------------------------------------------
        df_final = df_final[df_final["oip"] > 0].copy()


        
        # ---------------------------------------------------------
        # 6. Ordenar por criterios relevantes
        # ---------------------------------------------------------
        df_sorted = df_final.sort_values(
            by=["oip", "months_to_expiry", "market_value"],
            ascending=[False, True, True]
        )

        return df_sorted.reset_index(drop=True)

    def calculate_oip_team_past(self,
                           df: pd.DataFrame,
                           initial_date: str,
                           team_id: int,
                           t_window: float = 12,
                            feature_cols=None,
                            role_column: str = "player_positions",
                            age_column: str = "player_age",
                            age_cutoff: int = 22,
                            macro_position = None,
                            league: str | None = None,
                            nationality: str | None = None,
                            min_value: float | None = None,
                            max_value: float | None = None,
                            min_age: int | None = None,
                            max_age: int | None = None,
                           ) -> pd.DataFrame:
        """
        Calculate the OIP (Optimal Investment Profile) for all players outside a given team.

        Steps:
        - Extract the year from initial_date and filter the dataset by that year.
        - Clean Sofifa contract dates using initial_date.
        - Split into team players vs candidate players.
        - Fit team profile.
        - Compute OIP for all candidate players.

        Args:
            df (pd.DataFrame): Full Sofifa-like dataset.
            initial_date (str): Reference date such as "2017-06-30".
            team_id (int): Team ID to compute OIP against.

        Returns:
            pd.DataFrame: OIP scores for all candidate players.
        """
        self.macro_position = None
        self.t_window = t_window
        self.role_column = role_column
        self.age_column = age_column
        self.age_cutoff = age_cutoff
        self.macro_position = macro_position
        self.feature_cols = feature_cols
        
        df_players = df.copy()

        # 1) Extract year from initial_date
        year = pd.to_datetime(initial_date, errors="coerce").year
        if pd.isna(year):
            raise ValueError("Invalid initial_date format. Provide a valid date string.")
        # 2) Filter dataset to that year
        df_players = df_players[df_players["year"] == year].copy()


        # 3) Clean contract dates
        df_players = self.clean_contract_dates(df_players, initial_date=initial_date)

        # 4) Separate team players vs candidates
        df_team = df_players[df_players["club_team_id"] == team_id].copy()

        # Fix: compare column to a value, not to a DataFrame
        df_candidates = df_players[df_players["club_team_id"] != team_id].copy()

        # 5) Fit team profile
        self.fit_team_profile(df_team)

        # 6) Compute OIP for candidates
        df_results = self.calculate_oip_dataframe(df_candidates)

        return df_results.reset_index(drop=True)
    
    def calculate_oip_team(
            self,
            df: pd.DataFrame,
            initial_date: str,
            team_id: int,
            t_window: float = 12,
            feature_cols=None,
            role_column: str = "player_positions",
            macro_position=None,
            league: str | None = None,
            nationality: str | None = None,
            min_value: float | None = None,
            max_value: float | None = None,
            min_age: int | None = None,
            max_age: int | None = None
        ) -> pd.DataFrame:

        # Guardar parámetros en self
        self.t_window = t_window
        self.role_column = role_column
        self.macro_position = macro_position
        self.feature_cols = feature_cols

        df_players = df.copy()

        # 1) Filtrar por año
        year = pd.to_datetime(initial_date, errors="coerce").year
        if pd.isna(year):
            raise ValueError("Invalid initial_date format. Provide a valid date string.")

        df_players = df_players[df_players["year"] == year].copy()

        # 2) Limpiar fecha de contrato
        df_players = self.clean_contract_dates(df_players, initial_date=initial_date)

        # =============================================================
        # 3) FILTROS OPCIONALES
        # =============================================================

        # Filtro por valor mínimo
        if min_value is not None:
            df_players = df_players[df_players["market_value"] >= min_value].copy()

        # Filtro por valor máximo
        if max_value is not None:
            df_players = df_players[df_players["market_value"] <= max_value].copy()

        # Filtro por edad mínima
        if min_age is not None:
            df_players = df_players[df_players["player_age"] >= min_age].copy()

        # Filtro por edad máxima
        if max_age is not None:
            df_players = df_players[df_players["player_age"] <= max_age].copy()

        # =============================================================
        # 4) Separar jugadores del equipo y candidatos
        # =============================================================
        df_team = df_players[df_players["club_team_id"] == team_id].copy()
        
                # Filtro por liga
        if league is not None:
            df_players = df_players[df_players["league_name"] == league].copy()

        # Filtro por nacionalidad
        if nationality is not None:
            df_players = df_players[df_players["nationality_name"] == nationality].copy()
        
        
        df_candidates = df_players[df_players["club_team_id"] != team_id].copy()

        # 5) Fit del perfil
        self.fit_team_profile(df_team)

        # 6) Calcular OIP
        df_results = self.calculate_oip_dataframe(df_candidates)

        return df_results.reset_index(drop=True)


