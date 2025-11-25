"""
Data ingestion module for OIP analysis
"""

import pandas as pd
from pathlib import Path

# Setup paths and imports


class DataIngestor:
    """Class to load data from CSV files"""
    
    def __init__(self):
        """Initialize the data ingestor"""
        self.data_path = Path("data/raw")
    
    
    def get_players(self,num) -> pd.DataFrame:
        """
        Load players data from CSV
        
        Returns:
            DataFrame with players data
        """
        csv_path = self.data_path / f"players_{num}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")
            
        return pd.read_csv(csv_path)
    
