"""
Basic functionality tests for the Free Transfer Opportunity Model
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.oip_model import OIPModel
from utils.config import Config

class TestOIPModel(unittest.TestCase):
    """Test cases for OIP model functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.oip_model = OIPModel(t_window=12)
        
        # Create test player data
        self.test_player = {
            'current_market_value': 10000000,  # 10M EUR
            'future_market_value': 15000000,   # 15M EUR
            'months_to_expiration': 6          # 6 months
        }
        
        self.test_players = [
            {
                'name': 'Player A',
                'current_market_value': 10000000,
                'future_market_value': 15000000,
                'months_to_expiration': 6
            },
            {
                'name': 'Player B',
                'current_market_value': 5000000,
                'future_market_value': 4000000,
                'months_to_expiration': 18
            },
            {
                'name': 'Player C',
                'current_market_value': 20000000,
                'future_market_value': 25000000,
                'months_to_expiration': 3
            }
        ]
    
    def test_oip_calculation(self):
        """Test OIP calculation for a single player"""
        oip = self.oip_model.calculate_oip(self.test_player)
        
        # Expected calculation:
        # value_growth_potential = (15M - 10M) / 10M = 0.5
        # urgency_factor = 1 - (6 / 12) = 0.5
        # OIP = 0.5 * 0.5 = 0.25
        expected_oip = 0.25
        
        self.assertAlmostEqual(oip, expected_oip, places=4)
    
    def test_oip_batch_calculation(self):
        """Test OIP calculation for multiple players"""
        results = self.oip_model.calculate_oip_batch(self.test_players)
        
        self.assertEqual(len(results), 3)
        
        # Check that OIP values are added
        for result in results:
            self.assertIn('oip', result)
            self.assertIsInstance(result['oip'], float)
    
    def test_ranking(self):
        """Test player ranking by OIP"""
        # Calculate OIP for test players
        players_with_oip = self.oip_model.calculate_oip_batch(self.test_players)
        
        # Rank players
        ranked_players = self.oip_model.rank_players_by_oip(players_with_oip)
        
        # Check ranking order (should be descending)
        oip_values = [player['oip'] for player in ranked_players]
        self.assertEqual(oip_values, sorted(oip_values, reverse=True))
    
    def test_categorization(self):
        """Test opportunity categorization"""
        # Calculate OIP for test players
        players_with_oip = self.oip_model.calculate_oip_batch(self.test_players)
        
        # Categorize opportunities
        categories = self.oip_model.get_opportunity_categories(players_with_oip)
        
        # Check that all players are categorized
        total_categorized = sum(len(cat) for cat in categories.values())
        self.assertEqual(total_categorized, len(self.test_players))
    
    def test_dataframe_calculation(self):
        """Test OIP calculation with DataFrame"""
        # Create test DataFrame
        df = pd.DataFrame(self.test_players)
        
        # Calculate OIP
        df_with_oip = self.oip_model.calculate_oip_dataframe(df)
        
        # Check that OIP column is added
        self.assertIn('oip', df_with_oip.columns)
        
        # Check that all rows have OIP values
        self.assertEqual(len(df_with_oip), len(self.test_players))
        self.assertFalse(df_with_oip['oip'].isna().any())
    
    def test_validation(self):
        """Test player data validation"""
        # Valid player data
        self.assertTrue(self.oip_model.validate_player_data(self.test_player))
        
        # Invalid player data (missing field)
        invalid_player = self.test_player.copy()
        del invalid_player['current_market_value']
        self.assertFalse(self.oip_model.validate_player_data(invalid_player))
        
        # Invalid player data (negative value)
        invalid_player = self.test_player.copy()
        invalid_player['current_market_value'] = -1000000
        self.assertFalse(self.oip_model.validate_player_data(invalid_player))
    
    def test_edge_cases(self):
        """Test edge cases in OIP calculation"""
        # Player with no value growth
        no_growth_player = {
            'current_market_value': 10000000,
            'future_market_value': 10000000,
            'months_to_expiration': 6
        }
        oip = self.oip_model.calculate_oip(no_growth_player)
        self.assertEqual(oip, 0.0)
        
        # Player with contract expiring very soon
        urgent_player = {
            'current_market_value': 10000000,
            'future_market_value': 15000000,
            'months_to_expiration': 1
        }
        oip = self.oip_model.calculate_oip(urgent_player)
        # urgency_factor should be close to 1
        self.assertGreater(oip, 0.4)
    
    def test_config_parameters(self):
        """Test configuration parameters"""
        # Test with different T_window
        oip_model_custom = OIPModel(t_window=6)
        
        player_data = {
            'current_market_value': 10000000,
            'future_market_value': 15000000,
            'months_to_expiration': 3
        }
        
        oip_default = self.oip_model.calculate_oip(player_data)
        oip_custom = oip_model_custom.calculate_oip(player_data)
        
        # Should be different due to different T_window
        self.assertNotEqual(oip_default, oip_custom)

class TestConfig(unittest.TestCase):
    """Test configuration settings"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Should not raise an exception
        self.assertTrue(Config.validate_config())
    
    def test_data_paths(self):
        """Test data path generation"""
        paths = Config.get_data_paths()
        
        self.assertIn('players', paths)
        self.assertIn('valuations', paths)
        self.assertIn('contracts', paths)
        
        # Check that paths are strings
        for path in paths.values():
            self.assertIsInstance(path, str)

if __name__ == '__main__':
    unittest.main() 