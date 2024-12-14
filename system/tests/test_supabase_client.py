from unittest import TestCase
from datetime import datetime
import pandas as pd
from myapp.supabase_client import save_predictions_to_supabase, supabase

class TestSupabaseClient(TestCase):
    def setUp(self):
        self.train_data = {
            "Close": {
                datetime(2024, 1, 1): 150.0,
                datetime(2024, 1, 2): 155.0
            }
        }
        self.test_data = {
            "Close": {
                datetime(2024, 1, 3): 160.0
            }
        }
        self.predicted_prices = pd.Series([159.0])
        self.ticker_symbol = "AAPL"

    def test_save_predictions(self):
        try:
            save_predictions_to_supabase(self.train_data, self.test_data, self.predicted_prices, self.ticker_symbol)

            result = supabase.table('stock_predictions').select("*").match({'ticker': self.ticker_symbol}).execute()
            inserted_data = result.data

            self.assertIsNotNone(inserted_data, "No data returned from Supabase")
            self.assertEqual(len(inserted_data), 3, "Number of records inserted is incorrect")

            expected_data = [
                {'ticker': 'AAPL', 'date': '2024-01-01T00:00:00', 'actual_price': 150.0, 'predicted_price': None, 'data_type': 'train'},
                {'ticker': 'AAPL', 'date': '2024-01-02T00:00:00', 'actual_price': 155.0, 'predicted_price': None, 'data_type': 'train'},
                {'ticker': 'AAPL', 'date': '2024-01-03T00:00:00', 'actual_price': 160.0, 'predicted_price': 159.0, 'data_type': 'test'}
            ]

            for i, row in enumerate(inserted_data):
                self.assertEqual(row['ticker'], expected_data[i]['ticker'])
                actual_date = row['date'].split('+')[0]  # Removes the '+00:00'
                self.assertEqual(actual_date, expected_data[i]['date'])
                self.assertEqual(row['actual_price'], expected_data[i]['actual_price'])
                self.assertEqual(row['predicted_price'], expected_data[i]['predicted_price'])
                self.assertEqual(row['data_type'], expected_data[i]['data_type'])

        except Exception as e:
            self.fail(f"save_predictions_to_supabase raised an exception: {e}")
