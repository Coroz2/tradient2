from supabase import create_client
from django.conf import settings
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Get Supabase credentials from environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials not found in environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_predictions_to_supabase(train_data, test_data, predicted_prices, ticker_symbol):
    predictions = []
    
    # Add training data
    for date, actual in train_data['Close'].items():
        predictions.append({
            'ticker': ticker_symbol,
            'date': date.isoformat(),
            'actual_price': float(actual),
            'predicted_price': None,  # No predictions for training data
            'data_type': 'train'
        })
    
    # Add test data with predictions
    for i, (date, actual) in enumerate(test_data['Close'].items()):
        predictions.append({
            'ticker': ticker_symbol,
            'date': date.isoformat(),
            'actual_price': float(actual),
            'predicted_price': float(predicted_prices.iloc[i]),  # Use iloc instead of direct indexing
            'data_type': 'test'
        })
    
    # Clear existing data for this ticker
    supabase.table('stock_predictions').delete().match({'ticker': ticker_symbol}).execute()
    
    # Insert new data
    result = supabase.table('stock_predictions').insert(predictions).execute()
    return result