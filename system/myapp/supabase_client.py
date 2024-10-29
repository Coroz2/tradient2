from supabase import create_client
from django.conf import settings

supabase = create_client("https://qlrwrvfvpkykpcmikvdo.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFscndydmZ2cGt5a3BjbWlrdmRvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAxNzAwNjQsImV4cCI6MjA0NTc0NjA2NH0.K31NQ0nDXA4hWS5rwI5LgVHh7tFFJWhZuQHktyDfYJg")

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