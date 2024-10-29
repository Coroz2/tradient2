from supabase import create_client
from django.conf import settings

supabase = create_client("https://qlrwrvfvpkykpcmikvdo.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFscndydmZ2cGt5a3BjbWlrdmRvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAxNzAwNjQsImV4cCI6MjA0NTc0NjA2NH0.K31NQ0nDXA4hWS5rwI5LgVHh7tFFJWhZuQHktyDfYJg")

def save_predictions_to_supabase(test_data, predicted_prices, ticker_symbol):
    predictions = []
    
    for i, (date, actual) in enumerate(test_data['Close'].items()):
        # Delete existing prediction for the same ticker and date
        supabase.table('stock_predictions').delete().match({'ticker': ticker_symbol, 'date': date.isoformat()}).execute()
        
        predictions.append({
            'ticker': ticker_symbol,
            'date': date.isoformat(),
            'actual_price': float(actual),
            'predicted_price': float(predicted_prices[i][0]),
        })
    
    result = supabase.table('stock_predictions').insert(predictions).execute()
    return result