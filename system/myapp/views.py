from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import os
import pandas as pd
from .lstm import StockPredictor
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def get_available_tickers():
    """Get list of available tickers from major_exchanges.csv"""
    current_dir = Path(__file__).resolve().parent
    csv_path = os.path.join(current_dir, 'data', 'major_exchanges.csv')
    df = pd.read_csv(csv_path)
    tickers = df[['ticker', 'company name', 'exchange']].to_dict('records')
    return tickers

@api_view(['GET'])
def available_tickers(request):
    """Return list of available tickers"""
    tickers = get_available_tickers()
    return Response(tickers)

@api_view(['POST'])
def train_model(request):
    """Train LSTM model for selected ticker"""
    ticker = request.data.get('ticker')
    if not ticker:
        return Response({'error': 'Ticker is required'}, status=400)
    
    try:
        predictor = StockPredictor(
            ticker_symbol=ticker,
            neptune_project=os.getenv('NEPTUNE_PROJECT'),
            neptune_api_token=os.getenv('NEPTUNE_API_TOKEN'),
            run_name=f"{ticker}_prediction"
        )
        
        # Run prediction pipeline
        predictor.fetch_data()
        X_train, y_train = predictor.prepare_data()
        predictor.initialize_neptune()
        predictor.build_model(X_train.shape)
        predictor.train_model(X_train, y_train)
        predictions = predictor.make_predictions()
        rmse, mape = predictor.evaluate_model()
        # predictor.plot_predictions()
        model_path = predictor.save_model()
        predictor.run.stop()
        
        return Response({
            'status': 'success',
            'ticker': ticker,
            'rmse': float(rmse),
            'mape': float(mape)
        })
        
    except Exception as e:
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=500)
