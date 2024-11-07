from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import os
import pandas as pd
from .lstm import StockPredictor
from dotenv import load_dotenv
from pathlib import Path
import traceback
import logging
from rest_framework import status

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
        return Response(
            {'error': 'Ticker symbol is required'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        predictor = StockPredictor(
            ticker_symbol=ticker,
            neptune_project=os.getenv('NEPTUNE_PROJECT'),
            neptune_api_token=os.getenv('NEPTUNE_API_TOKEN'),
            run_name=f"{ticker}_prediction"
        )
        
        try:
            predictor.fetch_data()
        except ValueError as e:
            return Response({
                'status': 'error',
                'message': str(e),
                'error_type': 'data_fetch_error'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Continue with the rest of the pipeline
        X_train, y_train = predictor.prepare_data()
        predictor.initialize_neptune(["POST"])
        predictor.build_model(X_train.shape)
        predictor.train_model(X_train, y_train)
        predictions = predictor.make_predictions()
        rmse, mape = predictor.evaluate_model()
        model_path = predictor.save_model()
        predictor.run.stop()
        
        return Response({
            'status': 'success',
            'ticker': ticker,
            'rmse': float(rmse),
            'mape': float(mape)
        })
        
    except Exception as e:
        logging.error(f"Error training model: {traceback.format_exc()}")
        return Response({
            'status': 'error',
            'message': str(e),
            'error_type': 'training_error'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
