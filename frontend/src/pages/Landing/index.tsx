import React from 'react';
import StockPredictionChart from '../../components/StockPredictionChart';
import { useState, useEffect } from 'react';
import axios from 'axios';

interface Ticker {
  ticker: string;
  'company name': string;
  exchange: string;
}

function Landing() {
  const [tickers, setTickers] = useState<Ticker[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<string>('AAPL');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string>('');
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  useEffect(() => {
    axios.get(`${process.env.REACT_APP_API_URL}/api/available-tickers/`)
    .then(response => {
      setTickers(response.data);
      // console.log('Loaded tickers:', response.data);
    })
      .catch(err => setError('Failed to load tickers'));
  }, []);

  const handleTrainModel = async () => {
    if (!selectedTicker) return;
    
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/train-model/`, {
        ticker: selectedTicker
      });
      setResult(response.data);
      setRefreshTrigger(prev => prev + 1);
    } catch (err) {
      setError('Failed to train model');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 to-purple-600">
      <div className="container mx-auto px-4 py-8">
        {/* First white box with title and Get Started button */}
        <div className="w-full bg-white rounded-xl shadow-lg overflow-hidden mb-8">
          <div className="p-8">
            <div className="uppercase tracking-wide text-sm text-indigo-500 font-semibold">
              Welcome to
            </div>
            <h1 className="block mt-1 text-3xl leading-tight font-bold text-black">
              Stock Prediction App
            </h1>
            <p className="mt-2 text-slate-500">
              A modern application for predicting stock market trends using advanced analytics
            </p>
            <div className="mt-6">
              <button className="bg-indigo-500 text-white font-semibold py-2 px-4 rounded hover:bg-indigo-600 transition duration-300">
                Get Started
              </button>
            </div>
          </div>
        </div>

        {/* Second white box with ticker selector and chart */}
        <div className="w-full bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="p-8">
            {/* Ticker selector and train button */}
            <div className="mb-6 flex gap-4 items-center">
              <select 
                value={selectedTicker}
                onChange={(e) => setSelectedTicker(e.target.value)}
                className="border border-gray-300 p-2 rounded-md flex-grow max-w-md"
              >
                <option value="">Select a ticker</option>
                {tickers.map(ticker => (
                  <option key={ticker.ticker} value={ticker.ticker}>
                    {ticker.ticker} - {ticker['company name']}
                  </option>
                ))}
              </select>
              <button
                onClick={handleTrainModel}
                disabled={!selectedTicker || loading}
                className="bg-indigo-500 text-white font-semibold py-2 px-4 rounded hover:bg-indigo-600 transition duration-300 disabled:bg-gray-400 flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                    Training...
                  </>
                ) : (
                  'Train Model'
                )}
              </button>
            </div>

            {/* Error and Results display */}
            {error && (
              <div className="mb-4 text-red-500 bg-red-50 p-3 rounded">
                {error}
              </div>
            )}
            {result && (
              <div className="mb-4 bg-green-50 p-3 rounded">
                <h3 className="font-bold text-green-800">Model Training Results:</h3>
                <p className="text-green-700">RMSE: {result.rmse.toFixed(2)}</p>
                <p className="text-green-700">MAPE: {result.mape.toFixed(2)}%</p>
              </div>
            )}

            {/* Stock Prediction Chart */}
            <StockPredictionChart 
              selectedTicker={selectedTicker} 
              refreshTrigger={refreshTrigger} 
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Landing;
