import React, { useState, useEffect } from 'react';
import axios from 'axios';
import StockPredictionChart from './StockPredictionChart';

interface Ticker {
  ticker: string;
  'company name': string;
  exchange: string;
}

interface TrainingResult {
  rmse: number;
  mape: number;
}

interface CachedResults {
  [key: string]: TrainingResult;
}

function StockPredictionSection() {
  const [tickers, setTickers] = useState<Ticker[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<string>('AAPL');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [error, setError] = useState<string>('');
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [cachedResults, setCachedResults] = useState<CachedResults>({});

  useEffect(() => {
    axios.get(`${process.env.REACT_APP_API_URL}/api/available-tickers/`)
      .then(response => {
        setTickers(response.data);
      })
      .catch(err => setError('Failed to load tickers'));
  }, []);

  // Update result when ticker changes
  useEffect(() => {
    setResult(cachedResults[selectedTicker] || null);
  }, [selectedTicker, cachedResults]);

  const handleTrainModel = async () => {
    if (!selectedTicker) return;
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/train-model/`, {
        ticker: selectedTicker
      });
      
      // Cache the result for this ticker
      setCachedResults(prev => ({
        ...prev,
        [selectedTicker]: response.data
      }));
      setResult(response.data);
      setRefreshTrigger(prev => prev + 1);
    } catch (err) {
      setError('Failed to train model');
    } finally {
      setLoading(false);
    }
  };

  const handleTickerChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedTicker(e.target.value);
  };

  return (
    <div className="w-full bg-white rounded-xl shadow-lg overflow-hidden">
      <div className="p-8">
        <div className="mb-6 flex gap-4 items-center">
          <select 
            value={selectedTicker}
            onChange={handleTickerChange}
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

        <StockPredictionChart 
          selectedTicker={selectedTicker} 
          refreshTrigger={refreshTrigger} 
        />
      </div>
    </div>
  );
}

export default StockPredictionSection;