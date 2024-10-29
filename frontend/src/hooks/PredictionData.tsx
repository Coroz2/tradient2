import { useState, useEffect } from 'react';
import { supabase } from '../utils/supabaseClient';

export interface PredictionData {
  ticker: string;
  date: string;
  actual_price: number;
  predicted_price: number;
}

export function usePredictions() {
  const [data, setData] = useState<PredictionData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchPredictions() {
      try {
        const { data: predictions, error } = await supabase
          .from('stock_predictions')
          .select('*')
          .order('date', { ascending: true });

        if (error) throw error;
        setData(predictions);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch predictions');
      } finally {
        setLoading(false);
      }
    }

    fetchPredictions();
  }, []);

  return { data, loading, error };
}