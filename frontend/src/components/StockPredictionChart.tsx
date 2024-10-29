import { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { supabase } from '../utils/supabaseClient';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface PredictionData {
  date: string;
  actual_price: number;
  predicted_price: number;
  ticker: string;
}

export default function StockPredictionChart() {
  const [predictionData, setPredictionData] = useState<PredictionData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const { data, error: supabaseError } = await supabase
          .from('stock_predictions')
          .select('*')
          .order('date', { ascending: true });

        if (supabaseError) throw supabaseError;
        setPredictionData(data || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, []);

  if (loading) return (
    <div className="flex justify-center items-center h-64">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
    </div>
  );
  
  if (error) return (
    <div className="text-red-500 text-center p-4 bg-red-50 rounded-lg">
      Error: {error}
    </div>
  );

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: '2-digit',
      day: '2-digit',
      year: '2-digit'
    }).replace(/\//g, '-');
  };

  const chartData = {
    labels: predictionData.map(d => formatDate(d.date)),
    datasets: [
      {
        label: 'Actual Price',
        data: predictionData.map(d => d.actual_price),
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.05)',
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointRadius: 0,
        pointHoverRadius: 8,
        pointBackgroundColor: '#ffffff',
        pointHoverBackgroundColor: '#ffffff',
        pointBorderColor: '#6366f1',
        pointHoverBorderColor: '#6366f1',
        pointBorderWidth: 2,
        pointHoverBorderWidth: 2,
        pointStyle: 'circle',
        hitRadius: 8,
      },
      {
        label: 'Predicted Price',
        data: predictionData.map(d => d.predicted_price),
        borderColor: '#f43f5e',
        backgroundColor: 'rgba(244, 63, 94, 0.05)',
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointRadius: 0,
        pointHoverRadius: 8,
        pointBackgroundColor: '#ffffff',
        pointHoverBackgroundColor: '#ffffff',
        pointBorderColor: '#f43f5e',
        pointHoverBorderColor: '#f43f5e',
        pointBorderWidth: 2,
        pointHoverBorderWidth: 2,
        pointStyle: 'circle',
        hitRadius: 8,
      }
    ]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          padding: 20,
          usePointStyle: true,
          pointStyle: 'circle',
          font: {
            size: 14,
            family: "'Inter', system-ui, sans-serif",
            weight: 'normal' as const
          }
        }
      },
      title: {
        display: true,
        text: `${predictionData[0]?.ticker || 'Stock'} Price Predictions`,
        font: {
          size: 18,
          family: "'Inter', system-ui, sans-serif",
          weight: 'bold' as const
        },
        padding: { top: 20, bottom: 24 }
      },
      tooltip: {
        enabled: true,
        backgroundColor: 'rgba(255, 255, 255, 0.98)',
        titleColor: '#1F2937',
        bodyColor: '#1F2937',
        borderColor: '#E5E7EB',
        borderWidth: 1,
        padding: 12,
        displayColors: true,
        usePointStyle: true,
        bodyFont: {
          size: 14,
          family: "'Inter', system-ui, sans-serif",
          weight: 'normal' as const
        },
        titleFont: {
          size: 14,
          family: "'Inter', system-ui, sans-serif",
          weight: 'normal' as const
        }
      }
    },
    scales: {
      y: {
        grid: {
          color: 'rgba(156, 163, 175, 0.1)',
          drawBorder: false
        },
        ticks: {
          callback: function(value: string | number) {
            if (typeof value === 'number') {
              return `$${value.toFixed(2)}`;
            }
            return value;
          },
          font: {
            size: 12,
            family: "'Inter', system-ui, sans-serif",
            weight: 'normal' as const
          }
        }
      },
      x: {
        grid: {
          display: false,
          drawBorder: false
        },
        ticks: {
          font: {
            size: 12,
            family: "'Inter', system-ui, sans-serif",
            weight: 'normal' as const
          },
          maxTicksLimit: 8,
          maxRotation: 0,
          autoSkip: true,
        }
      }
    }
  };


  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white rounded-xl shadow-sm">
      <Line data={chartData} options={options} className="p-4" />
    </div>
  );
}