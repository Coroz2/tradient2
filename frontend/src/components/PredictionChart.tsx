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

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

interface PredictionChartProps {
    predictions: {
        historical: number[];
        predicted: number[];
        dates: string[];
    };
}

export default function PredictionChart({ predictions }: PredictionChartProps) {
    const data = {
        labels: predictions.dates,
        datasets: [
            {
                label: 'Historical',
                data: predictions.historical,
                borderColor: '#22c55e',
                backgroundColor: 'rgba(34, 197, 94, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 8,
                spanGaps: true
            },
            {
                label: 'Predicted',
                data: predictions.predicted,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 8,
                spanGaps: true,
                borderDash: [5, 5]
            }
        ]
    };

    const options = {
        responsive: true,
        interaction: {
            intersect: false,
            mode: 'index' as const
        },
        scales: {
            y: {
                beginAtZero: false,
                ticks: {
                    callback: function(value: number | string) {
                        return `$${Number(value).toFixed(2)}`;
                    }
                }
            }
        },
        plugins: {
            legend: {
                position: 'top' as const
            },
            tooltip: {
                callbacks: {
                    label: (context: any) => `$${context.parsed.y.toFixed(2)}`
                }
            }
        }
    };

    return (
        <div className="w-full h-full">
            <Line data={data} options={options} />
        </div>
    );
}
