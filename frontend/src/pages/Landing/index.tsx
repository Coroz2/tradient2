import React from 'react';
import StockPredictionChart from '../../components/StockPredictionChart';

function Landing() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 to-purple-600">
      <div className="container mx-auto px-4 py-8">
        {/* First white box with title and button */}
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

        {/* Second white box with chart */}
        <div className="w-full bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="p-8">
            <StockPredictionChart />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Landing;
