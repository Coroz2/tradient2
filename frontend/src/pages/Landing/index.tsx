import React from 'react';
import { useAuth } from '../../utils/AuthContext';
import { useNavigate } from 'react-router-dom';
import { useEffect } from 'react';
import { supabase } from '../../utils/supabaseClient';
import StockPredictionSection from '../../components/StockPredictionSection';

function Landing() {
  const { user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (user) {
      navigate('/dashboard');
    }
  }, [user, navigate]);

  const handleSignInWithGoogle = async () => {
    try {
      const { data, error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/dashboard`,
        }
      });
      if (error) throw error;
    } catch (error) {
      console.error('Error signing in with Google:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 to-purple-600">
      <div className="container mx-auto px-4 py-8">
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
              <button 
                onClick={handleSignInWithGoogle}
                className="bg-indigo-500 text-white font-semibold py-2 px-4 rounded hover:bg-indigo-600 transition duration-300 flex items-center gap-2"
              >
                <img 
                  src="https://www.google.com/favicon.ico" 
                  alt="Google" 
                  className="w-4 h-4"
                />
                Sign in with Google
              </button>
            </div>
          </div>
        </div>

        <StockPredictionSection />
      </div>
    </div>
  );
}

export default Landing;
