import React from 'react';
import StockPredictionSection from '../../components/StockPredictionSection';
import { useAuth } from '../../utils/AuthContext';
import { useNavigate } from 'react-router-dom';
import { useEffect } from 'react';
import { supabase } from '../../utils/supabaseClient';

function Dashboard() {
  const navigate = useNavigate();
  const { user, profile } = useAuth();

  useEffect(() => {
    if (!user) {
      navigate('/');
    }
  }, [user, navigate]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 to-purple-600">
      <div className="container mx-auto px-4 py-8">
        {/* User Profile Section */}
        <div className="w-full bg-white rounded-xl shadow-lg overflow-hidden mb-8">
          <div className="p-8 flex items-center gap-4">
            {profile?.avatar_url && (
              <img 
                src={profile.avatar_url} 
                alt="Profile" 
                className="w-12 h-12 rounded-full"
              />
            )}
            <div>
              <h2 className="text-xl font-semibold">{profile?.full_name}</h2>
              <p className="text-gray-600">{profile?.email}</p>
            </div>
            <button 
              onClick={() => supabase.auth.signOut()}
              className="ml-auto bg-indigo-500 text-white font-semibold py-2 px-4 rounded hover:bg-indigo-600 transition duration-300 flex items-center gap-2"
            >
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                className="h-4 w-4" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" 
                />
              </svg>
              Sign Out
            </button>
          </div>
        </div>
        <StockPredictionSection />
      </div>
    </div>
  );
}

export default Dashboard;