import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
const supabaseKey = process.env.REACT_APP_SUPABASE_KEY;

console.log('Supabase URL:', supabaseUrl); // Debug log

if (!supabaseUrl || !supabaseKey) {
    throw new Error('Missing Supabase environment variables');
  }

export const supabase = createClient(supabaseUrl, supabaseKey);