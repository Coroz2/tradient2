import { createClient } from '@supabase/supabase-js';

const supabaseUrl = "https://qlrwrvfvpkykpcmikvdo.supabase.co";
const supabaseKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFscndydmZ2cGt5a3BjbWlrdmRvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAxNzAwNjQsImV4cCI6MjA0NTc0NjA2NH0.K31NQ0nDXA4hWS5rwI5LgVHh7tFFJWhZuQHktyDfYJg";

export const supabase = createClient(supabaseUrl, supabaseKey);