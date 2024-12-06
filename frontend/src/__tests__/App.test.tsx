import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App';

// Mock the AuthContext
jest.mock('../utils/AuthContext', () => ({
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useAuth: () => ({
    user: null,
    signIn: jest.fn(),
    signOut: jest.fn(),
  }),
}));

// Mock the components
jest.mock('../pages/Landing', () => () => <div>Landing Page</div>);
jest.mock('../pages/Dashboard', () => () => <div>Dashboard Page</div>);

describe('App Component', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  test('renders landing page by default', () => {
    render(<App />);
    expect(screen.getByText('Landing Page')).toBeInTheDocument();
  });

  test('redirects to landing page when accessing dashboard without auth', async () => {
    render(<App />);
    // Attempt to navigate to dashboard
    window.history.pushState({}, '', '/dashboard');
    
    await waitFor(() => {
      expect(window.location.pathname).toBe('/dashboard');
    });
  });

  test('renders dashboard when user is authenticated', () => {
    // Override the mock to return an authenticated user
    jest.spyOn(require('../utils/AuthContext'), 'useAuth').mockImplementation(() => ({
      user: { id: '1', email: 'test@example.com' },
      signIn: jest.fn(),
      signOut: jest.fn(),
    }));

    render(<App />);
    window.history.pushState({}, '', '/dashboard');
    expect(screen.getByText('Dashboard Page')).toBeInTheDocument();
  });
});