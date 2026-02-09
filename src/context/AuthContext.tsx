import React, { createContext, useContext, useState, useEffect } from 'react';
import { API_CONFIG } from '../config';

interface User {
  id: number;
  phone_number: string;
  first_name: string;
  last_name: string;
  email: string;
  is_verified: boolean;
}

interface AuthContextType {
  isAuthenticated: boolean;
  user: User | null;
  loading: boolean;
  error: string | null;
  login: (phone_number: string, code: string) => Promise<void>;
  sendVerificationCode: (phone_number: string) => Promise<void>;
  register: (userData: RegisterData) => Promise<void>;
  logout: () => void;
}

interface RegisterData {
  phone_number: string;
  first_name?: string;
  last_name?: string;
  email?: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check if user is logged in on component mount
    const checkAuthStatus = async () => {
      const token = localStorage.getItem('access_token');
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.AUTH.PROFILE}`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (response.ok) {
          const userData = await response.json();
          setUser(userData);
          setIsAuthenticated(true);
        } else {
          // Token might be expired
          console.log('Auth check failed. Removing tokens.');
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
        }
      } catch (error) {
        console.error('Authentication check failed:', error);
      } finally {
        setLoading(false);
      }
    };

    checkAuthStatus();
  }, []);

  const sendVerificationCode = async (phone_number: string) => {
    setLoading(true);
    setError(null);

    // Debug info
    console.log('Sending verification to phone:', phone_number);
    console.log('Using API URL:', `${API_CONFIG.BASE_URL}${API_CONFIG.AUTH.SEND_VERIFICATION}`);

    try {
      console.log('Attempting fetch request...');
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.AUTH.SEND_VERIFICATION}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ phone_number }),
      });

      console.log('Fetch response status:', response.status);
      
      if (!response.ok) {
        let errorMessage = 'Failed to send verification code';
        try {
          const errorData = await response.json();
          errorMessage = errorData.message || errorMessage;
        } catch (parseError) {
          console.error('Error parsing error response:', parseError);
          // Try to get text response
          const errorText = await response.text();
          console.error('Error response text:', errorText);
        }
        throw new Error(errorMessage);
      }

      console.log('Verification code sent successfully');
    } catch (error) {
      console.error('Send verification error details:', error);
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const login = async (phone_number: string, code: string) => {
    setLoading(true);
    setError(null);

    try {
      // First verify the code
      const verifyResponse = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.AUTH.VERIFY_CODE}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ phone_number, code }),
      });

      if (!verifyResponse.ok) {
        const errorData = await verifyResponse.json();
        throw new Error(errorData.message || 'Invalid verification code');
      }

      const verifyData = await verifyResponse.json();
      console.log('Verification response:', verifyData);

      if (verifyData.user_exists) {
        // User exists, save tokens and set auth state
        console.log('User authenticated successfully. Setting tokens.');
        localStorage.setItem('access_token', verifyData.access);
        localStorage.setItem('refresh_token', verifyData.refresh);
        
        // Set the authentication state
        setUser(verifyData.user);
        setIsAuthenticated(true);
        
        // Log the successful authentication
        console.log('Authentication state updated. isAuthenticated:', true);
      } else {
        // User needs to register
        console.log('User not found, needs to register');
        throw new Error('user_exists_false');
      }
    } catch (error) {
      if (error instanceof Error && error.message === 'user_exists_false') {
        throw error;
      }
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const register = async (userData: RegisterData) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.AUTH.REGISTER}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Registration failed');
      }

      const data = await response.json();
      
      // Save tokens
      localStorage.setItem('access_token', data.access);
      localStorage.setItem('refresh_token', data.refresh);
      
      // Set auth state
      setUser(data.user);
      setIsAuthenticated(true);
      
      return data;
    } catch (error) {
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    setUser(null);
    setIsAuthenticated(false);
  };

  const contextValue = {
    isAuthenticated,
    user,
    loading,
    error,
    login,
    sendVerificationCode,
    register,
    logout
  };

  console.log('Auth Context State:', { isAuthenticated, loading, hasUser: !!user });

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}; 