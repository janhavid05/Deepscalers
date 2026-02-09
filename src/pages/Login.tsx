import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Login: React.FC = () => {
  const navigate = useNavigate();
  const { sendVerificationCode, login, loading, error, isAuthenticated } = useAuth();
  
  const [phone, setPhone] = useState('');
  const [verificationCode, setVerificationCode] = useState('');
  const [codeSent, setCodeSent] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  // If already authenticated, redirect to home
  useEffect(() => {
    console.log("Login page - isAuthenticated changed:", isAuthenticated);
    if (isAuthenticated) {
      console.log("Login page - redirecting to home due to authentication");
      navigate('/', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  const handleSendCode = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    
    if (!phone) {
      setFormError('Please enter your phone number');
      return;
    }

    try {
      await sendVerificationCode(phone);
      setCodeSent(true);
    } catch (error) {
      // Error is already set in the auth context
      console.error('Error sending verification code:', error);
    }
  };

  const handleVerifyCode = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    
    if (!verificationCode) {
      setFormError('Please enter the verification code');
      return;
    }

    try {
      console.log("Attempting to verify code for phone:", phone);
      await login(phone, verificationCode);
      console.log("Login successful, should redirect automatically");
      
      // For safety, also attempt manual navigation
      navigate('/', { replace: true });
    } catch (error) {
      console.error('Login error:', error);
      // If user doesn't exist, redirect to registration page
      if (error instanceof Error && error.message.includes('user_exists_false')) {
        console.log("User doesn't exist, redirecting to registration");
        navigate('/register', { state: { phone } });
      }
    }
  };
  
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Student Assistant
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            {codeSent ? 'Enter the verification code sent to your phone' : 'Sign in with your phone number'}
          </p>
        </div>
        
        {error && (
          <div className="rounded-md bg-red-50 p-4">
            <div className="flex">
              <div className="ml-3">
                <p className="text-sm font-medium text-red-800">{error}</p>
              </div>
            </div>
          </div>
        )}
        
        {formError && (
          <div className="rounded-md bg-red-50 p-4">
            <div className="flex">
              <div className="ml-3">
                <p className="text-sm font-medium text-red-800">{formError}</p>
              </div>
            </div>
          </div>
        )}
        
        {!codeSent ? (
          <form className="mt-8 space-y-6" onSubmit={handleSendCode}>
            <div className="rounded-md shadow-sm -space-y-px">
              <div>
                <label htmlFor="phone-number" className="sr-only">Phone Number</label>
                <input
                  id="phone-number"
                  name="phone"
                  type="tel"
                  required
                  className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                  placeholder="Phone Number"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                />
              </div>
            </div>

            <div>
              <button
                type="submit"
                disabled={loading}
                className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                {loading ? 'Sending...' : 'Send Verification Code'}
              </button>
            </div>
          </form>
        ) : (
          <form className="mt-8 space-y-6" onSubmit={handleVerifyCode}>
            <div className="rounded-md shadow-sm -space-y-px">
              <div>
                <label htmlFor="verification-code" className="sr-only">Verification Code</label>
                <input
                  id="verification-code"
                  name="code"
                  type="text"
                  required
                  className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                  placeholder="Verification Code"
                  value={verificationCode}
                  onChange={(e) => setVerificationCode(e.target.value)}
                />
              </div>
            </div>

            <div>
              <button
                type="submit"
                disabled={loading}
                className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                {loading ? 'Verifying...' : 'Verify Code'}
              </button>
            </div>
            
            <div className="text-center">
              <button
                type="button"
                onClick={() => setCodeSent(false)}
                className="font-medium text-indigo-600 hover:text-indigo-500"
              >
                Use a different phone number
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

export default Login; 