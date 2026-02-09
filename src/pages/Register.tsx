import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

interface LocationState {
  phone?: string;
}

const Register: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { register, loading, error } = useAuth();
  const { phone = '' } = (location.state as LocationState) || {};

  const [formData, setFormData] = useState({
    phone_number: phone,
    password: '',
    confirm_password: '',
    first_name: '',
    last_name: '',
    email: '',
  });
  
  const [formError, setFormError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    
    // Validate form data
    if (!formData.phone_number) {
      setFormError('Phone number is required');
      return;
    }
    
    if (!formData.password) {
      setFormError('Password is required');
      return;
    }
    
    if (formData.password !== formData.confirm_password) {
      setFormError('Passwords do not match');
      return;
    }
    
    try {
      // Register the user
      await register({
        phone_number: formData.phone_number,
        password: formData.password,
        first_name: formData.first_name || undefined,
        last_name: formData.last_name || undefined,
        email: formData.email || undefined,
      });
      
      // Redirect to main dashboard after successful registration
      navigate('/');
    } catch (error) {
      // Error is already set in the auth context
    }
  };
  
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Create your account
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Please fill in your details to complete registration
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
        
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="rounded-md shadow-sm -space-y-px">
            <div className="mb-4">
              <label htmlFor="phone_number" className="block text-sm font-medium text-gray-700 mb-1">
                Phone Number
              </label>
              <input
                id="phone_number"
                name="phone_number"
                type="tel"
                required
                className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                placeholder="Phone Number"
                value={formData.phone_number}
                onChange={handleChange}
                readOnly={!!phone}
              />
            </div>
            
            <div className="mb-4">
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                placeholder="Password"
                value={formData.password}
                onChange={handleChange}
              />
            </div>
            
            <div className="mb-4">
              <label htmlFor="confirm_password" className="block text-sm font-medium text-gray-700 mb-1">
                Confirm Password
              </label>
              <input
                id="confirm_password"
                name="confirm_password"
                type="password"
                required
                className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                placeholder="Confirm Password"
                value={formData.confirm_password}
                onChange={handleChange}
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="mb-4">
                <label htmlFor="first_name" className="block text-sm font-medium text-gray-700 mb-1">
                  First Name
                </label>
                <input
                  id="first_name"
                  name="first_name"
                  type="text"
                  className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  placeholder="First Name"
                  value={formData.first_name}
                  onChange={handleChange}
                />
              </div>
              
              <div className="mb-4">
                <label htmlFor="last_name" className="block text-sm font-medium text-gray-700 mb-1">
                  Last Name
                </label>
                <input
                  id="last_name"
                  name="last_name"
                  type="text"
                  className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  placeholder="Last Name"
                  value={formData.last_name}
                  onChange={handleChange}
                />
              </div>
            </div>
            
            <div className="mb-4">
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                Email (optional)
              </label>
              <input
                id="email"
                name="email"
                type="email"
                className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                placeholder="Email"
                value={formData.email}
                onChange={handleChange}
              />
            </div>
          </div>

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              {loading ? 'Registering...' : 'Register'}
            </button>
          </div>
          
          <div className="text-center">
            <div className="text-sm">
              <span className="text-gray-500">Already have an account? </span>
              <button
                type="button"
                onClick={() => navigate('/login')}
                className="font-medium text-indigo-600 hover:text-indigo-500"
              >
                Sign in
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Register; 