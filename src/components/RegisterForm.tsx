import React, { useState } from 'react';
import { FiUser, FiMail, FiArrowLeft, FiArrowRight } from 'react-icons/fi';
import { useAuth } from '../context/AuthContext';
import { createStudent } from '../lib/studentOperations';

interface RegisterFormProps {
  phone: string;
  onSuccess: () => void;
  onCancel: () => void;
}

const RegisterForm: React.FC<RegisterFormProps> = ({ phone, onSuccess, onCancel }) => {
  const { register, loading, error } = useAuth();
  
  const [formData, setFormData] = useState({
    phone_number: phone,
    first_name: '',
    last_name: '',
    email: '',
    student_id: '',
    department: '',
    year_of_study: '1st Year',
    current_semester: '',
  });
  
  const [formError, setFormError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
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
    
    if (!formData.student_id) {
      setFormError('Student ID is required');
      return;
    }
    
    try {
      // First register the user in the auth system
      const authData = await register({
        phone_number: formData.phone_number,
        first_name: formData.first_name || undefined,
        last_name: formData.last_name || undefined,
        email: formData.email || undefined,
      });

      // Then create the student record in Supabase
      const { data: studentData, error: studentError } = await createStudent({
        phone_number: formData.phone_number,
        email: formData.email || undefined,
        first_name: formData.first_name || undefined,
        last_name: formData.last_name || undefined,
        student_id: formData.student_id,
        department: formData.department || undefined,
        year_of_study: formData.year_of_study as any,
        current_semester: formData.current_semester || undefined,
        is_verified: false,
        is_active: true,
      });

      if (studentError) {
        throw studentError;
      }
      
      // Call onSuccess callback on successful registration
      onSuccess();
    } catch (error) {
      console.error('Registration error:', error);
      setFormError(error instanceof Error ? error.message : 'Registration failed');
    }
  };
  
  return (
    <div className="w-full max-w-md mx-auto">
      <form onSubmit={handleSubmit} className="space-y-6">
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
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="first_name" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              First Name
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-gray-400 dark:text-gray-500">
                <FiUser size={18} />
              </div>
              <input
                id="first_name"
                name="first_name"
                type="text"
                className="pl-10 block w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 sm:text-sm transition-all duration-200"
                placeholder="First name"
                value={formData.first_name}
                onChange={handleChange}
              />
            </div>
          </div>
          
          <div>
            <label htmlFor="last_name" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Last Name
            </label>
            <input
              id="last_name"
              name="last_name"
              type="text"
              className="block w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 sm:text-sm transition-all duration-200"
              placeholder="Last name"
              value={formData.last_name}
              onChange={handleChange}
            />
          </div>
        </div>

        <div>
          <label htmlFor="student_id" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Student ID
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-gray-400 dark:text-gray-500">
              <FiUser size={18} />
            </div>
            <input
              id="student_id"
              name="student_id"
              type="text"
              required
              className="pl-10 block w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 sm:text-sm transition-all duration-200"
              placeholder="Enter your student ID"
              value={formData.student_id}
              onChange={handleChange}
            />
          </div>
        </div>

        <div>
          <label htmlFor="department" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Department
          </label>
          <input
            id="department"
            name="department"
            type="text"
            className="block w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 sm:text-sm transition-all duration-200"
            placeholder="Enter your department"
            value={formData.department}
            onChange={handleChange}
          />
        </div>

        <div>
          <label htmlFor="year_of_study" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Year of Study
          </label>
          <select
            id="year_of_study"
            name="year_of_study"
            className="block w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 sm:text-sm transition-all duration-200"
            value={formData.year_of_study}
            onChange={handleChange}
          >
            <option value="1st Year">1st Year</option>
            <option value="2nd Year">2nd Year</option>
            <option value="3rd Year">3rd Year</option>
            <option value="4th Year">4th Year</option>
            <option value="Graduate">Graduate</option>
          </select>
        </div>

        <div>
          <label htmlFor="current_semester" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Current Semester
          </label>
          <input
            id="current_semester"
            name="current_semester"
            type="text"
            className="block w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 sm:text-sm transition-all duration-200"
            placeholder="Enter your current semester"
            value={formData.current_semester}
            onChange={handleChange}
          />
        </div>
        
        <div>
          <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Email (optional)
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-gray-400 dark:text-gray-500">
              <FiMail size={18} />
            </div>
            <input
              id="email"
              name="email"
              type="email"
              className="pl-10 block w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 sm:text-sm transition-all duration-200"
              placeholder="Enter your email"
              value={formData.email}
              onChange={handleChange}
            />
          </div>
        </div>
        
        <div className="flex gap-4">
          <button
            type="button"
            onClick={onCancel}
            className="flex-1 flex items-center justify-center py-3 px-4 border border-gray-300 dark:border-gray-700 rounded-lg shadow-sm text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 transition-colors duration-200 group"
          >
            <FiArrowLeft className="mr-2 group-hover:-translate-x-1 transition-transform duration-200" size={16} />
            Back
          </button>
          <button
            type="submit"
            disabled={loading}
            className="flex-1 flex items-center justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-teal-600 hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 group"
          >
            <span className="flex items-center">
              {loading ? 'Registering...' : 'Complete Registration'}
              <FiArrowRight className="ml-2 group-hover:translate-x-1 transition-transform duration-200" size={16} />
            </span>
          </button>
        </div>
      </form>
    </div>
  );
};

export default RegisterForm; 