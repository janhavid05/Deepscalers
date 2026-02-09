import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { FiX, FiPhone, FiLock, FiArrowRight, FiShield, FiChevronDown, FiGlobe } from 'react-icons/fi';
import RegisterForm from './RegisterForm';

interface LoginModalProps {
  isOpen: boolean;
  onClose: () => void;
}

// OTP Input component for verification code
const OtpInput: React.FC<{
  length: number;
  value: string;
  onChange: (value: string) => void;
}> = ({ length, value, onChange }) => {
  const [otpValues, setOtpValues] = useState<string[]>(Array(length).fill(''));
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);

  // Initialize refs array
  useEffect(() => {
    inputRefs.current = inputRefs.current.slice(0, length);
  }, [length]);

  // Update OTP values when the parent value changes
  useEffect(() => {
    const valueArray = value.split('').slice(0, length);
    setOtpValues([...valueArray, ...Array(length - valueArray.length).fill('')]);
  }, [value, length]);

  // Handle input change
  const handleChange = (index: number, e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    
    // Only accept numbers
    if (!/^\d*$/.test(newValue)) return;
    
    // Use only the last entered digit if multiple digits are pasted
    const digit = newValue.slice(-1);
    
    // Update the OTP values
    const newOtpValues = [...otpValues];
    newOtpValues[index] = digit;
    setOtpValues(newOtpValues);
    
    // Call parent onChange with the full string
    onChange(newOtpValues.join(''));
    
    // Focus next input if we entered a digit and there's a next input
    if (digit && index < length - 1) {
      inputRefs.current[index + 1]?.focus();
    }
  };

  // Handle key press for navigation between inputs
  const handleKeyDown = (index: number, e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Backspace' && !otpValues[index] && index > 0) {
      // Focus previous input on backspace when current input is empty
      inputRefs.current[index - 1]?.focus();
    } else if (e.key === 'ArrowLeft' && index > 0) {
      // Focus previous input on left arrow
      inputRefs.current[index - 1]?.focus();
    } else if (e.key === 'ArrowRight' && index < length - 1) {
      // Focus next input on right arrow
      inputRefs.current[index + 1]?.focus();
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pastedData = e.clipboardData.getData('text/plain').trim();
    if (!/^\d+$/.test(pastedData)) return;
    
    const digits = pastedData.split('').slice(0, length);
    const newOtpValues = [...otpValues];
    
    digits.forEach((digit, idx) => {
      newOtpValues[idx] = digit;
    });
    
    setOtpValues(newOtpValues);
    onChange(newOtpValues.join(''));
    
    // Focus the input after the last pasted digit
    if (digits.length < length) {
      inputRefs.current[digits.length]?.focus();
    }
  };

  return (
    <div className="flex justify-center gap-3">
      {otpValues.map((digit, index) => (
        <div 
          key={index} 
          className="w-12 h-14 relative"
        >
          <input
            ref={(el) => (inputRefs.current[index] = el)}
            type="text"
            inputMode="numeric"
            pattern="\d*"
            maxLength={1}
            value={digit}
            onChange={(e) => handleChange(index, e)}
            onKeyDown={(e) => handleKeyDown(index, e)}
            onPaste={handlePaste}
            autoFocus={index === 0 && !digit}
            className="w-full h-full text-center font-bold text-xl bg-gray-50 dark:bg-gray-800 border-2 border-gray-300 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 hover:border-teal-400 transition-all duration-200 shadow-sm"
            aria-label={`digit ${index + 1}`}
          />
          {index < length - 1 && (
            <div className="absolute top-1/2 -right-3 transform -translate-y-1/2">
              <div className="w-1 h-1 rounded-full bg-gray-300 dark:bg-gray-700"></div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

// Country code selector component
const CountryCodeSelector: React.FC<{
  selectedCode: string;
  onSelect: (code: string) => void;
}> = ({ selectedCode, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  // Top frequently used country codes with flags
  const countryCodes = [
    { code: '+91', country: 'India', flag: '🇮🇳' },
    { code: '+1', country: 'USA', flag: '🇺🇸' },
    { code: '+44', country: 'UK', flag: '🇬🇧' },
    { code: '+61', country: 'Australia', flag: '🇦🇺' },
    { code: '+971', country: 'UAE', flag: '🇦🇪' },
    { code: '+86', country: 'China', flag: '🇨🇳' },
    { code: '+81', country: 'Japan', flag: '🇯🇵' },
    { code: '+49', country: 'Germany', flag: '🇩🇪' },
    { code: '+33', country: 'France', flag: '🇫🇷' },
    { code: '+7', country: 'Russia', flag: '🇷🇺' },
  ];

  // Find selected country info
  const selectedCountry = countryCodes.find(country => country.code === selectedCode) || countryCodes[0];

  return (
    <div className="relative">
      <button
        type="button"
        className="flex items-center justify-between min-w-[100px] h-[46px] px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-200"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Select country code"
      >
        <span className="mr-1 text-lg">{selectedCountry.flag}</span>
        <span className="font-medium">{selectedCode}</span>
        <FiChevronDown className={`ml-1 transition-transform duration-200 ${isOpen ? 'transform rotate-180' : ''}`} />
      </button>
      
      {isOpen && (
        <>
          <div 
            className="fixed inset-0 z-10" 
            onClick={() => setIsOpen(false)}
          ></div>
          <div className="absolute z-20 mt-1 w-64 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg max-h-60 overflow-auto">
            <div className="sticky top-0 bg-white dark:bg-gray-800 p-2 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
                <FiGlobe className="mr-2" />
                <span>Select Country Code</span>
              </div>
            </div>
            <div className="py-1">
              {countryCodes.map((country) => (
                <button
                  key={country.code}
                  type="button"
                  className={`flex items-center w-full text-left px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-150 ${selectedCode === country.code ? 'bg-gray-100 dark:bg-gray-700' : ''}`}
                  onClick={() => {
                    onSelect(country.code);
                    setIsOpen(false);
                  }}
                >
                  <span className="mr-2 text-lg">{country.flag}</span>
                  <span className="font-medium">{country.code}</span>
                  <span className="ml-2 text-gray-500 dark:text-gray-400 text-sm">{country.country}</span>
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

const LoginModal: React.FC<LoginModalProps> = ({ isOpen, onClose }) => {
  const { sendVerificationCode, login, loading, error } = useAuth();
  
  const [countryCode, setCountryCode] = useState('+91');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [verificationCode, setVerificationCode] = useState('');
  const [codeSent, setCodeSent] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [showRegister, setShowRegister] = useState(false);

  // Combine country code and phone number
  const phone = `${countryCode}${phoneNumber}`;

  const handleSendCode = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    
    if (!phoneNumber) {
      setFormError('Please enter your phone number');
      return;
    }

    try {
      await sendVerificationCode(phone);
      setCodeSent(true);
    } catch (error) {
      console.error('Error sending verification code:', error);
    }
  };

  const handleVerifyCode = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    
    if (!verificationCode || verificationCode.length < 6) {
      setFormError('Please enter the 6-digit verification code');
      return;
    }

    try {
      console.log("Attempting to verify code for phone:", phone);
      await login(phone, verificationCode);
      console.log("Login successful");
      onClose(); // Close the modal after successful login
    } catch (error) {
      console.error('Login error:', error);
      
      // Check if this is a "user not found" error
      if (error instanceof Error && error.message.includes('user_exists_false')) {
        console.log("User doesn't exist, showing registration form");
        setShowRegister(true);
      }
    }
  };
  
  const resetForm = () => {
    setShowRegister(false);
    setCodeSent(false);
    setCountryCode('+91');
    setPhoneNumber('');
    setVerificationCode('');
    setFormError(null);
  };
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70 backdrop-blur-sm">
      <div className="relative w-full max-w-md mx-auto rounded-2xl overflow-hidden shadow-2xl transform transition-all duration-300 animate-fadeIn">
        {/* Gradient header */}
        <div className="bg-gradient-to-r from-teal-500 to-teal-600 p-6 text-white">
          <button 
            onClick={onClose}
            className="absolute top-4 right-4 text-white hover:text-white/80 transition-colors"
          >
            <FiX size={24} />
          </button>
          
          <div className="mb-1">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-white/20 backdrop-blur-sm text-white mb-4">
              {showRegister ? (
                <FiShield size={24} />
              ) : codeSent ? (
                <FiLock size={24} />
              ) : (
                <FiPhone size={24} />
              )}
            </div>
            <h2 className="text-2xl font-bold">
              {showRegister 
                ? 'Create Account' 
                : codeSent 
                  ? 'Verify Code' 
                  : 'Welcome Back'}
            </h2>
            <p className="text-white/80 text-sm">
              {showRegister 
                ? 'Complete your profile to continue' 
                : codeSent 
                  ? `Enter the verification code sent to ${phone}` 
                  : 'Sign in to continue learning'}
            </p>
          </div>
        </div>
        
        {/* Content area with white/dark background */}
        <div className="bg-white dark:bg-gray-900 p-6">
          {error && (
            <div className="mb-4 rounded-lg bg-red-50 dark:bg-red-900/20 p-4 border-l-4 border-red-500 dark:border-red-500">
              <p className="text-sm font-medium text-red-800 dark:text-red-400">{error}</p>
            </div>
          )}
          
          {formError && (
            <div className="mb-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 p-4 border-l-4 border-amber-500 dark:border-amber-500">
              <p className="text-sm font-medium text-amber-800 dark:text-amber-400">{formError}</p>
            </div>
          )}
          
          {showRegister ? (
            <RegisterForm 
              phone={phone} 
              onSuccess={onClose} 
              onCancel={() => setShowRegister(false)} 
            />
          ) : (
            <>
              {!codeSent ? (
                <form className="space-y-5" onSubmit={handleSendCode}>
                  <div>
                    <label htmlFor="phone-number" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Phone Number
                    </label>
                    <div className="flex">
                      <CountryCodeSelector 
                        selectedCode={countryCode} 
                        onSelect={setCountryCode} 
                      />
                      <div className="relative flex-1">
                        <input
                          id="phone-number"
                          name="phone"
                          type="tel"
                          required
                          className="block w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-r-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 hover:border-teal-400 sm:text-sm transition-all duration-200"
                          placeholder="Enter your phone number"
                          value={phoneNumber}
                          onChange={(e) => setPhoneNumber(e.target.value)}
                        />
                      </div>
                    </div>
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                      We'll send a verification code to this number
                    </p>
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className="relative w-full flex items-center justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-teal-600 hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 group"
                  >
                    <span className="flex items-center">
                      {loading ? 'Sending Code...' : 'Send Verification Code'}
                      <FiArrowRight className="ml-2 group-hover:translate-x-1 transition-transform duration-200" size={16} />
                    </span>
                  </button>
                </form>
              ) : (
                <form className="space-y-5" onSubmit={handleVerifyCode}>
                  <div className="text-center">
                    <label htmlFor="verification-code" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
                      Verification Code
                    </label>
                    
                    <OtpInput 
                      length={6} 
                      value={verificationCode} 
                      onChange={setVerificationCode} 
                    />
                    
                    <p className="mt-4 text-xs text-gray-500 dark:text-gray-400">
                      Didn't receive the code? <button type="button" className="text-teal-600 hover:text-teal-500 dark:text-teal-400 dark:hover:text-teal-300 transition-colors duration-200 font-medium">Resend</button>
                    </p>
                  </div>

                  <button
                    type="submit"
                    disabled={loading || verificationCode.length < 6}
                    className="relative w-full flex items-center justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-teal-600 hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 group"
                  >
                    <span className="flex items-center">
                      {loading ? 'Verifying...' : 'Verify & Continue'}
                      <FiArrowRight className="ml-2 group-hover:translate-x-1 transition-transform duration-200" size={16} />
                    </span>
                  </button>
                  
                  <div className="text-center">
                    <button
                      type="button"
                      onClick={() => setCodeSent(false)}
                      className="text-sm font-medium text-teal-600 hover:text-teal-500 dark:text-teal-400 dark:hover:text-teal-300 transition-colors duration-200"
                    >
                      Use a different phone number
                    </button>
                  </div>
                </form>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default LoginModal; 