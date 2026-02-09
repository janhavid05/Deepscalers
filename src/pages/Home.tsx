// Home page component - Updated for Vercel deployment test
import React, { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { FiSend, FiSearch, FiBookOpen, FiClock, FiUsers, FiArrowRight, FiCommand, FiZap, FiLogIn, FiBriefcase, FiWifi, FiWifiOff } from 'react-icons/fi'
import { useAuth } from '../context/AuthContext'
import { testBackendConnection, API_CONFIG } from '../config'
import { navigateToChat } from '../lib/navigation'

// Example queries for suggestions
const exampleQueries = [
  "Explain quantum computing in simple terms",
  "How do I solve quadratic equations?",
  "What's the difference between mitosis and meiosis?",
  "Explain Newton's laws of motion",
  "How does photosynthesis work?",
  "What is the theory of relativity?"
];

// Example capabilities to showcase
const capabilities = [
  {
    icon: FiSearch,
    title: "Research Assistant",
    description: "Find answers to complex academic questions instantly"
  },
  {
    icon: FiBookOpen,
    title: "Study Guide",
    description: "Get explanations on academic topics in simple language"
  },
  {
    icon: FiClock,
    title: "Deadline Management",
    description: "Track your assignments and exam schedules"
  },
  {
    icon: FiUsers,
    title: "Faculty Interaction",
    description: "Access verified answers from professors"
  }
];

interface HomeProps {
  onLoginClick?: () => void;
}

const Home: React.FC<HomeProps> = ({ onLoginClick }) => {
  const [query, setQuery] = useState('');
  const [isInputFocused, setIsInputFocused] = useState(false);
  const [activeSuggestion, setActiveSuggestion] = useState(-1);
  const [typing, setTyping] = useState(false);
  const [animateIn, setAnimateIn] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [testResult, setTestResult] = useState<any>(null);
  const [isTestingBackend, setIsTestingBackend] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { isAuthenticated, user } = useAuth();

  // Auto-run the animation after component mounts
  useEffect(() => {
    setAnimateIn(true);
  }, []);

  // Handle query submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      // Process the message directly
      const message = query.trim();
      setQuery('');
      
      // Navigate to chat with the message
      navigate('/chat', {
        state: {
          initialQuery: message,
          autoSend: true
        }
      });
    }
  };

  // Handle example query click
  const handleExampleClick = (example: string) => {
    setQuery('');
    setTyping(true);
    
    // Type the example letter by letter
    let i = 0;
    const interval = setInterval(() => {
      if (i <= example.length) {
        setQuery(example.slice(0, i));
        i++;
      } else {
        clearInterval(interval);
        setTyping(false);
      }
    }, 25); // typing speed
    
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };
  
  // Handle keyboard navigation for suggestions
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveSuggestion(prev => Math.min(prev + 1, exampleQueries.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveSuggestion(prev => Math.max(prev - 1, -1));
    } else if (e.key === 'Enter' && activeSuggestion !== -1) {
      e.preventDefault();
      handleExampleClick(exampleQueries[activeSuggestion]);
      setActiveSuggestion(-1);
    }
  };

  // Add backend connection test function
  const handleTestBackend = async () => {
    setIsTestingBackend(true);
    setTestResult(null);
    try {
      const result = await testBackendConnection();
      setTestResult(result);
    } catch (error) {
      setTestResult({ 
        success: false, 
        error: error instanceof Error ? error.message : String(error) 
      });
    } finally {
      setIsTestingBackend(false);
    }
  };

  // Add secret code to show debug panel (5 rapid clicks on the logo)
  const [clickCount, setClickCount] = useState(0);
  useEffect(() => {
    if (clickCount >= 5) {
      setShowDebug(true);
      setClickCount(0);
    }
    
    const timer = setTimeout(() => {
      if (clickCount > 0) {
        setClickCount(0);
      }
    }, 2000);
    
    return () => clearTimeout(timer);
  }, [clickCount]);

  return (
    <div className="flex flex-col min-h-screen bg-white dark:bg-black animate-fadeIn">
      <div 
        className={`flex-1 flex flex-col items-center justify-center px-4 py-12 transform transition-all duration-700 ${
          animateIn ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
        }`}
      >
        {/* Logo and Title */}
        <div className="mb-12 text-center transform transition-all duration-500" style={{ transitionDelay: '200ms' }}>
          <div 
            className="inline-block w-20 h-20 rounded-2xl bg-gradient-to-br from-teal-400 to-teal-600 flex items-center justify-center text-white text-3xl font-bold mb-6 shadow-lg transform hover:rotate-3 hover:scale-110 transition-all duration-300"
            onClick={() => setClickCount(count => count + 1)}
          >
            AI
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
            Student Assistant
          </h1>
          
          {isAuthenticated && user ? (
            <div className="mt-3 text-gray-700 dark:text-gray-200 text-lg">
              <p>Welcome back, <span className="font-medium">{user.first_name || 'Student'}</span>!</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                What would you like to learn today?
              </p>
            </div>
          ) : (
            <div className="mt-3">
              <p className="text-gray-600 dark:text-gray-300 text-lg">
                Your personal AI academic companion
              </p>
              <button
                onClick={onLoginClick}
                className="mt-4 flex items-center justify-center mx-auto px-4 py-2 text-sm text-white bg-teal-600 hover:bg-teal-700 rounded-md transition-colors duration-200"
              >
                <FiLogIn className="mr-2" size={16} />
                <span>Login to get started</span>
              </button>
            </div>
          )}
        </div>
        
        {/* Query Input */}
        <div 
          className={`w-full max-w-2xl transition-all duration-500 transform ${
            isInputFocused ? 'scale-105' : 'scale-100'
          }`}
          style={{ transitionDelay: '400ms' }}
        >
          <form onSubmit={handleSubmit} className="relative" onKeyDown={handleKeyDown}>
            <div className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400">
              <FiSearch size={20} className={isInputFocused ? 'text-teal-500' : ''} />
            </div>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => {
                setIsInputFocused(true);
                setActiveSuggestion(-1);
              }}
              onBlur={() => setIsInputFocused(false)}
              disabled={typing}
              placeholder="Ask me anything about your studies..."
              className="w-full p-5 pl-12 pr-14 border-2 border-gray-200 dark:border-gray-800 rounded-xl shadow-sm bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-4 focus:ring-teal-500/20 focus:border-teal-500 dark:focus:border-teal-500 transition-all duration-200"
            />
            <button
              type="submit"
              disabled={!query.trim() || typing}
              className={`absolute right-3 top-1/2 transform -translate-y-1/2 p-2 rounded-lg transition-all duration-200 ${
                query.trim() && !typing
                  ? 'bg-teal-500 text-white shadow-md hover:bg-teal-600 hover:shadow-lg active:scale-95'
                  : 'bg-gray-200 dark:bg-gray-800 text-gray-400 cursor-not-allowed'
              }`}
            >
              <FiArrowRight size={20} />
            </button>
          </form>
          
          <div className="mt-2 flex items-center justify-between px-1">
            <div className="flex items-center text-xs text-gray-500 dark:text-gray-400">
              <FiCommand className="mr-1" size={14} />
              <span>+ K to focus</span>
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              <FiZap className="inline-block mr-1" size={14} />
              <span>Powered by AI</span>
            </div>
          </div>
        </div>
        
        {/* Example Queries */}
        <div 
          className="mt-12 w-full max-w-3xl transform transition-all duration-500 delay-500"
          style={{ transitionDelay: '600ms' }}
        >
          <h2 className="text-center text-lg font-medium text-gray-700 dark:text-gray-300 mb-5">
            Examples you can ask
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {exampleQueries.map((example, index) => (
              <button 
                key={index}
                onClick={() => handleExampleClick(example)}
                onMouseEnter={() => setActiveSuggestion(index)}
                onMouseLeave={() => setActiveSuggestion(-1)}
                className={`p-4 text-left border border-gray-200 dark:border-gray-800 rounded-xl group hover:border-teal-300 dark:hover:border-teal-700 transition-all duration-200 ${
                  activeSuggestion === index 
                    ? 'bg-teal-50 dark:bg-gray-800 border-teal-300 dark:border-teal-700 shadow-sm' 
                    : 'bg-white dark:bg-gray-900 hover:bg-gray-50 dark:hover:bg-gray-800'
                }`}
              >
                <span className="text-gray-700 dark:text-gray-300 group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors duration-200">
                  "{example}"
                </span>
              </button>
            ))}
          </div>
        </div>
        
        {/* Capabilities */}
        <div 
          className="mt-16 w-full max-w-4xl transform transition-all duration-500"
          style={{ transitionDelay: '800ms' }}
        >
          <h2 className="text-center text-lg font-medium text-gray-700 dark:text-gray-300 mb-6">Capabilities</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {capabilities.map((capability, index) => (
              <div 
                key={index} 
                className="p-6 border border-gray-200 dark:border-gray-800 rounded-xl bg-white dark:bg-gray-900 text-center hover:shadow-md hover:border-teal-300 dark:hover:border-teal-700 transition-all duration-300 group"
                style={{ transitionDelay: `${900 + index * 100}ms` }}
              >
                <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-teal-100 dark:bg-gray-800 text-teal-600 dark:text-teal-400 mb-4 group-hover:scale-110 group-hover:bg-teal-200 dark:group-hover:bg-gray-700 transition-all duration-300">
                  <capability.icon size={24} />
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors duration-200">{capability.title}</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">{capability.description}</p>
              </div>
            ))}
          </div>
        </div>
        
        {/* Debug Panel */}
        {showDebug && (
          <div className="mt-8 w-full max-w-3xl border-2 border-orange-300 dark:border-orange-700 rounded-lg p-4 bg-orange-50 dark:bg-gray-900">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-medium text-orange-800 dark:text-orange-400 flex items-center">
                <FiBriefcase className="mr-2" />
                Debug Tools
              </h3>
              <button 
                onClick={() => setShowDebug(false)}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                Close
              </button>
            </div>
            
            <div className="mb-4">
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                <strong>API URL:</strong> {API_CONFIG.BASE_URL}
              </p>
              
              <button
                onClick={handleTestBackend}
                disabled={isTestingBackend}
                className={`flex items-center px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                  isTestingBackend 
                    ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-wait' 
                    : 'bg-teal-600 hover:bg-teal-700 text-white'
                }`}
              >
                {isTestingBackend ? (
                  <>Testing Connection...</>
                ) : (
                  <>
                    {testResult?.success ? <FiWifi className="mr-2" /> : <FiWifiOff className="mr-2" />}
                    Test Backend Connection
                  </>
                )}
              </button>
              
              {testResult && (
                <div className={`mt-3 p-3 rounded text-sm ${
                  testResult.success 
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400'
                    : 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-400'
                }`}>
                  <p><strong>Status:</strong> {testResult.success ? 'Connected' : 'Failed'}</p>
                  {testResult.status && <p><strong>HTTP Status:</strong> {testResult.status}</p>}
                  {testResult.time && <p><strong>Response Time:</strong> {testResult.time}ms</p>}
                  {testResult.error && <p><strong>Error:</strong> {testResult.error}</p>}
                </div>
              )}
            </div>
            
            <div className="text-xs text-gray-500 dark:text-gray-400 italic">
              This panel is for debugging purposes only.
            </div>
          </div>
        )}
      </div>
      
      {/* Keyboard Shortcut Handler */}
      <KeyboardShortcut 
        keys={['Meta', 'k']} 
        callback={() => inputRef.current?.focus()} 
      />
      
      {/* Footer */}
      <footer className="w-full py-5 text-center text-sm text-gray-500 dark:text-gray-400 border-t border-gray-200 dark:border-gray-800">
        <p className="max-w-3xl mx-auto px-4">
          Student AI Assistant © {new Date().getFullYear()} • 
          <span className="mx-2">·</span>
          Built for academic excellence
          <span className="mx-2">·</span>
          <span className="text-teal-500 dark:text-teal-400">Privacy-focused</span>
        </p>
      </footer>
    </div>
  )
}

// Keyboard Shortcut Handler Component
const KeyboardShortcut = ({ 
  keys, 
  callback 
}: { 
  keys: string[], 
  callback: () => void 
}) => {
  useEffect(() => {
    const keysPressed: Record<string, boolean> = {};
    
    const handleKeyDown = (e: KeyboardEvent) => {
      keysPressed[e.key] = true;
      
      // Convert Command key (Mac) to Meta
      if (e.key === 'Meta') {
        keysPressed['Meta'] = true;
      }
      
      // Check if all required keys are pressed
      const allKeysPressed = keys.every(key => {
        if (key === 'Meta' && (keysPressed['Meta'] || keysPressed['Control'])) {
          return true;
        }
        return keysPressed[key];
      });
      
      if (allKeysPressed) {
        e.preventDefault();
        callback();
      }
    };
    
    const handleKeyUp = (e: KeyboardEvent) => {
      delete keysPressed[e.key];
      
      // Convert Command key (Mac) to Meta
      if (e.key === 'Meta') {
        delete keysPressed['Meta'];
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [keys, callback]);
  
  return null;
};

export default Home; 
