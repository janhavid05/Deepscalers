// Debug page for testing backend connection - Updated for Vercel deployment
import React, { useState, useEffect } from 'react';
import { FiArrowLeft, FiWifi, FiWifiOff, FiRefreshCw, FiCheck, FiX, FiAlertTriangle } from 'react-icons/fi';
import { Link, useNavigate } from 'react-router-dom';
import { API_CONFIG, testBackendConnection } from '../config';
import { useAuth } from '../context/AuthContext';

const DebugPage: React.FC = () => {
  const [testResult, setTestResult] = useState<any>(null);
  const [isTestingBackend, setIsTestingBackend] = useState(false);
  const [phoneForTest, setPhoneForTest] = useState('+911234567890');
  const [testEndpoint, setTestEndpoint] = useState('/auth/send-verification/');
  const [customTestUrl, setCustomTestUrl] = useState('');
  const [responseData, setResponseData] = useState<any>(null);
  const [isCustomTesting, setIsCustomTesting] = useState(false);
  const navigate = useNavigate();
  const { loading } = useAuth();

  // Test basic backend connection
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

  // Test sending verification code
  const handleTestSendVerification = async () => {
    setIsCustomTesting(true);
    setResponseData(null);
    
    const startTime = Date.now();
    try {
      console.log(`Testing endpoint: ${API_CONFIG.BASE_URL}${testEndpoint}`, { phone_number: phoneForTest });
      
      const response = await fetch(`${API_CONFIG.BASE_URL}${testEndpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ phone_number: phoneForTest }),
      });
      
      const elapsed = Date.now() - startTime;
      
      let data = null;
      let text = '';
      try {
        text = await response.text();
        data = JSON.parse(text);
      } catch (e) {
        console.error('Error parsing response:', e);
      }
      
      setResponseData({
        success: response.ok,
        status: response.status,
        statusText: response.statusText,
        time: elapsed,
        data: data,
        text: !data ? text : undefined,
        headers: Object.fromEntries([...response.headers.entries()]),
      });
      
    } catch (error) {
      const elapsed = Date.now() - startTime;
      setResponseData({
        success: false,
        time: elapsed,
        error: error instanceof Error ? error.message : String(error)
      });
    } finally {
      setIsCustomTesting(false);
    }
  };

  // Test custom endpoint
  const handleTestCustomEndpoint = async () => {
    setIsCustomTesting(true);
    setResponseData(null);
    
    const startTime = Date.now();
    const url = customTestUrl.startsWith('http') 
      ? customTestUrl 
      : `${API_CONFIG.BASE_URL}${customTestUrl}`;
    
    try {
      console.log(`Testing custom endpoint: ${url}`);
      
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      });
      
      const elapsed = Date.now() - startTime;
      
      let data = null;
      let text = '';
      try {
        text = await response.text();
        data = JSON.parse(text);
      } catch (e) {
        console.error('Error parsing response:', e);
      }
      
      setResponseData({
        success: response.ok,
        status: response.status,
        statusText: response.statusText,
        time: elapsed,
        data: data,
        text: !data ? text : undefined,
        headers: Object.fromEntries([...response.headers.entries()]),
      });
      
    } catch (error) {
      const elapsed = Date.now() - startTime;
      setResponseData({
        success: false,
        time: elapsed,
        error: error instanceof Error ? error.message : String(error)
      });
    } finally {
      setIsCustomTesting(false);
    }
  };

  // Run initial connection test on component mount
  useEffect(() => {
    handleTestBackend();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center">
            <button
              onClick={() => navigate(-1)}
              className="mr-4 p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <FiArrowLeft size={20} className="text-gray-600 dark:text-gray-300" />
            </button>
            <h1 className="text-xl font-bold text-gray-800 dark:text-white flex items-center">
              <span className="bg-yellow-500 text-white p-1 rounded mr-2 text-xs font-medium">DEBUG</span>
              API Connectivity Tester
            </h1>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-6">
        {/* Connection Status */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-medium text-gray-800 dark:text-white">Backend Connection Status</h2>
            <button
              onClick={handleTestBackend}
              disabled={isTestingBackend}
              className={`p-2 rounded-md transition-colors ${
                isTestingBackend
                  ? 'bg-gray-100 dark:bg-gray-700 text-gray-400'
                  : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300'
              }`}
            >
              <FiRefreshCw
                size={18}
                className={isTestingBackend ? 'animate-spin' : ''}
              />
            </button>
          </div>

          <div className="mb-4">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              <strong>API Base URL:</strong> {API_CONFIG.BASE_URL}
            </p>
          </div>

          {isTestingBackend ? (
            <div className="flex items-center justify-center py-6">
              <FiRefreshCw className="animate-spin text-teal-500 mr-2" size={20} />
              <span className="text-gray-600 dark:text-gray-400">Testing connection...</span>
            </div>
          ) : testResult ? (
            <div className={`flex items-start p-4 rounded-lg ${
              testResult.success
                ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400'
                : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400'
            }`}>
              <div className="flex-shrink-0 mr-3">
                {testResult.success ? (
                  <div className="bg-green-100 dark:bg-green-800/30 p-2 rounded-full">
                    <FiCheck size={18} />
                  </div>
                ) : (
                  <div className="bg-red-100 dark:bg-red-800/30 p-2 rounded-full">
                    <FiX size={18} />
                  </div>
                )}
              </div>
              <div>
                <h3 className="font-medium">
                  {testResult.success ? 'Connected Successfully' : 'Connection Failed'}
                </h3>
                <div className="mt-1 text-sm">
                  {testResult.status && <p><strong>HTTP Status:</strong> {testResult.status}</p>}
                  {testResult.time && <p><strong>Response Time:</strong> {testResult.time}ms</p>}
                  {testResult.error && (
                    <p className="mt-1">
                      <strong>Error:</strong> {testResult.error}
                    </p>
                  )}
                </div>
              </div>
            </div>
          ) : null}
        </div>

        {/* Test verification endpoint */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-medium text-gray-800 dark:text-white mb-4">
            Test Phone Verification
          </h2>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Phone Number
            </label>
            <input
              type="text"
              value={phoneForTest}
              onChange={(e) => setPhoneForTest(e.target.value)}
              className="w-full p-2 border border-gray-300 dark:border-gray-700 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
              placeholder="+910000000000"
            />
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Endpoint
            </label>
            <select
              value={testEndpoint}
              onChange={(e) => setTestEndpoint(e.target.value)}
              className="w-full p-2 border border-gray-300 dark:border-gray-700 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
            >
              <option value="/auth/send-verification/">Send Verification Code</option>
              <option value="/auth/verify-code/">Verify Code (requires code parameter)</option>
              <option value="/auth/profile/">User Profile (requires auth)</option>
            </select>
          </div>

          <button
            onClick={handleTestSendVerification}
            disabled={isCustomTesting}
            className={`w-full flex items-center justify-center py-3 px-4 rounded-md transition-colors ${
              isCustomTesting
                ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-teal-600 hover:bg-teal-700 text-white'
            }`}
          >
            {isCustomTesting ? (
              <>
                <FiRefreshCw className="animate-spin mr-2" size={16} />
                Testing...
              </>
            ) : (
              <>Test Endpoint</>
            )}
          </button>
        </div>

        {/* Test custom endpoint */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-medium text-gray-800 dark:text-white mb-4">
            Test Custom Request
          </h2>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Custom Endpoint
            </label>
            <input
              type="text"
              value={customTestUrl}
              onChange={(e) => setCustomTestUrl(e.target.value)}
              className="w-full p-2 border border-gray-300 dark:border-gray-700 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
              placeholder="/auth/profile/ or https://example.com/api"
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Enter a path (e.g., "/auth/profile/") or full URL
            </p>
          </div>

          <button
            onClick={handleTestCustomEndpoint}
            disabled={isCustomTesting || !customTestUrl}
            className={`w-full flex items-center justify-center py-3 px-4 rounded-md transition-colors ${
              isCustomTesting || !customTestUrl
                ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {isCustomTesting ? (
              <>
                <FiRefreshCw className="animate-spin mr-2" size={16} />
                Testing...
              </>
            ) : (
              <>Test Custom Endpoint</>
            )}
          </button>
        </div>

        {/* Response Data */}
        {responseData && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6 mb-6">
            <h2 className="text-lg font-medium text-gray-800 dark:text-white mb-4 flex items-center">
              Response Data
              <span className={`ml-2 text-xs font-medium px-2 py-1 rounded ${
                responseData.success 
                  ? 'bg-green-100 dark:bg-green-900/40 text-green-800 dark:text-green-400' 
                  : 'bg-red-100 dark:bg-red-900/40 text-red-800 dark:text-red-400'
              }`}>
                {responseData.status || (responseData.success ? 'Success' : 'Error')}
              </span>
            </h2>

            <div className="space-y-4">
              {responseData.time && (
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Response Time</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{responseData.time}ms</p>
                </div>
              )}

              {responseData.status && (
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300">HTTP Status</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {responseData.status} {responseData.statusText}
                  </p>
                </div>
              )}

              {responseData.headers && (
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Headers</p>
                  <div className="mt-1 bg-gray-50 dark:bg-gray-900 p-3 rounded-md overflow-x-auto">
                    <pre className="text-xs text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
                      {JSON.stringify(responseData.headers, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {responseData.data && (
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Response Data</p>
                  <div className="mt-1 bg-gray-50 dark:bg-gray-900 p-3 rounded-md overflow-x-auto">
                    <pre className="text-xs text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
                      {JSON.stringify(responseData.data, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {responseData.text && !responseData.data && (
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Response Text</p>
                  <div className="mt-1 bg-gray-50 dark:bg-gray-900 p-3 rounded-md overflow-x-auto">
                    <pre className="text-xs text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
                      {responseData.text}
                    </pre>
                  </div>
                </div>
              )}

              {responseData.error && (
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Error</p>
                  <div className="mt-1 bg-red-50 dark:bg-red-900/20 p-3 rounded-md overflow-x-auto">
                    <pre className="text-xs text-red-800 dark:text-red-400 whitespace-pre-wrap">
                      {responseData.error}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Network Info */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-medium text-gray-800 dark:text-white mb-4">
            Browser Network Info
          </h2>

          <div className="space-y-3">
            <div>
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Online Status</p>
              <p className="flex items-center text-sm">
                {navigator.onLine ? (
                  <>
                    <span className="inline-block w-2 h-2 rounded-full bg-green-500 mr-2"></span>
                    <span className="text-green-600 dark:text-green-400">Online</span>
                  </>
                ) : (
                  <>
                    <span className="inline-block w-2 h-2 rounded-full bg-red-500 mr-2"></span>
                    <span className="text-red-600 dark:text-red-400">Offline</span>
                  </>
                )}
              </p>
            </div>

            <div>
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300">User Agent</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 break-words">
                {navigator.userAgent}
              </p>
            </div>

            <div>
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Current URL</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 break-words">
                {window.location.href}
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default DebugPage; 