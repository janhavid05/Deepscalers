// Configuration constants for the application

// API URL configuration with environment fallbacks
export const API_CONFIG = {
  // Use environment variable if available, otherwise determine based on hostname
  BASE_URL: import.meta.env.VITE_API_URL || 
    (window.location.hostname === 'localhost' 
      ? 'http://localhost:8000/api' 
      : 'https://deepscalers-backend-production.up.railway.app/api'),
  
  // Path constants
  AUTH: {
    SEND_VERIFICATION: '/student/send-verification/',
    VERIFY_CODE: '/student/verify-code/',
    PROFILE: '/student/profile/',
    REGISTER: '/student/register/',
    TOKEN: '/student/token/',
    TOKEN_REFRESH: '/student/token/refresh/',
  },
};

// Test connectivity to backend
export async function testBackendConnection() {
  const startTime = Date.now();
  try {
    console.log(`Testing backend connection to: ${API_CONFIG.BASE_URL}`);
    console.log('Full URL:', `${API_CONFIG.BASE_URL}${API_CONFIG.AUTH.SEND_VERIFICATION}`);
    
    // Try to connect to a specific endpoint
    const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.AUTH.SEND_VERIFICATION}`, {
      method: 'OPTIONS',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Origin': window.location.origin,
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
      credentials: 'include',
      mode: 'cors',
    });
    
    const elapsed = Date.now() - startTime;
    console.log(`Backend connection test completed in ${elapsed}ms`);
    console.log(`Status: ${response.status}`);
    console.log('Response headers:', Object.fromEntries(response.headers.entries()));
    
    if (response.ok) {
      console.log('Backend connection successful!');
      return { success: true, status: response.status, time: elapsed };
    } else {
      console.error('Backend connection failed with status:', response.status);
      const errorText = await response.text();
      console.error('Error response:', errorText);
      return { 
        success: false, 
        status: response.status, 
        time: elapsed,
        error: errorText 
      };
    }
  } catch (error) {
    const elapsed = Date.now() - startTime;
    console.error('Backend connection error:', error);
    console.error('Error details:', {
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined
    });
    return { 
      success: false, 
      error: error instanceof Error ? error.message : String(error),
      time: elapsed 
    };
  }
}

// Other application configuration
export const APP_CONFIG = {
  // Default timeout for API requests (in milliseconds)
  DEFAULT_TIMEOUT: 15000,
  
  // Frontend routes
  ROUTES: {
    HOME: '/',
    LOGIN: '/login',
    REGISTER: '/register',
    DASHBOARD: '/dashboard',
    PROFILE: '/profile',
  },
}; 