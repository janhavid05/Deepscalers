// Script to debug authentication issues
// Open browser devtools on the login page and run this in the console

// Function to check localStorage tokens
function checkTokens() {
  const accessToken = localStorage.getItem('access_token');
  const refreshToken = localStorage.getItem('refresh_token');
  
  console.log('Access token exists:', !!accessToken);
  console.log('Refresh token exists:', !!refreshToken);
  
  if (accessToken) {
    // Check token format (should be a JWT)
    const tokenParts = accessToken.split('.');
    if (tokenParts.length !== 3) {
      console.error('Access token is not in valid JWT format');
    } else {
      console.log('Access token appears to be in valid JWT format');
      try {
        // Decode JWT payload (middle part)
        const payload = JSON.parse(atob(tokenParts[1]));
        console.log('Token payload:', payload);
        
        // Check expiration
        if (payload.exp) {
          const expDate = new Date(payload.exp * 1000);
          const now = new Date();
          console.log('Token expires:', expDate);
          console.log('Token expired:', expDate < now);
        }
      } catch (e) {
        console.error('Failed to decode token:', e);
      }
    }
  }
}

// Function to test API endpoints
async function testAuthEndpoints() {
  try {
    // Test profile endpoint with the token
    const token = localStorage.getItem('access_token');
    if (token) {
      console.log('Testing profile endpoint...');
      const profileResponse = await fetch('http://localhost:8000/api/auth/profile/', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      console.log('Profile API status:', profileResponse.status);
      if (profileResponse.ok) {
        const userData = await profileResponse.json();
        console.log('User data:', userData);
      } else {
        console.error('Profile API failed. Response:', await profileResponse.text());
      }
    } else {
      console.log('No token available to test profile endpoint');
    }
  } catch (error) {
    console.error('API test failed:', error);
  }
}

// Run the diagnostic functions
console.log('=== AUTH DEBUGGING ===');
checkTokens();
testAuthEndpoints();
console.log('=== END DEBUGGING ===');

// Instructions:
// 1. After login attempt, paste this script in browser console
// 2. Check console output for authentication issues
// 3. Look for any API errors or token problems 