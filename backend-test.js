// Backend API testing script
// This checks if the backend API is working correctly

// Configuration 
const API_BASE_URL = 'http://localhost:8000/api';
const PHONE_NUMBER = '1234567890'; // Use a test phone number

// Test API functions
async function testSendVerification() {
  console.log('Testing send verification code endpoint...');
  try {
    const response = await fetch(`${API_BASE_URL}/auth/send-verification/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ phone_number: PHONE_NUMBER }),
    });
    
    console.log('Status:', response.status);
    if (response.ok) {
      console.log('✅ Send verification success!');
      const data = await response.json();
      console.log('Response:', data);
      return true;
    } else {
      console.error('❌ Send verification failed!');
      const errorText = await response.text();
      console.error('Error:', errorText);
      return false;
    }
  } catch (error) {
    console.error('❌ Network error:', error);
    return false;
  }
}

async function testVerifyCode(code = '123456') {
  console.log('Testing verify code endpoint...');
  try {
    const response = await fetch(`${API_BASE_URL}/auth/verify-code/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        phone_number: PHONE_NUMBER,
        code: code 
      }),
    });
    
    console.log('Status:', response.status);
    const responseText = await response.text();
    
    try {
      // Try to parse as JSON
      const data = JSON.parse(responseText);
      console.log('Response data:', data);
      
      if (response.ok) {
        console.log('✅ Verify code success!');
        
        // Check if important fields are present
        if (data.access && data.refresh && data.user_exists !== undefined) {
          console.log('✅ Response has required fields (access, refresh, user_exists)');
          
          if (data.user_exists) {
            console.log('✅ User exists, has user data:', !!data.user);
          } else {
            console.log('ℹ️ User does not exist, registration required');
          }
        } else {
          console.error('❌ Response missing required fields!');
          console.log('Has access token:', !!data.access);
          console.log('Has refresh token:', !!data.refresh);
          console.log('Has user_exists flag:', data.user_exists !== undefined);
        }
      } else {
        console.error('❌ Verify code failed!');
      }
    } catch (e) {
      console.error('❌ Response is not valid JSON:', responseText);
    }
  } catch (error) {
    console.error('❌ Network error:', error);
  }
}

// Function to test the profile endpoint
async function testProfileEndpoint(token) {
  if (!token) {
    console.log('No token provided, skipping profile test');
    return;
  }
  
  console.log('Testing profile endpoint...');
  try {
    const response = await fetch(`${API_BASE_URL}/auth/profile/`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    console.log('Status:', response.status);
    if (response.ok) {
      const userData = await response.json();
      console.log('✅ Profile endpoint success!');
      console.log('User data:', userData);
    } else {
      console.error('❌ Profile endpoint failed!');
      const errorText = await response.text();
      console.error('Error:', errorText);
    }
  } catch (error) {
    console.error('❌ Network error:', error);
  }
}

// Run the tests
console.log('==============================');
console.log('BACKEND API TESTS');
console.log('==============================');
console.log('API Base URL:', API_BASE_URL);
console.log('Test Phone Number:', PHONE_NUMBER);
console.log('==============================');

// Instructions for use:
// 1. Make sure the backend is running
// 2. Open the browser console on the login page
// 3. Paste this script to run the tests
// 4. Check the console for test results
// 5. For full testing, update the verification code when prompted 