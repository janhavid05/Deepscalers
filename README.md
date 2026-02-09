# AI Student Assistant Platform

A responsive front-end for an AI-powered student assistant platform built with React and TailwindCSS.

## Features

- 📝 Chat Interface: Ask questions and receive AI-powered responses
- 👨‍🏫 Faculty Verified Answers: Access information validated by faculty members
- 📅 Deadline Management: Keep track of assignments and exams
- 📱 Responsive Design: Works on desktop, tablet, and mobile devices
- 🌙 Dark Mode: Toggle between light and dark themes

## Pages

1. **Home**: Introduction to the platform with feature cards
2. **Chat**: Main interface for asking questions and receiving AI responses
3. **My Questions**: Dashboard showing all previously asked questions
4. **Faculty Answers**: Collection of faculty-verified responses
5. **Deadlines**: Manage upcoming assignments and exams
6. **Faculty Dashboard**: For faculty to review and verify AI responses
7. **Settings**: Profile and notification preferences

## Tech Stack

- React (with TypeScript)
- TailwindCSS for styling
- React Router for navigation
- React Icons for UI elements

## Getting Started

### Prerequisites

- Node.js and npm installed

### Installation

1. Clone the repository:
```
git clone <repository-url>
```

2. Install dependencies:
```
cd student-assistant
npm install
```

3. Start the development server:
```
npm run dev
```

4. Open your browser and navigate to `http://localhost:5173`

## Deployment

### Vercel Deployment

1. Update the backend API URL in `src/config.ts`:

```typescript
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 
    (window.location.hostname === 'localhost' 
      ? 'http://localhost:8000/api' 
      : 'https://your-backend-api-url.com/api'), // Update this with your actual deployed backend URL
  
  // Rest of the config...
};
```

2. Update the environment variable in `vercel.json`:

```json
{
  "env": {
    "VITE_API_URL": "https://your-backend-api-url.com/api"  // Your actual backend URL
  }
}
```

3. Deploy to Vercel:

```
vercel
```

### Troubleshooting Mobile Issues

If you encounter "Failed to fetch" errors on mobile devices:

1. Ensure your backend is properly deployed and accessible publicly
2. Verify that CORS is properly configured on your backend to allow requests from your frontend domain
3. Check if your backend URL is correctly set in both `vercel.json` and `src/config.ts`
4. Verify that your mobile device has internet connectivity and can reach your backend

## Project Structure

- `/src`
  - `/components`: Reusable UI components
  - `/pages`: Main page components
  - `/context`: React context providers (if applicable)
  - `App.tsx`: Main application component
  - `main.tsx`: Entry point

## Notes

This is a front-end only implementation using dummy data. In a production environment, you would integrate with a backend API for:

- User authentication
- Persistent data storage
- AI model integration for generating responses
- Faculty user management

## Future Enhancements

- Integration with actual AI models (e.g., OpenAI, Anthropic)
- Real-time notifications
- File management system for assignments
- More interactive components like whiteboard and code editor 