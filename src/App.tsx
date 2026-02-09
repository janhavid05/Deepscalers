import { useEffect, useState } from 'react'
import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import LoginModal from './components/LoginModal'
import { AuthProvider, useAuth } from './context/AuthContext'
import BranchSelection from './pages/BranchSelection'
import ChatPage from './pages/ChatPage'
import Deadlines from './pages/Deadlines'
import DebugPage from './pages/Debug'
import FacultyAnswers from './pages/FacultyAnswers'
import FacultyDashboard from './pages/FacultyDashboard'
import Home from './pages/Home'
import MyQuestions from './pages/MyQuestions'
import Settings from './pages/Settings'
import FAQ from './components/FAQ'

function AppContent() {
  const [darkMode, setDarkMode] = useState(false)
  const { isAuthenticated, loading } = useAuth();
  const [showLoginModal, setShowLoginModal] = useState(false);

  // Load dark mode preference from localStorage on mount
  useEffect(() => {
    const savedMode = localStorage.getItem('darkMode')
    if (savedMode) {
      setDarkMode(savedMode === 'true')
    } else {
      // Check for system preference
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      setDarkMode(prefersDark)
    }
  }, [])

  // Update localStorage when darkMode changes
  useEffect(() => {
    localStorage.setItem('darkMode', darkMode.toString())
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  // Handle login button click
  const handleLoginClick = () => {
    setShowLoginModal(true);
  };

  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'dark bg-black text-white' : 'bg-white text-gray-900'}`}>
      {/* Main application layout */}
      <Routes>
        <Route path="/" element={
          <Layout 
            darkMode={darkMode} 
            setDarkMode={setDarkMode} 
            onLoginClick={handleLoginClick} 
          />
        }>
          <Route index element={<Home onLoginClick={handleLoginClick} />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="my-questions" element={<MyQuestions />} />
          <Route path="faculty-answers" element={<FacultyAnswers />} />
          <Route path="deadlines" element={<Deadlines />} />
          <Route path="settings" element={<Settings />} />
          <Route path="faculty-dashboard" element={<BranchSelection />} />
          <Route path="faculty-dashboard/dashboard" element={<FacultyDashboard />} />
          <Route path="debug" element={<DebugPage />} />
          <Route path="faq" element={<FAQ />} />
        </Route>
      </Routes>

      {/* Login Modal (shown when login button is clicked) */}
      <LoginModal 
        isOpen={showLoginModal} 
        onClose={() => setShowLoginModal(false)} 
      />

      {/* Full-screen loading indicator */}
      {loading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-white dark:bg-black bg-opacity-75 dark:bg-opacity-75">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-t-indigo-500 border-gray-200 rounded-full animate-spin mx-auto"></div>
            <p className="mt-3 text-gray-700 dark:text-gray-300">Loading...</p>
          </div>
        </div>
      )}
    </div>
  )
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App 