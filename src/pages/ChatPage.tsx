import { useState, useRef, useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import { FiSend, FiPaperclip, FiMic, FiChevronDown, FiCommand, FiZap, FiRefreshCw, FiThumbsUp, FiThumbsDown } from 'react-icons/fi'
import ChatBubble, { MessageType } from '../components/ChatBubble'
import { processMessage, createUserMessage, createAIMessage } from '../lib/chatOperations'
import axios from 'axios'
import { API_CONFIG } from '../config'

// Add these animation styles to your tailwind.css or append to a style tag
const VoiceAnimations = () => (
  <style>
    {`
     @keyframes ping-slow {
       0% {
         transform: scale(0.2);
         opacity: 0.8;
       }
       80%, 100% {
         transform: scale(2);
         opacity: 0;
       }
     }
     .animate-ping-slow {
       animation: ping-slow 2.5s cubic-bezier(0, 0, 0.2, 1) infinite;
     }
     @keyframes fadeIn {
       from { opacity: 0; }
       to { opacity: 1; }
     }
     .animate-fadeIn {
       animation: fadeIn 0.3s ease-in;
     }
     @keyframes glow {
       0% {
         box-shadow: 0 0 5px rgba(20, 184, 166, 0.3);
       }
       50% {
         box-shadow: 0 0 20px rgba(20, 184, 166, 0.6);
       }
       100% {
         box-shadow: 0 0 5px rgba(20, 184, 166, 0.3);
       }
     }
     .ring-teal-500\/20 {
       box-shadow: 0 0 10px rgba(20, 184, 166, 0.3);
       animation: glow 2s ease-in-out infinite;
     }
    `}
  </style>
);

// Define VoiceRecognitionClient interface
interface VoiceRecognitionResult {
  text: string;
  confidence?: number;
  isFinal?: boolean;
}

interface VoiceRecognitionClient {
  apiBaseUrl: string;
  isListening: boolean;
  selectedMicrophoneIndex: number | null;
  setRecognitionResultCallback: (callback: (result: VoiceRecognitionResult) => void) => void;
  setErrorCallback: (callback: (error: string) => void) => void;
  setStatusCallback: (callback: (status: string) => void) => void;
  startRecognition: () => Promise<VoiceRecognitionResult | null>;
  getMicrophones: () => Promise<any[]>;
  setMicrophone: (index: number) => void;
  stopListening: () => void;
}

// Extend Window interface
declare global {
  interface Window {
    VoiceRecognitionClient?: new (apiBaseUrl?: string) => VoiceRecognitionClient;
    voiceClient?: VoiceRecognitionClient;
  }
}

// Initial welcome message
const welcomeMessage: MessageType = {
  id: '1',
  text: 'Hello! How can I help you with your studies today?',
  sender: 'ai',
  timestamp: new Date(Date.now() - 60000 * 5),
}

// Subject options for dropdown
const subjects = [
  'All Subjects',
  'Mathematics',
  'Physics',
  'Chemistry',
  'Biology',
  'Computer Science',
  'English',
  'History',
]

interface LocationState {
  initialQuery?: string;
  autoSend?: boolean;
  newChat?: boolean;
}

// Add type definitions for processMessage callbacks
type MessageCallback = (message: MessageType) => void;
type ErrorCallback = (error: MessageType) => void;

const ChatPage = () => {
  const location = useLocation();
  const state = location.state as LocationState;
  const initialQuery = state?.initialQuery || '';
  const shouldAutoSend = state?.autoSend || false;
  const isNewChat = state?.newChat || false;
  
  const [messages, setMessages] = useState<MessageType[]>(() => {
    if (isNewChat) return [welcomeMessage];
    const savedMessages = localStorage.getItem('chatHistory');
    return savedMessages ? JSON.parse(savedMessages) : [welcomeMessage];
  });
  const [newMessage, setNewMessage] = useState(initialQuery)
  const [subject, setSubject] = useState(subjects[0])
  const [isLoading, setIsLoading] = useState(false)
  const [showDropdown, setShowDropdown] = useState(false)
  const [regenerating, setRegenerating] = useState(false)
  const [animateIn, setAnimateIn] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const [showError, setShowError] = useState<string | null>(null);
  const [showTranscriptionSuccess, setShowTranscriptionSuccess] = useState(false);
  const [lastTranscript, setLastTranscript] = useState('');
  const [isListening, setIsListening] = useState(false)
  const [voiceClient, setVoiceClient] = useState<VoiceRecognitionClient | null>(null)
  const [isInputFocused, setIsInputFocused] = useState(false);
  const initialQueryProcessedRef = useRef(false);
  
  // Save messages to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('chatHistory', JSON.stringify(messages));
  }, [messages]);
  
  // Initialize the voice client from the global instance
  useEffect(() => {
    console.log("Initializing voice recognition client...");
    
    // Helper function to set up the voice client
    function setupVoiceClient(client: VoiceRecognitionClient) {
      // Set up event handlers
      client.setRecognitionResultCallback((result: VoiceRecognitionResult) => {
        console.log("Voice recognition result:", result);
        if (result && result.text) {
          setNewMessage(result.text);
          setLastTranscript(result.text);
          setShowTranscriptionSuccess(true);
          setTimeout(() => setShowTranscriptionSuccess(false), 3000);
        }
      });
      
      client.setErrorCallback((error: string) => {
        console.error('Voice recognition error:', error);
        setIsListening(false);
        setShowError(error);
        setTimeout(() => setShowError(null), 5000);
      });
      
      client.setStatusCallback((status: string) => {
        console.log('Voice recognition status:', status);
        if (status === 'Recognition complete' || status === 'Recognition stopped by user') {
          setIsListening(false);
        }
      });
      
      // Store client
      setVoiceClient(client);
    }
    
    // Helper function to create a new voice client
    function createNewVoiceClient() {
      try {
        // Use the API URL from env if available and is a valid string
        const apiBaseUrl = import.meta.env.VITE_API_URL && typeof import.meta.env.VITE_API_URL === 'string' 
          ? `${import.meta.env.VITE_API_URL}/voice-recognition`
          : '/api/voice-recognition';
          
        console.log("Creating new voice client with API base URL:", apiBaseUrl);
        
        // Create a new instance with our API base URL
        if (window.VoiceRecognitionClient) {
          const client = new window.VoiceRecognitionClient(apiBaseUrl);
          setupVoiceClient(client);
          console.log("Voice client ready (created new instance)");
        } else {
          throw new Error("VoiceRecognitionClient constructor not available");
        }
      } catch (error) {
        console.error("Error creating new voice client:", error);
        setShowError("Failed to initialize voice recognition");
      }
    }
    
    const loadVoiceClient = async () => {
      // Try to use the global voice client if it exists
      if (window.voiceClient) {
        try {
          // Use the existing global instance
          const client = window.voiceClient;
          setupVoiceClient(client);
          console.log("Voice client ready from global instance");
        } catch (error) {
          console.error("Error initializing global voice client:", error);
          createNewVoiceClient();
        }
      } 
      // If global client doesn't exist but the class constructor does
      else if (window.VoiceRecognitionClient) {
        console.log("No global voice client instance, creating a new one");
        createNewVoiceClient();
      } 
      // No voice recognition available - try to load it dynamically
      else {
        console.error("Voice recognition client not available");
        console.log("Window object properties:", Object.keys(window));
        
        // In production or development, try to dynamically load the script
        const script = document.createElement('script');
        
        // Base URL is different in development vs production
        const baseUrl = import.meta.env.DEV ? '' : '';
        script.src = `${baseUrl}/static/voice_recognition/js/voice_client.js`;
        
        script.onload = () => {
          console.log("Voice client script loaded dynamically");
          if (window.VoiceRecognitionClient) {
            createNewVoiceClient();
          } else {
            console.error("Voice client loaded but constructor not found");
            setShowError("Voice client loaded but constructor not found");
            
            // Debug what was loaded
            console.log("Script content loaded from:", script.src);
            console.log("Window keys after load:", Object.keys(window));
          }
        };
        
        script.onerror = (e) => {
          console.error("Failed to load voice client script:", e);
          setShowError(`Failed to load voice recognition from ${script.src}`);
          
          // Try one more alternative location in case the path is different in production
          const altScript = document.createElement('script');
          altScript.src = `/voice_client.js`;
          
          altScript.onload = () => {
            console.log("Voice client script loaded from alternative location");
            if (window.VoiceRecognitionClient) {
              createNewVoiceClient();
            } else {
              setShowError("Voice client alternative load failed");
            }
          };
          
          altScript.onerror = () => {
            console.error("Failed to load voice client from alternative location");
            setShowError("Voice recognition not available - all loading attempts failed");
          };
          
          document.body.appendChild(altScript);
        };
        
        document.body.appendChild(script);
      }
    };
    
    loadVoiceClient();
    
  }, []);
  
  // Voice recognition handler
  const handleVoiceRecognition = async () => {
    if (!voiceClient) {
      setShowError("Voice recognition not available");
      return;
    }
    
    if (isListening) {
      voiceClient.stopListening();
      setIsListening(false);
    } else {
      setIsListening(true);
      try {
        await voiceClient.startRecognition();
      } catch (error) {
        setIsListening(false);
        setShowError(error instanceof Error ? error.message : String(error));
      }
    }
  };
  
  // Animate in after component mounts
  useEffect(() => {
    setAnimateIn(true);
  }, []);
  
  // Submit initial query if provided
  useEffect(() => {
    if (initialQuery && messages.length === 1 && !initialQueryProcessedRef.current) {
      initialQueryProcessedRef.current = true;
      const currentMessage = initialQuery;
      
      // Add user message
      const userMessage = createUserMessage(currentMessage);
      setMessages(prev => [...prev, userMessage]);
      
      setIsLoading(true);
      setNewMessage('');
      
      // Process the message
      processMessage(currentMessage)
        .then((aiMessage) => {
          setMessages(prev => [...prev, aiMessage]);
          setIsLoading(false);
        })
        .catch((error) => {
          const errorMessage: MessageType = {
            id: Date.now().toString(),
            text: `Error: ${error.message}`,
            sender: 'ai',
            timestamp: new Date()
          };
          setMessages(prev => [...prev, errorMessage]);
          setIsLoading(false);
        });
    }
  }, [initialQuery, shouldAutoSend]);
  
  // Auto-scroll to bottom when messages update
  useEffect(() => {
    if (messagesEndRef.current) {
      // Wait for any animations to complete
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  }, [messages]);
  
  // Auto resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`
    }
  }, [newMessage])
  
  // Add keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + K to focus on input
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        textareaRef.current?.focus();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);
  
  // Handle auto-send when navigating from homepage
  useEffect(() => {
    if (location.state?.autoSend && location.state?.initialQuery && !initialQueryProcessedRef.current) {
      setNewMessage(location.state.initialQuery);
      handleSendMessage();
      initialQueryProcessedRef.current = true;
    }
  }, [location.state]);
  
  // Handle sending a message
  const handleSendMessage = async () => {
    if (!newMessage.trim() || isLoading) return;
    
    const currentMessage = newMessage;
    // Check if this message is already in the chat
    const isDuplicate = messages.some(msg => 
      msg.text === currentMessage && 
      msg.sender === 'user' && 
      Date.now() - new Date(msg.timestamp).getTime() < 5000 // Within last 5 seconds
    );
    
    if (isDuplicate) {
      return;
    }

    // Check if this is the initial query being sent again
    if (currentMessage === initialQuery && initialQueryProcessedRef.current) {
      return;
    }
    
    // Add user message
    const userMessage = createUserMessage(currentMessage);
    setMessages(prev => [...prev, userMessage]);
    
    setIsLoading(true);
    setNewMessage('');
    
    // Process the message
    processMessage(currentMessage)
      .then((aiMessage) => {
        setMessages(prev => [...prev, aiMessage]);
        setIsLoading(false);
      })
      .catch((error) => {
        const errorMessage: MessageType = {
          id: Date.now().toString(),
          text: `Error: ${error.message}`,
          sender: 'ai',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
        setIsLoading(false);
      });
  };
  
  // Regenerate last AI response
  const handleRegenerate = () => {
    if (isLoading || regenerating) return;
    
    setRegenerating(true);
    
    // Find last user message
    const lastUserMessageIndex = [...messages].reverse().findIndex(m => m.sender === 'user');
    if (lastUserMessageIndex === -1) {
      setRegenerating(false);
      return;
    }
    
    const lastUserMessage = messages[messages.length - 1 - lastUserMessageIndex];
    
    // Remove the last AI response
    const newMessages = messages.slice(0, -1);
    setMessages(newMessages);
    
    // Show loading indicator
    setIsLoading(true);
    
    // Generate new response
    processMessage(lastUserMessage.text)
      .then((aiMessage) => {
        setMessages(prev => [...prev, aiMessage]);
        setIsLoading(false);
        setRegenerating(false);
      })
      .catch((error) => {
        const errorMessage: MessageType = {
          id: Date.now().toString(),
          text: `Error: ${error.message}`,
          sender: 'ai',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
        setIsLoading(false);
        setRegenerating(false);
      });
  };
  
  // Handle Enter key to send message
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }
  
  // Toggle subject dropdown
  const toggleDropdown = () => {
    setShowDropdown(!showDropdown)
  }

  // Handle subject selection
  const handleSubjectChange = (selectedSubject: string) => {
    setSubject(selectedSubject)
    setShowDropdown(false)
  }

  // Handle feedback clicks (thumbs up/down)
  const handleFeedback = (isPositive: boolean) => {
    // In a real app, you would send this feedback to your backend
    alert(`Thank you for your ${isPositive ? 'positive' : 'negative'} feedback!`);
  };

  // Display text with highlight on last transcription
  const displayMessageWithHighlight = () => {
    if (!lastTranscript || !showTranscriptionSuccess) {
      return newMessage;
    }
    
    // Find the position of the last transcript in the message
    const lastIndex = newMessage.lastIndexOf(lastTranscript);
    
    if (lastIndex === -1) {
      return newMessage;
    }
    
    // Split the message into parts
    const beforeTranscript = newMessage.substring(0, lastIndex);
    const afterTranscript = newMessage.substring(lastIndex + lastTranscript.length);
    
    // Return the message with the last transcript highlighted
    return (
      <>
        {beforeTranscript}
        <span className="bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200 px-1 rounded">
          {lastTranscript}
        </span>
        {afterTranscript}
      </>
    );
  };

  return (
    <div 
      className="flex flex-col h-full"
      ref={chatContainerRef}
    >
      {/* Include voice animations */}
      <VoiceAnimations />
      
      {/* Messages Container */}
      <div 
        className={`flex-1 overflow-y-auto pb-32 transition-opacity duration-500 ${
          animateIn ? 'opacity-100' : 'opacity-0'
        }`}
      >
        {/* Subject Selector positioned at top */}
        <div className="sticky top-0 z-10 bg-white dark:bg-black py-2 px-4 border-b border-gray-200 dark:border-gray-800 backdrop-blur-sm bg-opacity-80 dark:bg-opacity-80">
          <div className="relative max-w-xs">
            <button 
              onClick={toggleDropdown}
              className="flex items-center justify-between w-full p-2 text-sm rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors duration-200"
            >
              <span>{subject}</span>
              <FiChevronDown className={`transition-transform duration-200 ${showDropdown ? 'rotate-180' : ''}`} />
            </button>
            
            {showDropdown && (
              <div className="absolute mt-1 w-full z-20 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-md shadow-lg animate-fadeIn">
                <ul className="py-1 max-h-56 overflow-y-auto">
                  {subjects.map((sub) => (
                    <li 
                      key={sub} 
                      className={`px-3 py-2 text-sm cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors duration-150 ${
                        subject === sub ? 'bg-gray-100 dark:bg-gray-800' : ''
                      }`}
                      onClick={() => handleSubjectChange(sub)}
                    >
                      {sub}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
        
        {/* Welcome message */}
        {messages.length === 1 && (
          <div className="flex justify-center items-center py-10 px-4 animate-fadeIn">
            <div className="text-center max-w-lg">
              <h2 className="text-2xl font-bold mb-4 text-gray-800 dark:text-white">Welcome to Student AI</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Ask any academic question and get instant, intelligent answers to help with your studies.
              </p>
              <div className="flex items-center justify-center text-sm text-gray-500 dark:text-gray-400">
                <FiCommand className="mr-1" size={14} />
                <span>Press </span>
                <kbd className="mx-1 px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded border border-gray-300 dark:border-gray-700 font-sans text-xs">Ctrl/⌘</kbd>
                <span>+</span>
                <kbd className="mx-1 px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded border border-gray-300 dark:border-gray-700 font-sans text-xs">K</kbd>
                <span> to focus on the input</span>
              </div>
            </div>
          </div>
        )}
        
        {/* Chat messages with alternating backgrounds */}
        <div className="space-y-4">
          {messages.map((message, index) => (
            <div 
              key={message.id} 
              className={`py-6 ${message.sender === 'ai' ? 'bg-white dark:bg-black' : 'bg-gray-50 dark:bg-gray-900'}`}
              style={{
                animationDelay: `${index * 0.1}s`,
                opacity: 0,
                animation: 'fadeSlideUp 0.3s ease forwards'
              }}
            >
              <div className="max-w-3xl mx-auto px-4">
                <ChatBubble message={message} />
                
                {/* Show feedback buttons only for AI responses and not the initial welcome message */}
                {message.sender === 'ai' && index > 0 && (
                  <div className="flex items-center mt-2 ml-12 space-x-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                    <button 
                      onClick={() => handleFeedback(true)}
                      className="p-1 rounded-md text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-all duration-200"
                      aria-label="Thumbs up"
                    >
                      <FiThumbsUp size={14} />
                    </button>
                    <button 
                      onClick={() => handleFeedback(false)}
                      className="p-1 rounded-md text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-all duration-200"
                      aria-label="Thumbs down"
                    >
                      <FiThumbsDown size={14} />
                    </button>
                    
                    {/* Regenerate button - only show for the last AI message */}
                    {index === messages.length - 1 && (
                      <button 
                        onClick={handleRegenerate}
                        disabled={regenerating || isLoading}
                        className="flex items-center ml-2 p-1 px-2 text-xs rounded-md text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 border border-gray-200 dark:border-gray-800 transition-all duration-200"
                        aria-label="Regenerate response"
                      >
                        <FiRefreshCw className={`mr-1 ${regenerating ? 'animate-spin' : ''}`} size={12} />
                        Regenerate
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {/* Loading indicator */}
          {isLoading && (
            <div className="py-6 bg-white dark:bg-black animate-fadeIn">
              <div className="max-w-3xl mx-auto px-4">
                <div className="flex items-start">
                  <div className="flex-shrink-0 mr-4">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-teal-400 to-teal-600 flex items-center justify-center text-white pulse-animation">
                      <div className="w-5 h-5 rounded-full bg-teal-400 animate-ping absolute"></div>
                      AI
                    </div>
                  </div>
                  <div className="flex space-x-2 p-4 bg-gray-100 dark:bg-gray-900 rounded-lg">
                    <div className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-600 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-600 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-600 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Invisible element for auto-scrolling */}
          <div ref={messagesEndRef} className="h-4" />
        </div>
      </div>
      
      {/* Show error messages */}
      {showError && (
        <div className="fixed top-4 left-1/2 transform -translate-x-1/2 bg-red-100 border border-red-400 text-red-700 px-4 py-2 rounded z-50 shadow-md">
          {showError}
        </div>
      )}
      
      {/* Input Area - Fixed at bottom with glass effect */}
      <div className="fixed bottom-0 left-0 right-0 bg-white dark:bg-black border-t border-gray-200 dark:border-gray-800 p-4">
        <div className="max-w-3xl mx-auto">
          <div className={`relative flex items-end ${
            isInputFocused 
              ? 'bg-white dark:bg-gray-950 shadow-lg ring-4 ring-teal-500/20 border-teal-500 dark:border-teal-500 transform scale-[1.02]' 
              : 'bg-gray-100 dark:bg-gray-900 border-transparent'
            } rounded-lg p-2 pr-3 border-2 transition-all duration-300`}>
            {/* Attachment button */}
            <button 
              className="p-2 rounded-full text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-800 transition-colors duration-200 mr-1"
              aria-label="Attach file"
            >
              <FiPaperclip />
            </button>
            
            {/* Voice recognition button */}
            <button
              onClick={handleVoiceRecognition}
              className={`relative p-2 rounded-full transition-all duration-300 mr-1 ${
                isListening 
                  ? 'bg-primary-50 dark:bg-primary-900 text-primary-600 dark:text-primary-400 transform scale-110' 
                  : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-800'
              }`}
              aria-label={isListening ? "Stop listening" : "Start voice recognition"}
            >
              <FiMic className={isListening ? 'relative z-10' : ''} />
              {isListening && (
                <span className="absolute inset-0 flex items-center justify-center">
                  <span className="absolute w-10 h-10 rounded-full bg-primary-500 opacity-20 animate-ping-slow"></span>
                  <span className="absolute w-7 h-7 rounded-full bg-primary-400 opacity-40 animate-ping" style={{ animationDelay: '0.5s' }}></span>
                </span>
              )}
            </button>
            
            {/* Message input */}
            {showTranscriptionSuccess ? (
              <div 
                className="flex-1 outline-none bg-transparent py-2 px-2 max-h-32 overflow-y-auto"
                onClick={() => textareaRef.current?.focus()}
              >
                {displayMessageWithHighlight()}
              </div>
            ) : (
              <textarea
                ref={textareaRef}
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                onFocus={() => setIsInputFocused(true)}
                onBlur={() => setIsInputFocused(false)}
                placeholder="Type your message or use voice recognition..."
                className="flex-1 outline-none bg-transparent resize-none max-h-32 py-2 px-2"
                rows={1}
              />
            )}
            
            {/* Send button */}
            <button
              onClick={handleSendMessage}
              disabled={!newMessage.trim() || isLoading}
              title="Send message"
              className={`p-2 rounded-lg transition-all duration-200 ${
                newMessage.trim() && !isLoading
                  ? 'bg-primary-500 text-white shadow-md hover:bg-primary-600 hover:shadow-lg active:scale-95'
                  : 'bg-gray-200 dark:bg-gray-800 text-gray-400 cursor-not-allowed'
              }`}
            >
              <FiSend size={18} className={newMessage.trim() && !isLoading ? 'hover:animate-pulse' : ''} />
            </button>
          </div>
        </div>
      </div>
      
      {/* Keyboard shortcut indicator */}
      <div className="fixed bottom-4 right-4">
        <div className="bg-white dark:bg-gray-900 text-xs px-2 py-1 rounded-md border border-gray-200 dark:border-gray-700 flex items-center">
          <FiCommand className="mr-1" size={12} />
          <span>Press </span>
          <kbd className="mx-1 px-1 bg-gray-100 dark:bg-gray-800 rounded border border-gray-300 dark:border-gray-700 text-xs">Ctrl/⌘</kbd>
          <span>+</span>
          <kbd className="mx-1 px-1 bg-gray-100 dark:bg-gray-800 rounded border border-gray-300 dark:border-gray-700 text-xs">K</kbd>
          <span> to focus on the input</span>
        </div>
      </div>
      
      {/* Voice recognition status indicator */}
      {isListening && (
        <div className="fixed bottom-20 left-1/2 transform -translate-x-1/2 bg-primary-50 dark:bg-primary-900 
                        text-primary-600 dark:text-primary-400 px-3 py-1 rounded-full text-xs 
                        border border-primary-200 dark:border-primary-800 shadow-md animate-fadeIn flex items-center">
          <span className="inline-block w-2 h-2 rounded-full bg-primary-500 animate-pulse mr-2"></span>
          <span>Listening...</span>
        </div>
      )}
      
      {/* Transcription success toast */}
      {showTranscriptionSuccess && (
        <div className="fixed bottom-20 left-1/2 transform -translate-x-1/2 bg-green-50 dark:bg-green-900 
                        text-green-600 dark:text-green-400 px-3 py-1 rounded-full text-xs 
                        border border-green-200 dark:border-green-800 shadow-md animate-fadeIn">
          Transcription successful!
        </div>
      )}
    </div>
  )
}

export default ChatPage 