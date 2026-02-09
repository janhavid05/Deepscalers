import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiMessageSquare } from 'react-icons/fi';

interface ChatHistoryItem {
  id: string;
  title: string;
  timestamp: Date;
}

const Questions = () => {
  const navigate = useNavigate();
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>(() => {
    const savedHistory = localStorage.getItem('chatHistory');
    return savedHistory ? JSON.parse(savedHistory) : [];
  });

  const handleChatClick = (chatId: string) => {
    navigate('/chat', { state: { chatId } });
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">My Content</h1>
      
      {/* Chat History Section */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">My Questions</h2>
        <div className="bg-white rounded-lg shadow-lg p-6">
          {chatHistory.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              <FiMessageSquare className="w-12 h-12 mx-auto mb-4" />
              <p>No questions yet</p>
            </div>
          ) : (
            <div className="space-y-4">
              {chatHistory.map((chat) => (
                <div
                  key={chat.id}
                  onClick={() => handleChatClick(chat.id)}
                  className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors duration-200"
                >
                  <div className="font-medium text-gray-800 truncate">{chat.title}</div>
                  <div className="text-sm text-gray-500">
                    {new Date(chat.timestamp).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Other content sections can be added here */}
    </div>
  );
};

export default Questions; 