import { useState, useEffect } from 'react'
import { FiMessageSquare, FiChevronLeft } from 'react-icons/fi'
import { useNavigate } from 'react-router-dom'
import { MessageType } from '../components/ChatBubble'

interface Chat {
  id: string
  title: string
  messages: MessageType[]
  lastUpdated: Date
}

const MyQuestions = () => {
  const navigate = useNavigate()
  const [chats, setChats] = useState<Chat[]>([])
  const [selectedChat, setSelectedChat] = useState<Chat | null>(null)
  
  useEffect(() => {
    try {
      const savedHistory = localStorage.getItem('chatHistory')
      if (savedHistory) {
        const parsedHistory = JSON.parse(savedHistory) as MessageType[]
        const formattedHistory = parsedHistory.map(message => ({
          ...message,
          timestamp: new Date(message.timestamp)
        }))
        const chatGroups = formattedHistory.reduce((groups: Record<string, MessageType[]>, message) => {
          const chatId = message.timestamp.toLocaleDateString()
          if (!groups[chatId]) {
            groups[chatId] = []
          }
          groups[chatId].push(message)
          return groups
        }, {})
        const chatList: Chat[] = Object.entries(chatGroups).map(([id, messages]) => ({
          id,
          title: `Chat from ${id}`,
          messages: messages.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime()),
          lastUpdated: new Date(Math.max(...messages.map(m => m.timestamp.getTime())))
        }))
        chatList.sort((a, b) => b.lastUpdated.getTime() - a.lastUpdated.getTime())
        setChats(chatList)
      }
    } catch (error) {
      console.error('Error loading chat history:', error)
    }
  }, [])

  const handleChatClick = (chat: Chat) => setSelectedChat(chat)
  const handleBack = () => setSelectedChat(null)

  if (selectedChat) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center mb-6">
          <button 
            onClick={handleBack}
            className="mr-4 p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full transition-colors"
          >
            <FiChevronLeft size={20} />
          </button>
          <h1 className="text-2xl font-bold">{selectedChat.title}</h1>
        </div>
        <div className="space-y-6">
          {selectedChat.messages.map((message) => (
            <div 
              key={message.id} 
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                message.sender === 'user' 
                  ? 'bg-primary-500 text-white rounded-tr-none' 
                  : 'bg-gray-100 dark:bg-gray-900 text-gray-800 dark:text-gray-100 rounded-tl-none'
              }`}>
                <div className="prose dark:prose-invert max-w-none">
                  {message.text.split('\n').map((line, i) => (
                    <p key={i} className={message.sender === 'user' ? 'text-white' : 'text-gray-700 dark:text-gray-300'}>
                      {line}
                    </p>
                  ))}
                </div>
                <div className={`text-xs mt-1 ${
                  message.sender === 'user' ? 'text-white/70' : 'text-gray-500 dark:text-gray-400'
                }`}>
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Chat History</h1>
      {chats.length === 0 ? (
        <div className="text-center py-12">
          <FiMessageSquare className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <p className="text-gray-500 dark:text-gray-400">No chat history yet</p>
        </div>
      ) : (
        <div className="space-y-2">
          {chats.map((chat) => (
            <button
              key={chat.id}
              onClick={() => handleChatClick(chat)}
              className="w-full text-left p-4 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="font-medium text-gray-900 dark:text-white">{chat.title}</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {chat.messages.length} messages
                  </p>
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  {chat.lastUpdated.toLocaleString()}
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default MyQuestions 