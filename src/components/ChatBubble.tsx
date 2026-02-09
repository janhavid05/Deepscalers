import React from 'react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { formatDistanceToNow } from 'date-fns'

export interface MessageType {
  id: string
  text: string
  sender: 'user' | 'ai'
  timestamp: Date
}

interface ChatBubbleProps {
  message: MessageType
}

const ChatBubble: React.FC<ChatBubbleProps> = ({ message }) => {
  const { text, sender, timestamp } = message
  const formattedTime = formatDistanceToNow(new Date(timestamp), { addSuffix: true })
  
  // Custom renderer for code blocks
  const components = {
    code({ node, inline, className, children, ...props }: any) {
      const match = /language-(\w+)/.exec(className || '')
      const language = match ? match[1] : 'text'
      
      return !inline ? (
        <div className="my-4 rounded-md overflow-hidden">
          <div className="flex items-center justify-between bg-gray-800 px-4 py-2 text-xs text-gray-200">
            <span>{language}</span>
            <button 
              onClick={() => {
                navigator.clipboard.writeText(String(children).replace(/\n$/, ''))
              }}
              className="px-2 py-1 text-xs rounded hover:bg-gray-700 transition-colors"
            >
              Copy
            </button>
          </div>
          <SyntaxHighlighter
            language={language}
            style={oneDark}
            customStyle={{ margin: 0, borderRadius: '0 0 0.375rem 0.375rem' }}
            showLineNumbers
            wrapLines
          >
            {String(children).replace(/\n$/, '')}
          </SyntaxHighlighter>
        </div>
      ) : (
        <code className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded font-mono text-sm">
          {children}
        </code>
      )
    },
    p: ({ children }: any) => <p className="mb-4 last:mb-0">{children}</p>,
    ul: ({ children }: any) => <ul className="list-disc pl-6 mb-4 space-y-2">{children}</ul>,
    ol: ({ children }: any) => <ol className="list-decimal pl-6 mb-4 space-y-2">{children}</ol>,
    li: ({ children }: any) => <li className="mb-1">{children}</li>,
    h1: ({ children }: any) => <h1 className="text-2xl font-bold mb-4 mt-6">{children}</h1>,
    h2: ({ children }: any) => <h2 className="text-xl font-bold mb-3 mt-5">{children}</h2>,
    h3: ({ children }: any) => <h3 className="text-lg font-bold mb-2 mt-4">{children}</h3>,
    a: ({ href, children }: any) => (
      <a href={href} className="text-primary-500 hover:underline" target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    ),
    blockquote: ({ children }: any) => (
      <blockquote className="border-l-4 border-gray-300 dark:border-gray-700 pl-4 italic my-4">
        {children}
      </blockquote>
    ),
    table: ({ children }: any) => (
      <div className="overflow-x-auto my-4">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
          {children}
        </table>
      </div>
    ),
    th: ({ children }: any) => (
      <th className="px-3 py-2 bg-gray-100 dark:bg-gray-800 text-left text-sm font-semibold text-gray-900 dark:text-white">
        {children}
      </th>
    ),
    td: ({ children }: any) => (
      <td className="px-3 py-2 whitespace-nowrap text-sm border-b border-gray-200 dark:border-gray-800">
        {children}
      </td>
    ),
  }
  
  return (
    <div className={`flex ${sender === 'user' ? 'justify-end' : 'justify-start'} mb-4 group`}>
      {sender === 'ai' && (
        <div className="flex-shrink-0 mr-4">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-teal-400 to-teal-600 flex items-center justify-center text-white">
            AI
          </div>
        </div>
      )}
      
      <div className={`
        max-w-[80%] rounded-2xl px-4 py-3 
        ${sender === 'user' 
          ? 'bg-primary-500 text-white rounded-tr-none' 
          : 'bg-gray-100 dark:bg-gray-900 text-gray-800 dark:text-gray-100 rounded-tl-none'
        }
      `}>
        {sender === 'ai' ? (
          <div className="prose dark:prose-invert prose-sm max-w-none">
            <ReactMarkdown components={components}>
              {text}
            </ReactMarkdown>
          </div>
        ) : (
          <div>{text}</div>
        )}
        
        <div className={`
          text-xs mt-1 opacity-70
          ${sender === 'user' ? 'text-right text-white/70' : 'text-gray-500 dark:text-gray-400'}
        `}>
          {formattedTime}
        </div>
      </div>
      
      {sender === 'user' && (
        <div className="flex-shrink-0 ml-4">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-gray-700 to-gray-900 dark:from-gray-500 dark:to-gray-700 flex items-center justify-center text-white">
            You
          </div>
        </div>
      )}
    </div>
  )
}

export default ChatBubble 