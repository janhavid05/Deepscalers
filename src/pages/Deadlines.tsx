import React, { useState } from 'react'
import { FiCalendar, FiClock, FiCheck, FiTrash, FiPlus } from 'react-icons/fi'

// Types
interface Deadline {
  id: string
  title: string
  description: string
  dueDate: Date
  subject: string
  completed: boolean
  priority: 'low' | 'medium' | 'high'
}

// Mock data
const initialDeadlines: Deadline[] = [
  {
    id: '1',
    title: 'Physics Assignment: Newton\'s Laws',
    description: 'Solve problems 1-10 from Chapter 4',
    dueDate: new Date(Date.now() + 86400000), // Tomorrow
    subject: 'Physics',
    completed: false,
    priority: 'high',
  },
  {
    id: '2',
    title: 'Math Quiz',
    description: 'Online quiz on Calculus',
    dueDate: new Date(Date.now() + 86400000 * 3), // 3 days from now
    subject: 'Mathematics',
    completed: false,
    priority: 'medium',
  },
  {
    id: '3',
    title: 'English Essay',
    description: 'Write a 1000-word essay on Shakespeare',
    dueDate: new Date(Date.now() + 86400000 * 7), // 7 days from now
    subject: 'English',
    completed: false,
    priority: 'medium',
  },
  {
    id: '4',
    title: 'Biology Lab Report',
    description: 'Complete lab report on cellular respiration',
    dueDate: new Date(Date.now() - 86400000 * 2), // 2 days ago
    subject: 'Biology',
    completed: true,
    priority: 'high',
  },
]

const Deadlines = () => {
  const [deadlines, setDeadlines] = useState<Deadline[]>(initialDeadlines)
  const [showForm, setShowForm] = useState(false)
  const [filter, setFilter] = useState<'all' | 'completed' | 'pending'>('all')
  
  // New deadline form state
  const [newDeadline, setNewDeadline] = useState<Omit<Deadline, 'id' | 'completed'>>({
    title: '',
    description: '',
    dueDate: new Date(),
    subject: '',
    priority: 'medium',
  })
  
  // Toggle completion status
  const toggleCompletion = (id: string) => {
    setDeadlines(deadlines.map(d => 
      d.id === id ? { ...d, completed: !d.completed } : d
    ))
  }
  
  // Delete a deadline
  const handleDelete = (id: string) => {
    setDeadlines(deadlines.filter(d => d.id !== id))
  }
  
  // Add a new deadline
  const handleAddDeadline = (e: React.FormEvent) => {
    e.preventDefault()
    
    const newId = Date.now().toString()
    setDeadlines([
      ...deadlines,
      {
        ...newDeadline,
        id: newId,
        completed: false,
      },
    ])
    
    // Reset form
    setNewDeadline({
      title: '',
      description: '',
      dueDate: new Date(),
      subject: '',
      priority: 'medium',
    })
    setShowForm(false)
  }
  
  // Get filtered deadlines
  const filteredDeadlines = deadlines.filter(d => {
    if (filter === 'all') return true
    if (filter === 'completed') return d.completed
    if (filter === 'pending') return !d.completed
    return true
  })
  
  // Sort deadlines by due date (closest first)
  const sortedDeadlines = [...filteredDeadlines].sort((a, b) => a.dueDate.getTime() - b.dueDate.getTime())
  
  // Helper for formatting dates
  const formatDueDate = (date: Date) => {
    const today = new Date()
    const tomorrow = new Date(today)
    tomorrow.setDate(tomorrow.getDate() + 1)
    
    if (date.toDateString() === today.toDateString()) {
      return 'Today'
    } else if (date.toDateString() === tomorrow.toDateString()) {
      return 'Tomorrow'
    } else {
      return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
    }
  }
  
  // Calculate days remaining
  const getDaysRemaining = (date: Date) => {
    const today = new Date()
    today.setHours(0, 0, 0, 0)
    const dueDate = new Date(date)
    dueDate.setHours(0, 0, 0, 0)
    
    const diffTime = dueDate.getTime() - today.getTime()
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))
    
    if (diffDays < 0) {
      return `${Math.abs(diffDays)} day${Math.abs(diffDays) !== 1 ? 's' : ''} overdue`
    } else if (diffDays === 0) {
      return 'Due today'
    } else {
      return `${diffDays} day${diffDays !== 1 ? 's' : ''} left`
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Deadlines</h1>
        <div className="flex space-x-2">
          <button 
            onClick={() => setFilter('all')}
            className={`px-3 py-1 rounded ${filter === 'all' ? 'bg-primary-100 text-primary-600 dark:bg-primary-900 dark:text-primary-300' : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'}`}
          >
            All
          </button>
          <button 
            onClick={() => setFilter('pending')}
            className={`px-3 py-1 rounded ${filter === 'pending' ? 'bg-primary-100 text-primary-600 dark:bg-primary-900 dark:text-primary-300' : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'}`}
          >
            Pending
          </button>
          <button 
            onClick={() => setFilter('completed')}
            className={`px-3 py-1 rounded ${filter === 'completed' ? 'bg-primary-100 text-primary-600 dark:bg-primary-900 dark:text-primary-300' : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'}`}
          >
            Completed
          </button>
        </div>
      </div>
      
      {/* Add Deadline Button */}
      <button
        onClick={() => setShowForm(true)}
        className="flex items-center px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600"
      >
        <FiPlus className="mr-2" /> Add Deadline
      </button>
      
      {/* Add Deadline Form */}
      {showForm && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold mb-4">Add New Deadline</h2>
          <form onSubmit={handleAddDeadline}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-gray-700 dark:text-gray-300 mb-1">Title</label>
                <input
                  type="text"
                  value={newDeadline.title}
                  onChange={(e) => setNewDeadline({ ...newDeadline, title: e.target.value })}
                  className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                  required
                />
              </div>
              <div>
                <label className="block text-gray-700 dark:text-gray-300 mb-1">Subject</label>
                <input
                  type="text"
                  value={newDeadline.subject}
                  onChange={(e) => setNewDeadline({ ...newDeadline, subject: e.target.value })}
                  className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                  required
                />
              </div>
              <div>
                <label className="block text-gray-700 dark:text-gray-300 mb-1">Due Date</label>
                <input
                  type="date"
                  value={newDeadline.dueDate.toISOString().slice(0, 10)}
                  onChange={(e) => setNewDeadline({ ...newDeadline, dueDate: new Date(e.target.value) })}
                  className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                  required
                />
              </div>
              <div>
                <label className="block text-gray-700 dark:text-gray-300 mb-1">Priority</label>
                <select
                  value={newDeadline.priority}
                  onChange={(e) => setNewDeadline({ ...newDeadline, priority: e.target.value as 'low' | 'medium' | 'high' })}
                  className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
            </div>
            <div className="mb-4">
              <label className="block text-gray-700 dark:text-gray-300 mb-1">Description</label>
              <textarea
                value={newDeadline.description}
                onChange={(e) => setNewDeadline({ ...newDeadline, description: e.target.value })}
                className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                rows={3}
              />
            </div>
            <div className="flex justify-end space-x-2">
              <button
                type="button"
                onClick={() => setShowForm(false)}
                className="px-4 py-2 border rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600"
              >
                Save Deadline
              </button>
            </div>
          </form>
        </div>
      )}
      
      {/* Deadlines List */}
      <div className="space-y-4">
        {sortedDeadlines.length === 0 ? (
          <div className="text-center p-6 bg-white dark:bg-gray-800 rounded-lg shadow">
            <p className="text-gray-500 dark:text-gray-400">No deadlines found.</p>
          </div>
        ) : (
          sortedDeadlines.map((deadline) => (
            <div 
              key={deadline.id} 
              className={`bg-white dark:bg-gray-800 rounded-lg shadow p-4 ${
                deadline.completed 
                  ? 'border-l-4 border-green-500' 
                  : deadline.priority === 'high' 
                    ? 'border-l-4 border-red-500' 
                    : deadline.priority === 'medium' 
                      ? 'border-l-4 border-yellow-500'
                      : 'border-l-4 border-blue-500'
              }`}
            >
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <h3 className={`text-lg font-medium ${deadline.completed ? 'line-through text-gray-500 dark:text-gray-400' : ''}`}>
                    {deadline.title}
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">{deadline.subject}</p>
                  {deadline.description && (
                    <p className="mt-2 text-gray-600 dark:text-gray-300">{deadline.description}</p>
                  )}
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => toggleCompletion(deadline.id)}
                    className={`p-2 rounded-full ${
                      deadline.completed 
                        ? 'bg-green-100 text-green-600 hover:bg-green-200 dark:bg-green-900 dark:text-green-300' 
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300'
                    }`}
                  >
                    <FiCheck size={18} />
                  </button>
                  <button
                    onClick={() => handleDelete(deadline.id)}
                    className="p-2 rounded-full bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300"
                  >
                    <FiTrash size={18} />
                  </button>
                </div>
              </div>
              
              <div className="mt-3 flex items-center text-sm">
                <div className="flex items-center text-gray-500 dark:text-gray-400 mr-4">
                  <FiCalendar className="mr-1" />
                  {formatDueDate(deadline.dueDate)}
                </div>
                <div className={`flex items-center ${
                  deadline.completed
                    ? 'text-green-500'
                    : deadline.dueDate < new Date() 
                      ? 'text-red-500' 
                      : 'text-yellow-500'
                }`}>
                  <FiClock className="mr-1" />
                  {getDaysRemaining(deadline.dueDate)}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default Deadlines 