import React, { useState } from 'react'
import { FiSave, FiUser, FiMail, FiPhone, FiLock, FiToggleLeft, FiToggleRight } from 'react-icons/fi'

// Mock user data
const initialUserData = {
  name: 'John Doe',
  email: 'john.doe@university.edu',
  phone: '+1 (555) 123-4567',
  studentId: 'STU12345',
  department: 'Computer Science',
  year: '3rd Year',
  profileImage: 'https://i.imgur.com/7LuPZzE.jpg',
}

const Settings = () => {
  const [userData, setUserData] = useState(initialUserData)
  const [notifications, setNotifications] = useState({
    emailNotifications: true,
    pushNotifications: false,
    deadlineReminders: true,
    newAnswers: true,
    marketingEmails: false,
  })
  const [isEditing, setIsEditing] = useState(false)
  const [tempUserData, setTempUserData] = useState(userData)
  
  // Toggle notification settings
  const toggleNotification = (key: keyof typeof notifications) => {
    setNotifications({
      ...notifications,
      [key]: !notifications[key],
    })
  }
  
  // Handle user data form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setUserData(tempUserData)
    setIsEditing(false)
  }
  
  const handleCancel = () => {
    setTempUserData(userData)
    setIsEditing(false)
  }

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold">Settings</h1>
      
      {/* Profile Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-semibold">Profile Information</h2>
          <button
            onClick={() => setIsEditing(!isEditing)}
            className="px-4 py-2 rounded-md text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            {isEditing ? 'Cancel' : 'Edit Profile'}
          </button>
        </div>
        
        <div className="flex flex-col md:flex-row gap-6">
          {/* Profile Image */}
          <div className="flex flex-col items-center">
            <div className="w-32 h-32 rounded-full overflow-hidden mb-3">
              <img
                src={userData.profileImage}
                alt={userData.name}
                className="w-full h-full object-cover"
              />
            </div>
            {isEditing && (
              <button className="text-sm text-primary-600 dark:text-primary-400">
                Change Photo
              </button>
            )}
          </div>
          
          {/* Profile Details */}
          <div className="flex-1">
            {isEditing ? (
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Full Name
                    </label>
                    <div className="relative">
                      <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-gray-500 dark:text-gray-400">
                        <FiUser size={16} />
                      </span>
                      <input
                        type="text"
                        value={tempUserData.name}
                        onChange={(e) => setTempUserData({ ...tempUserData, name: e.target.value })}
                        className="w-full py-2 pl-10 pr-3 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Email
                    </label>
                    <div className="relative">
                      <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-gray-500 dark:text-gray-400">
                        <FiMail size={16} />
                      </span>
                      <input
                        type="email"
                        value={tempUserData.email}
                        onChange={(e) => setTempUserData({ ...tempUserData, email: e.target.value })}
                        className="w-full py-2 pl-10 pr-3 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Phone
                    </label>
                    <div className="relative">
                      <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-gray-500 dark:text-gray-400">
                        <FiPhone size={16} />
                      </span>
                      <input
                        type="tel"
                        value={tempUserData.phone}
                        onChange={(e) => setTempUserData({ ...tempUserData, phone: e.target.value })}
                        className="w-full py-2 pl-10 pr-3 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Student ID
                    </label>
                    <div className="relative">
                      <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-gray-500 dark:text-gray-400">
                        <FiLock size={16} />
                      </span>
                      <input
                        type="text"
                        value={tempUserData.studentId}
                        readOnly
                        className="w-full py-2 pl-10 pr-3 border rounded-md bg-gray-100 dark:bg-gray-800 dark:border-gray-600"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Department
                    </label>
                    <input
                      type="text"
                      value={tempUserData.department}
                      onChange={(e) => setTempUserData({ ...tempUserData, department: e.target.value })}
                      className="w-full py-2 px-3 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Year
                    </label>
                    <select
                      value={tempUserData.year}
                      onChange={(e) => setTempUserData({ ...tempUserData, year: e.target.value })}
                      className="w-full py-2 px-3 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                    >
                      <option>1st Year</option>
                      <option>2nd Year</option>
                      <option>3rd Year</option>
                      <option>4th Year</option>
                      <option>Graduate</option>
                    </select>
                  </div>
                </div>
                
                <div className="flex justify-end space-x-2 pt-4">
                  <button
                    type="button"
                    onClick={handleCancel}
                    className="px-4 py-2 border rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="flex items-center px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600"
                  >
                    <FiSave className="mr-2" /> Save Changes
                  </button>
                </div>
              </form>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Full Name</p>
                  <p className="font-medium">{userData.name}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Email</p>
                  <p className="font-medium">{userData.email}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Phone</p>
                  <p className="font-medium">{userData.phone}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Student ID</p>
                  <p className="font-medium">{userData.studentId}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Department</p>
                  <p className="font-medium">{userData.department}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Year</p>
                  <p className="font-medium">{userData.year}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Notification Settings */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-6">Notification Settings</h2>
        
        <div className="space-y-4">
          <div className="flex justify-between items-center py-2">
            <div>
              <h3 className="font-medium">Email Notifications</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">Receive notifications via email</p>
            </div>
            <button 
              onClick={() => toggleNotification('emailNotifications')}
              className={`p-1 rounded-full ${notifications.emailNotifications ? 'text-primary-600 dark:text-primary-400' : 'text-gray-400 dark:text-gray-600'}`}
            >
              {notifications.emailNotifications ? <FiToggleRight size={28} /> : <FiToggleLeft size={28} />}
            </button>
          </div>
          
          <div className="flex justify-between items-center py-2">
            <div>
              <h3 className="font-medium">Push Notifications</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">Receive notifications on your device</p>
            </div>
            <button 
              onClick={() => toggleNotification('pushNotifications')}
              className={`p-1 rounded-full ${notifications.pushNotifications ? 'text-primary-600 dark:text-primary-400' : 'text-gray-400 dark:text-gray-600'}`}
            >
              {notifications.pushNotifications ? <FiToggleRight size={28} /> : <FiToggleLeft size={28} />}
            </button>
          </div>
          
          <div className="flex justify-between items-center py-2">
            <div>
              <h3 className="font-medium">Deadline Reminders</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">Get reminded about upcoming deadlines</p>
            </div>
            <button 
              onClick={() => toggleNotification('deadlineReminders')}
              className={`p-1 rounded-full ${notifications.deadlineReminders ? 'text-primary-600 dark:text-primary-400' : 'text-gray-400 dark:text-gray-600'}`}
            >
              {notifications.deadlineReminders ? <FiToggleRight size={28} /> : <FiToggleLeft size={28} />}
            </button>
          </div>
          
          <div className="flex justify-between items-center py-2">
            <div>
              <h3 className="font-medium">New Answers</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">Get notified when your questions are answered</p>
            </div>
            <button 
              onClick={() => toggleNotification('newAnswers')}
              className={`p-1 rounded-full ${notifications.newAnswers ? 'text-primary-600 dark:text-primary-400' : 'text-gray-400 dark:text-gray-600'}`}
            >
              {notifications.newAnswers ? <FiToggleRight size={28} /> : <FiToggleLeft size={28} />}
            </button>
          </div>
          
          <div className="flex justify-between items-center py-2">
            <div>
              <h3 className="font-medium">Marketing Emails</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">Receive marketing and promotional emails</p>
            </div>
            <button 
              onClick={() => toggleNotification('marketingEmails')}
              className={`p-1 rounded-full ${notifications.marketingEmails ? 'text-primary-600 dark:text-primary-400' : 'text-gray-400 dark:text-gray-600'}`}
            >
              {notifications.marketingEmails ? <FiToggleRight size={28} /> : <FiToggleLeft size={28} />}
            </button>
          </div>
        </div>
      </div>
      
      {/* Account Settings */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-6">Account Settings</h2>
        
        <div className="space-y-4">
          <button className="w-full py-2 text-left px-4 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors">
            Change Password
          </button>
          <button className="w-full py-2 text-left px-4 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors">
            Privacy Settings
          </button>
          <button className="w-full py-2 text-left px-4 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors text-red-600 dark:text-red-400">
            Delete Account
          </button>
        </div>
      </div>
    </div>
  )
}

export default Settings 