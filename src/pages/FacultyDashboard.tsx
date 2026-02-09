import { useState } from 'react'
import { FiBookOpen, FiCheck, FiEdit, FiX } from 'react-icons/fi'
import { useLocation } from 'react-router-dom'
import { v4 as uuidv4 } from 'uuid'
import { supabase } from '../lib/supabase'
import PDFViewer from '../components/PDFViewer'

// Types
type QuestionStatus = 'pending' | 'approved' | 'rejected'

interface StudentQuestion {
  id: string
  studentId: string
  studentName: string
  question: string
  aiAnswer: string
  facultyAnswer?: string
  status: QuestionStatus
  date: Date
  subject: string
}

// Mock data
const initialQuestions: StudentQuestion[] = [
  {
    id: '1',
    studentId: 'STU001',
    studentName: 'Alex Johnson',
    question: 'Can you explain the process of photosynthesis in detail?',
    aiAnswer: 'Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigments. It involves converting light energy into chemical energy and storing it in the chemical bonds of sugar.',
    status: 'pending',
    date: new Date(2023, 7, 20),
    subject: 'Biology',
  },
  {
    id: '2',
    studentId: 'STU002',
    studentName: 'Sam Wilson',
    question: 'How do I calculate the derivative of a function?',
    aiAnswer: 'To calculate the derivative of a function, you use the limit definition: f\'(x) = lim(h→0) [f(x+h) - f(x)]/h. There are also various rules like the power rule, product rule, quotient rule, and chain rule that make differentiation easier.',
    facultyAnswer: 'The derivative represents the rate of change of a function. For a function f(x), the derivative f\'(x) can be found using various rules depending on the function type. For polynomials, use the power rule: d/dx(x^n) = n*x^(n-1).',
    status: 'approved',
    date: new Date(2023, 7, 18),
    subject: 'Mathematics',
  },
  {
    id: '3',
    studentId: 'STU003',
    studentName: 'Emily Davis',
    question: 'What is the difference between RAM and ROM?',
    aiAnswer: 'RAM (Random Access Memory) is volatile and temporary, storing data that is actively being used. ROM (Read-Only Memory) is non-volatile and permanent, containing instructions for the computer that do not change.',
    status: 'pending',
    date: new Date(2023, 7, 22),
    subject: 'Computer Science',
  },
  {
    id: '4',
    studentId: 'STU004',
    studentName: 'Rachel Green',
    question: 'Can you explain Newton\'s laws of motion?',
    aiAnswer: 'Newton\'s First Law: An object at rest stays at rest, and an object in motion stays in motion unless acted upon by an external force. Newton\'s Second Law: Force equals mass times acceleration (F = ma). Newton\'s Third Law: For every action, there is an equal and opposite reaction.',
    facultyAnswer: 'Newton\'s laws form the foundation of classical mechanics. The first law describes inertia, the second quantifies the relationship between force, mass, and acceleration, and the third law describes the reciprocal nature of forces.',
    status: 'rejected',
    date: new Date(2023, 7, 15),
    subject: 'Physics',
  },
]

// Add mock data for most common questions
const mostCommonQuestions = [
  {
    question: 'What is the exam pattern for this subject?',
    answer: 'The exam consists of two sections: theory and practical.'
  },
  {
    question: 'How can I improve my grades?',
    answer: 'Attend all lectures, complete assignments on time, and practice previous year papers.'
  },
  {
    question: 'Are there any recommended books?',
    answer: 'Yes, refer to the syllabus for a list of recommended books.'
  },
];

// Add mock subjects for the upload syllabus dropdown
const syllabusSubjects = [
  '-- Select Subject --',
  'Mathematics',
  'Physics',
  'Computer Science',
  'Biology',
  'Chemistry',
];

const tabList = [
  'Student Questions',
  'Most Common Questions',
  'Upload Syllabus',
];

const statusColors = {
  pending: 'text-yellow-600',
  approved: 'text-green-600',
  rejected: 'text-red-600',
};

const FacultyDashboard = () => {
  const [questions, setQuestions] = useState<StudentQuestion[]>(initialQuestions)
  const [filter, setFilter] = useState<QuestionStatus | 'all'>('all')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editAnswer, setEditAnswer] = useState('')
  const [activeTab, setActiveTab] = useState(0)
  const location = useLocation();
  const branchSubjects: string[] = location.state?.subjects && Array.isArray(location.state.subjects)
    ? location.state.subjects.slice(0, 10)
    : syllabusSubjects;
  const [selectedSubject, setSelectedSubject] = useState(branchSubjects[0]);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);
  
  // Filter questions based on status
  const filteredQuestions = filter === 'all' 
    ? questions 
    : questions.filter(q => q.status === filter)
  
  // Start editing an answer
  const handleEdit = (question: StudentQuestion) => {
    setEditingId(question.id)
    setEditAnswer(question.facultyAnswer || question.aiAnswer)
  }
  
  // Save faculty answer
  const handleSave = (id: string) => {
    setQuestions(questions.map(q => 
      q.id === id ? { ...q, facultyAnswer: editAnswer, status: 'approved' } : q
    ))
    setEditingId(null)
  }
  
  // Approve AI answer
  const handleApprove = (id: string) => {
    setQuestions(questions.map(q => 
      q.id === id ? { ...q, status: 'approved' } : q
    ))
  }
  
  // Reject answer
  const handleReject = (id: string) => {
    setQuestions(questions.map(q => 
      q.id === id ? { ...q, status: 'rejected' } : q
    ))
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file || !selectedSubject) {
      setUploadMessage({ text: 'Please select both a subject and a file', type: 'error' });
      return;
    }

    try {
      setUploading(true);
      setUploadMessage(null);

      const fileExt = file.name.split('.').pop();
      const fileName = `${selectedSubject}_${uuidv4()}.${fileExt}`;

      // Upload file to Supabase Storage
      const { error: uploadError } = await supabase.storage
        .from('data')
        .upload(fileName, file);

      if (uploadError) {
        console.error('Upload error:', uploadError);
        throw uploadError;
      }

      setUploadMessage({ text: 'Syllabus uploaded successfully!', type: 'success' });
      setFile(null);
    } catch (error: any) {
      console.error('Error uploading file:', error);
      setUploadMessage({ 
        text: `Error uploading syllabus: ${error?.message || 'Unknown error'}`, 
        type: 'error' 
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Faculty Dashboard</h1>
      
      {/* Add PDF Viewer Section */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">PDF Text Extraction</h2>
        <PDFViewer />
      </div>

      <div className="flex flex-col items-center min-h-screen bg-gradient-to-br from-blue-50 to-green-50 dark:from-gray-900 dark:to-gray-800 py-12">
        <div className="w-full max-w-5xl mx-auto bg-white rounded-3xl shadow-2xl p-10 flex flex-col items-center">
          {/* Tab Bar */}
          <div className="flex space-x-3 mb-6">
            {tabList.map((tab, idx) => (
              <button
                key={tab}
                className={`px-5 py-2 rounded-full font-semibold text-sm transition-colors duration-200 border-none focus:outline-none ${
                  activeTab === idx
                    ? idx === 1
                      ? 'bg-green-500 text-white shadow'
                      : 'bg-blue-600 text-white shadow'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-blue-100 dark:hover:bg-gray-700'
                }`}
                onClick={() => setActiveTab(idx)}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          {activeTab === 0 && (
            <>
              {/* Dashboard Title */}
              <div className="flex items-center mb-6 w-full">
                <div className="w-1.5 h-8 bg-blue-600 rounded-r-lg mr-3" />
                <FiBookOpen className="text-blue-600 mr-2" size={28} />
                <h1 className="text-3xl font-bold text-blue-700 dark:text-blue-400">Faculty Dashboard</h1>
              </div>

              {/* Filter Buttons */}
              <div className="flex space-x-2 mb-6 w-full">
                {['all', 'pending', 'approved', 'rejected'].map((status) => (
                  <button
                    key={status}
                    onClick={() => setFilter(status as any)}
                    className={`px-4 py-1 rounded-full font-medium text-sm transition-colors duration-200 border-none focus:outline-none ${
                      filter === status
                        ? 'bg-blue-600 text-white shadow'
                        : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-blue-100 dark:hover:bg-gray-700'
                    }`}
                  >
                    {status.charAt(0).toUpperCase() + status.slice(1)}
                  </button>
                ))}
              </div>

              {/* Questions Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
                {filteredQuestions.map((question) => (
                  <div
                    key={question.id}
                    className="bg-[#f8fafc] dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm p-6 flex flex-col"
                  >
                    <div className="flex justify-between items-center mb-2">
                      <div>
                        <span className="font-bold text-lg text-gray-800 dark:text-gray-100">{question.studentName}</span>
                        <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">({question.studentId})</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-500 dark:text-gray-400">{question.date.toLocaleDateString()}</span>
                        <span className="text-xs px-2 py-1 rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200 font-medium ml-2">{question.subject}</span>
                      </div>
                    </div>
                    <div className="mb-2">
                      <span className="font-semibold text-gray-700 dark:text-gray-200">Question:</span>
                      <span className="ml-1 text-gray-700 dark:text-gray-300">{question.question.length > 60 ? question.question.slice(0, 60) + '...' : question.question}</span>
                    </div>
                    <div className="mb-2">
                      <span className="font-semibold text-gray-700 dark:text-gray-200">AI Answer:</span>
                      <span className="ml-1 text-gray-700 dark:text-gray-300">{question.aiAnswer.length > 60 ? question.aiAnswer.slice(0, 60) + '...' : question.aiAnswer}</span>
                    </div>
                    {question.status === 'approved' && question.facultyAnswer && (
                      <div className="mb-2">
                        <span className="font-semibold text-green-700 dark:text-green-400">Faculty Answer:</span>
                        <span className="ml-1 text-gray-700 dark:text-gray-300">{question.facultyAnswer.length > 60 ? question.facultyAnswer.slice(0, 60) + '...' : question.facultyAnswer}</span>
                      </div>
                    )}
                    {editingId === question.id ? (
                      <div className="mb-2">
                        <textarea
                          value={editAnswer}
                          onChange={(e) => setEditAnswer(e.target.value)}
                          className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                          rows={3}
                        />
                        <div className="mt-2 flex justify-end space-x-2">
                          <button
                            onClick={() => setEditingId(null)}
                            className="px-3 py-1 border rounded hover:bg-gray-100 dark:hover:bg-gray-700"
                          >
                            Cancel
                          </button>
                          <button
                            onClick={() => handleSave(question.id)}
                            className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
                          >
                            Save Answer
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex space-x-2 mt-2">
                        {question.status === 'pending' && (
                          <>
                            <button
                              onClick={() => handleApprove(question.id)}
                              className="flex items-center px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
                            >
                              <FiCheck className="mr-1" /> Approve
                            </button>
                            <button
                              onClick={() => handleReject(question.id)}
                              className="flex items-center px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600"
                            >
                              <FiX className="mr-1" /> Reject
                            </button>
                          </>
                        )}
                        <button
                          onClick={() => handleEdit(question)}
                          className="flex items-center px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                        >
                          <FiEdit className="mr-1" /> Edit Answer
                        </button>
                      </div>
                    )}
                    <div className="flex justify-between items-center mt-4">
                      <span className={`text-xs font-semibold px-3 py-1 rounded-full ${statusColors[question.status] || 'bg-gray-100 text-gray-500'}`}>
                        Status: {question.status.charAt(0).toUpperCase() + question.status.slice(1)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {activeTab === 1 && (
            <div className="w-full flex flex-col items-center">
              <div className="w-full bg-gradient-to-br from-green-100/80 via-white to-green-50 rounded-2xl shadow-lg p-10 border-l-8 border-green-400">
                <h2 className="text-2xl font-bold text-green-700 mb-2 flex items-center">
                  <span className="w-3 h-3 rounded-full bg-green-500 mr-3" />
                  Most Common Questions
                </h2>
                <p className="text-green-700 mb-6 text-sm font-medium">Frequently asked by students in this subject.</p>
                <div className="divide-y divide-green-200">
                  {mostCommonQuestions.map((item, idx) => (
                    <div key={idx} className="py-6 group transition-all">
                      <div className="flex items-center mb-1">
                        <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-green-200 text-green-700 mr-2 font-bold">Q</span>
                        <span className="font-semibold text-blue-700 group-hover:text-blue-900 underline cursor-pointer transition-colors">{item.question}</span>
                      </div>
                      <div className="flex items-center mb-1">
                        <span className="w-6 h-6" />
                        <span className="block w-2 h-2 rounded-full bg-green-300 mx-2" />
                        <span className="text-gray-700 font-medium">{item.answer}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
          {activeTab === 2 && (
            <div className="w-full flex flex-col items-center">
              <div className="w-full max-w-md bg-gradient-to-br from-purple-100/80 via-white to-purple-50 rounded-2xl shadow-lg p-10 border-l-8 border-purple-500">
                <h2 className="text-2xl font-bold text-purple-700 mb-2 flex items-center">
                  <span className="w-3 h-3 rounded-full bg-purple-500 mr-3" />
                  Upload Syllabus
                </h2>
                <p className="text-purple-700 mb-6 text-sm font-medium">Upload or update the syllabus for a specific subject below.</p>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Select Subject</label>
                  <div className="relative">
                    <select
                      className="w-full p-3 pr-10 border border-purple-200 rounded-xl bg-white/60 backdrop-blur shadow focus:ring-2 focus:ring-purple-400 focus:border-purple-400 appearance-none"
                      value={selectedSubject}
                      onChange={e => setSelectedSubject(e.target.value)}
                    >
                      {branchSubjects.map((subject: string, idx: number) => (
                        <option key={idx} value={subject}>{subject}</option>
                      ))}
                    </select>
                    <span className="absolute right-3 top-1/2 -translate-y-1/2 text-purple-400 pointer-events-none">
                      <svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path stroke="currentColor" strokeWidth="2" d="M6 9l6 6 6-6"/></svg>
                    </span>
                  </div>
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Syllabus File</label>
                  <input 
                    type="file" 
                    className="block w-full text-sm text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100"
                    onChange={handleFileChange}
                    accept=".pdf,.doc,.docx,.ppt,.pptx,image/*,.mp4,.avi,.mov"
                    disabled={uploading}
                  />
                </div>
                {uploadMessage && (
                  <div className={`mb-4 p-3 rounded-lg ${uploadMessage.type === 'success' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                    {uploadMessage.text}
                  </div>
                )}
                <button
                  className="w-full py-3 rounded-xl bg-purple-500 text-white font-semibold mt-2 shadow-md hover:bg-purple-600 transition disabled:opacity-70 disabled:cursor-not-allowed"
                  onClick={handleUpload}
                  disabled={uploading || !file || selectedSubject === '-- Select Subject --'}
                >
                  <span className="inline-flex items-center justify-center">
                    {uploading ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Uploading...
                      </>
                    ) : (
                      <>
                        <svg className="mr-2" width="20" height="20" fill="none" viewBox="0 0 24 24"><path stroke="currentColor" strokeWidth="2" d="M12 4v16m8-8H4"/></svg>
                        Upload Syllabus
                      </>
                    )}
                  </span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default FacultyDashboard 