import { FiCheckCircle, FiUser } from 'react-icons/fi'

// Types
interface FacultyAnswer {
  id: string
  question: string
  answer: string
  faculty: {
    name: string
    department: string
    image?: string
  }
  date: Date
  subject: string
}

// Mock data
const facultyAnswers: FacultyAnswer[] = [
  {
    id: '1',
    question: 'What is the importance of quantum mechanics in modern physics?',
    answer: 'Quantum mechanics is essential in modern physics as it explains phenomena at the atomic and subatomic level that classical physics cannot account for. It underpins our understanding of many technologies like lasers, transistors, and nuclear energy, and helps explain fundamental aspects of chemistry and material science. Unlike classical physics, it describes nature probabilistically rather than deterministically, which has profound implications for our understanding of reality.',
    faculty: {
      name: 'Dr. Richard Feynman',
      department: 'Physics',
      image: 'https://i.imgur.com/JLMr5tF.jpg'
    },
    date: new Date(2023, 6, 15),
    subject: 'Physics'
  },
  {
    id: '2',
    question: 'How do neural networks learn patterns in data?',
    answer: 'Neural networks learn patterns through a process called backpropagation. Initially, the network makes predictions with random weights. It then compares these predictions to actual values, calculating the error. This error is "propagated back" through the network to adjust weights incrementally. Through many iterations with different data samples, the network gradually improves its predictions by minimizing the error function. This process, called gradient descent, allows neural networks to identify complex patterns that might be difficult to program explicitly.',
    faculty: {
      name: 'Prof. Ada Lovelace',
      department: 'Computer Science',
      image: 'https://i.imgur.com/7nZ2bJe.jpg'
    },
    date: new Date(2023, 6, 20),
    subject: 'Computer Science'
  },
  {
    id: '3',
    question: 'Explain the process of cellular respiration in detail.',
    answer: 'Cellular respiration is a metabolic process where cells convert glucose into energy in the form of ATP. It occurs in three main stages: glycolysis, the Krebs cycle, and electron transport chain. Glycolysis breaks glucose into pyruvate, producing a small amount of ATP. The Krebs cycle then oxidizes pyruvate to CO2, generating electron carriers (NADH and FADH2). Finally, these carriers donate electrons to the electron transport chain, which creates a proton gradient to drive ATP synthesis. This process efficiently produces up to 36 ATP molecules per glucose molecule, powering cellular functions.',
    faculty: {
      name: 'Dr. Elizabeth Blackwell',
      department: 'Biology',
      image: 'https://i.imgur.com/9KYjwXN.jpg'
    },
    date: new Date(2023, 7, 5),
    subject: 'Biology'
  },
]

const FacultyAnswers = () => {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Faculty Verified Answers</h1>
      </div>
      
      <div className="space-y-6">
        {facultyAnswers.map((answer) => (
          <div key={answer.id} className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <div className="flex items-center mb-4">
              <div className="flex items-center justify-center h-10 w-10 rounded-full bg-primary-100 text-primary-600 dark:bg-primary-900 dark:text-primary-300 mr-3">
                {answer.faculty.image ? (
                  <img 
                    src={answer.faculty.image} 
                    alt={answer.faculty.name}
                    className="h-10 w-10 rounded-full object-cover"
                  />
                ) : (
                  <FiUser size={20} />
                )}
              </div>
              <div>
                <div className="font-medium">{answer.faculty.name}</div>
                <div className="text-sm text-gray-500 dark:text-gray-400">{answer.faculty.department}</div>
              </div>
              <div className="ml-auto flex items-center text-green-600 bg-green-100 px-2 py-1 rounded-full text-xs font-medium">
                <FiCheckCircle className="mr-1" />
                Verified by Faculty
              </div>
            </div>
            
            <div className="mb-4">
              <h3 className="text-lg font-medium mb-2">Question:</h3>
              <p className="text-gray-700 dark:text-gray-300">{answer.question}</p>
            </div>
            
            <div className="mb-4">
              <h3 className="text-lg font-medium mb-2">Answer:</h3>
              <p className="text-gray-700 dark:text-gray-300">{answer.answer}</p>
            </div>
            
            <div className="flex justify-between items-center text-sm text-gray-500 dark:text-gray-400 pt-3 border-t dark:border-gray-700">
              <div>{answer.subject}</div>
              <div>
                Answered on {answer.date.toLocaleDateString('en-US', { 
                  year: 'numeric', 
                  month: 'long', 
                  day: 'numeric' 
                })}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default FacultyAnswers 