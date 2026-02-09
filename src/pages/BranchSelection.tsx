import React, { useState } from 'react';
import { FiActivity, FiCode, FiCpu, FiSettings, FiTool, FiWifi, FiZap } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';

const departments = [
  {
    name: 'Computer Science & Engineering',
    icon: <FiCpu className="text-blue-500" size={24} />,
    subjects: [
      'Data Structures', 'Algorithms', 'DBMS', 'Operating Systems', 'Computer Networks', 'AI', 'Machine Learning', 'Software Engineering', 'Web Development', 'Compiler Design',
    ],
  },
  {
    name: 'Information Technology',
    icon: <FiCode className="text-purple-500" size={24} />,
    subjects: [
      'Programming', 'Web Technologies', 'Database Systems', 'Cloud Computing', 'Cyber Security', 'Mobile Computing', 'Software Testing', 'E-Commerce', 'Data Mining', 'Distributed Systems',
    ],
  },
  {
    name: 'Electronics & Communication Engineering',
    icon: <FiWifi className="text-green-500" size={24} />,
    subjects: [
      'Digital Electronics', 'Analog Circuits', 'DSP', 'Microprocessors', 'VLSI', 'Communication Systems', 'Control Systems', 'Embedded Systems', 'Microwave Engineering', 'Antenna Theory',
    ],
  },
  {
    name: 'Mechanical Engineering',
    icon: <FiTool className="text-yellow-500" size={24} />,
    subjects: [
      'Thermodynamics', 'Fluid Mechanics', 'Heat Transfer', 'Machine Design', 'Manufacturing', 'Dynamics', 'Mechanics of Materials', 'Robotics', 'Automobile Engineering', 'CAD/CAM',
    ],
  },
  {
    name: 'Civil Engineering',
    icon: <FiSettings className="text-orange-500" size={24} />,
    subjects: [
      'Structural Engineering', 'Geotechnical Engineering', 'Transportation Engineering', 'Surveying', 'Construction Materials', 'Water Resources', 'Environmental Engineering', 'Concrete Technology', 'Hydraulics', 'Building Planning',
    ],
  },
  {
    name: 'Electrical Engineering',
    icon: <FiZap className="text-purple-500" size={24} />,
    subjects: [
      'Electrical Machines', 'Power Systems', 'Control Systems', 'Power Electronics', 'Measurement', 'Microcontrollers', 'Switchgear', 'High Voltage Engineering', 'Renewable Energy', 'Electrical Drives',
    ],
  },
  {
    name: 'Chemical Engineering',
    icon: <FiSettings className="text-pink-500" size={24} />,
    subjects: [
      'Chemical Process', 'Thermodynamics', 'Fluid Mechanics', 'Heat Transfer', 'Mass Transfer', 'Process Control', 'Polymer Technology', 'Biochemical Engineering', 'Petroleum Refining', 'Environmental Engineering',
    ],
  },
  {
    name: 'Aeronautical Engineering',
    icon: <FiActivity className="text-blue-400" size={24} />,
    subjects: [
      'Aerodynamics', 'Propulsion', 'Flight Mechanics', 'Aircraft Structures', 'Avionics', 'Materials Science', 'Aircraft Design', 'Space Technology', 'Control Engineering', 'Jet Engines',
    ],
  },
  {
    name: 'Biomedical Engineering',
    icon: <FiActivity className="text-green-400" size={24} />,
    subjects: [
      'Medical Imaging', 'Biomechanics', 'Biomaterials', 'Bioinstrumentation', 'Rehabilitation Engineering', 'Clinical Engineering', 'Tissue Engineering', 'Medical Devices', 'Physiology', 'Biosignal Processing',
    ],
  },
  {
    name: 'Automobile Engineering',
    icon: <FiTool className="text-red-400" size={24} />,
    subjects: [
      'Vehicle Dynamics', 'Automotive Engines', 'Chassis Design', 'Transmission Systems', 'Automotive Electronics', 'Fuel Technology', 'Hybrid Vehicles', 'Automotive Materials', 'CAD for Automobiles', 'Emission Control',
    ],
  },
  // Add more branches as needed
];

const BranchSelection: React.FC = () => {
  const [selected, setSelected] = useState<number | null>(null);
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-green-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
      <div className="w-full max-w-3xl p-8 bg-white dark:bg-gray-900 rounded-2xl shadow-lg mt-16">
        <h2 className="text-3xl font-bold text-green-700 dark:text-green-400 mb-2 flex items-center">
          <span className="mr-2"> <FiActivity size={28} /> </span> Faculty Portal
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">Select your department to access the faculty dashboard</p>
        <h3 className="text-lg font-semibold mb-4">Select Your Department</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {departments.map((dept, idx) => (
            <div
              key={dept.name}
              className={`border rounded-xl p-5 bg-white dark:bg-gray-800 shadow hover:shadow-lg transition cursor-pointer ${selected === idx ? 'ring-2 ring-green-400' : 'hover:ring-1 hover:ring-green-200'}`}
              onClick={() => setSelected(idx)}
            >
              <div className="flex items-center mb-2">
                <div className="mr-3">{dept.icon}</div>
                <div>
                  <div className="font-semibold text-lg text-gray-800 dark:text-gray-100">{dept.name}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">{dept.subjects.length} subjects</div>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 mt-2">
                {dept.subjects.map((subj) => (
                  <span key={subj} className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-xs rounded-full text-gray-700 dark:text-gray-200">{subj}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
        <button
          className={`w-full py-3 rounded-lg text-white font-semibold transition-colors ${selected !== null ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-300 cursor-not-allowed'}`}
          disabled={selected === null}
          onClick={() => selected !== null && navigate('/faculty-dashboard/dashboard', { state: { branch: departments[selected].name, subjects: departments[selected].subjects } })}
        >
          Continue to Dashboard
        </button>
      </div>
    </div>
  );
};

export default BranchSelection; 