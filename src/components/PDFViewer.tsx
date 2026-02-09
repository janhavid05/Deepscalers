import React, { useState } from 'react';
import axios from 'axios';
import { API_CONFIG } from '../config';

interface PDFViewerProps {
  onTextExtracted?: (text: string) => void;
}

interface QAPair {
  question: string;
  answer: string;
}

const PDFViewer: React.FC<PDFViewerProps> = ({ onTextExtracted }) => {
  const [file, setFile] = useState<File | null>(null);
  const [extractedText, setExtractedText] = useState<string>('');
  const [qaPairs, setQAPairs] = useState<QAPair[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid PDF file');
      setFile(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a PDF file first');
      return;
    }

    setLoading(true);
    setError(null);
    setQAPairs([]);

    const formData = new FormData();
    formData.append('pdf_file', file);

    try {
      console.log('Uploading PDF to:', `${API_CONFIG.BASE_URL}/student-assistance/pdf/extract-text/`);
      const response = await axios.post(
        `${API_CONFIG.BASE_URL}/student-assistance/pdf/extract-text/`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      console.log('PDF extraction response:', response.data);
      setExtractedText(response.data.text);
      if (response.data.qa_pairs) {
        setQAPairs(response.data.qa_pairs);
      }
      if (onTextExtracted) {
        onTextExtracted(response.data.text);
      }
    } catch (err: any) {
      console.error('PDF extraction error:', err);
      if (err.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error('Error response data:', err.response.data);
        console.error('Error response status:', err.response.status);
        console.error('Error response headers:', err.response.headers);
        setError(err.response.data.error || 'Error extracting text from PDF. Please try again.');
      } else if (err.request) {
        // The request was made but no response was received
        console.error('Error request:', err.request);
        setError('No response from server. Please check your connection and try again.');
      } else {
        // Something happened in setting up the request that triggered an Error
        console.error('Error message:', err.message);
        setError('Error setting up the request. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Upload PDF
        </label>
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100"
        />
      </div>

      {error && (
        <div className="mb-4 p-2 bg-red-50 text-red-700 rounded">
          {error}
        </div>
      )}

      <button
        onClick={handleUpload}
        disabled={!file || loading}
        className={`px-4 py-2 rounded-md text-white font-medium
          ${!file || loading
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-700'
          }`}
      >
        {loading ? 'Processing...' : 'Extract Text & Generate Q&A'}
      </button>

      {extractedText && (
        <div className="mt-4">
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Extracted Text
          </h3>
          <div className="p-4 bg-gray-50 rounded-lg max-h-96 overflow-y-auto">
            <pre className="whitespace-pre-wrap text-sm text-gray-700">
              {extractedText}
            </pre>
          </div>
        </div>
      )}

      {qaPairs.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Generated Q&A Pairs
          </h3>
          <div className="space-y-4">
            {qaPairs.map((qa, index) => (
              <div key={index} className="p-4 bg-gray-50 rounded-lg">
                <h4 className="font-medium text-blue-600 mb-2">
                  Q: {qa.question}
                </h4>
                <p className="text-gray-700">
                  A: {qa.answer}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PDFViewer; 