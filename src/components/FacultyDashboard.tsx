import React, { useState } from 'react';
import { Alert, Button, Container, Form } from 'react-bootstrap';
import { v4 as uuidv4 } from 'uuid';
import { supabase } from '../lib/supabase';

const FacultyDashboard: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'danger' } | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setMessage({ text: 'Please select a file', type: 'danger' });
      return;
    }

    try {
      setLoading(true);
      setMessage(null);

      // Generate a unique file name
      const fileExt = file.name.split('.').pop();
      const fileName = `${uuidv4()}.${fileExt}`;
      const filePath = `syllabus/${fileName}`;

      // Upload file to Supabase Storage
      const { error: uploadError } = await supabase.storage
        .from('syllabus')
        .upload(filePath, file);

      if (uploadError) {
        throw uploadError;
      }

      // Get the public URL
      const { data: { publicUrl } } = supabase.storage
        .from('syllabus')
        .getPublicUrl(filePath);

      // Store the file information in the database
      const { error: dbError } = await supabase
        .from('syllabus_files')
        .insert([
          {
            file_name: file.name,
            file_path: filePath,
            file_url: publicUrl,
            file_type: file.type,
            file_size: file.size,
            uploaded_at: new Date().toISOString(),
          },
        ]);

      if (dbError) {
        throw dbError;
      }

      setMessage({ text: 'File uploaded successfully!', type: 'success' });
      setFile(null);
    } catch (error) {
      console.error('Error uploading file:', error);
      setMessage({ text: 'Error uploading file. Please try again.', type: 'danger' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="py-5">
      <h2 className="mb-4">Upload Syllabus</h2>
      <Form onSubmit={handleUpload}>
        <Form.Group className="mb-3">
          <Form.Label>Select File (PDF or Image)</Form.Label>
          <Form.Control
            type="file"
            accept=".pdf,image/*"
            onChange={handleFileChange}
            disabled={loading}
          />
        </Form.Group>

        {message && (
          <Alert variant={message.type} className="mb-3">
            {message.text}
          </Alert>
        )}

        <Button
          variant="primary"
          type="submit"
          disabled={loading || !file}
        >
          {loading ? 'Uploading...' : 'Upload File'}
        </Button>
      </Form>
    </Container>
  );
};

export default FacultyDashboard; 