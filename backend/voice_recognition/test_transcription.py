"""
Direct test script for voice recognition functionality.

This script allows testing audio transcription directly without going through the API.
"""

import os
import sys
import logging
import wave
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voice_recognition_test")

# Add the parent directory to the path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Set up Django environment
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

# Now we can import our modules
from voice_recognition.whisper_utils import transcribe_audio
from voice_recognition.file_utils import save_uploaded_file, cleanup_temp_file

def create_test_audio():
    """Create a test audio file with a sine wave tone"""
    logger.info("Creating test audio file...")
    
    try:
        # Create a simple sine wave
        sample_rate = 16000
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz tone
        
        # Convert to int16
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create a temporary file
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Save as WAV
        with wave.open(temp_file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
            
        logger.info(f"Created test audio file at: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        logger.error(f"Failed to create test audio: {e}")
        return None

def test_transcription_with_test_file():
    """Test transcription with a generated test file"""
    test_file_path = create_test_audio()
    if not test_file_path:
        logger.error("Failed to create test audio file")
        return False
        
    try:
        # Transcribe the audio
        logger.info(f"Transcribing file: {test_file_path}")
        result = transcribe_audio(test_file_path)
        logger.info(f"Transcription result: {result}")
        
        # Cleanup
        cleanup_temp_file(test_file_path)
        return True
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        cleanup_temp_file(test_file_path)
        return False

def test_transcription_with_file(file_path):
    """Test transcription with a user-provided file"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
        
    try:
        # Get absolute path
        abs_path = os.path.abspath(file_path)
        logger.info(f"Transcribing file: {abs_path}")
        
        # Transcribe the audio
        result = transcribe_audio(abs_path)
        logger.info(f"Transcription result: {result}")
        return True
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return False

def main():
    """Run the test script"""
    logger.info("=" * 50)
    logger.info("Voice Recognition Test")
    logger.info("=" * 50)
    
    # First test with a generated test file
    logger.info("Testing with generated test audio...")
    if test_transcription_with_test_file():
        logger.info("✓ Test with generated audio successful")
    else:
        logger.error("✗ Test with generated audio failed")
    
    # Check if a file path was provided as an argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        logger.info(f"Testing with provided file: {file_path}")
        if test_transcription_with_file(file_path):
            logger.info("✓ Test with provided file successful")
        else:
            logger.error("✗ Test with provided file failed")
    
    logger.info("=" * 50)
    logger.info("Test complete")
    logger.info("=" * 50)

if __name__ == "__main__":
    main() 