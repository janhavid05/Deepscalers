"""
Voice Recognition Setup Checker

This script performs diagnostics to ensure the environment is properly set up for
voice recognition functionality.
"""

import os
import sys
import logging
import tempfile
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voice_recognition_diagnostics")

# Get the media root directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
MEDIA_ROOT = os.path.join(BACKEND_DIR, 'media')

def check_directories():
    """Check if necessary directories exist and are writable"""
    logger.info("Checking directory structure and permissions...")
    
    # Create a list of directories to check
    dirs_to_check = [
        MEDIA_ROOT,
        os.path.join(MEDIA_ROOT, 'temp_audio'),
        os.path.join(MEDIA_ROOT, 'audio_files'),
    ]
    
    for dir_path in dirs_to_check:
        # Ensure directory exists
        os.makedirs(dir_path, exist_ok=True)
        
        if os.path.exists(dir_path):
            logger.info(f"✓ Directory exists: {dir_path}")
            
            # Check if writable by creating a temp file
            test_file = os.path.join(dir_path, "write_test.tmp")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                logger.info(f"✓ Directory is writable: {dir_path}")
            except Exception as e:
                logger.error(f"✗ Directory is not writable: {dir_path}")
                logger.error(f"  Error: {str(e)}")
        else:
            logger.error(f"✗ Directory could not be created: {dir_path}")

def create_test_audio():
    """Create a test audio file for diagnostics"""
    logger.info("Creating test audio file...")
    
    # Create a simple sine wave audio file
    try:
        import numpy as np
        
        temp_dir = os.path.join(MEDIA_ROOT, 'temp_audio')
        os.makedirs(temp_dir, exist_ok=True)
        
        test_audio_path = os.path.join(temp_dir, "test_audio.wav")
        
        # Generate a 1-second 440 Hz sine wave
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        # Convert to int16 PCM
        audio = (audio * 32767).astype(np.int16)
        
        # Create a WAV file directly
        import wave
        
        with wave.open(test_audio_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        
        logger.info(f"✓ Test audio file created at: {test_audio_path}")
        return test_audio_path
    except ImportError:
        logger.error("✗ NumPy is not installed. Cannot create test audio file.")
        return None
    except Exception as e:
        logger.error(f"✗ Failed to create test audio: {str(e)}")
        traceback.print_exc()
        return None

def test_audio_processing():
    """Test basic audio processing functions"""
    logger.info("Testing audio processing...")
    
    # Create test audio
    test_audio_path = create_test_audio()
    if not test_audio_path or not os.path.exists(test_audio_path):
        logger.error("✗ Cannot test audio processing because test audio creation failed")
        return
    
    # Try to open the file using wave module
    try:
        import wave
        with wave.open(test_audio_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            frames = wf.getnframes()
            
        logger.info(f"✓ Successfully opened WAV file: {channels} channels, {frame_rate} Hz, {frames} frames")
    except Exception as e:
        logger.error(f"✗ Failed to open WAV file: {str(e)}")
    
    # Try to process with pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(test_audio_path)
        logger.info(f"✓ Successfully loaded audio with pydub: {len(audio)}ms duration, {audio.channels} channels")
        
        # Test audio processing
        normalized = audio.normalize()
        mono = audio.set_channels(1)
        resampled = audio.set_frame_rate(16000)
        logger.info("✓ Successfully performed audio transformations")
        
    except ImportError:
        logger.error("✗ pydub is not installed")
    except Exception as e:
        logger.error(f"✗ Failed to process audio with pydub: {str(e)}")
        traceback.print_exc()
    
    # Clean up
    try:
        os.remove(test_audio_path)
        logger.info(f"✓ Test audio file removed")
    except Exception as e:
        logger.error(f"✗ Failed to remove test file: {str(e)}")

def test_audio_libraries():
    """Test if required audio libraries are functioning properly"""
    logger.info("Testing audio libraries...")
    
    # Test pydub
    try:
        from pydub import AudioSegment
        # Create a simple audio segment
        audio = AudioSegment.silent(duration=1000)
        logger.info("✓ pydub is working properly")
    except ImportError:
        logger.error("✗ pydub is not installed")
    except Exception as e:
        logger.error(f"✗ pydub error: {str(e)}")
    
    # Test SpeechRecognition
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        logger.info("✓ SpeechRecognition module loaded correctly")
    except ImportError:
        logger.error("✗ SpeechRecognition is not installed")
    except Exception as e:
        logger.error(f"✗ SpeechRecognition error: {str(e)}")

def display_system_info():
    """Display system information that might be relevant to voice recognition"""
    logger.info("System Information:")
    
    # Python version
    logger.info(f"Python version: {sys.version}")
    
    # Operating system
    logger.info(f"Operating system: {sys.platform}")
    
    # Temp directory
    logger.info(f"System temp directory: {tempfile.gettempdir()}")
    
    # Media root
    logger.info(f"MEDIA_ROOT: {MEDIA_ROOT}")
    
    # Check if directories exist
    if os.path.exists(MEDIA_ROOT):
        logger.info(f"MEDIA_ROOT exists: {os.path.abspath(MEDIA_ROOT)}")
    else:
        logger.error(f"MEDIA_ROOT does not exist: {MEDIA_ROOT}")
    
    # List installed packages that might be relevant
    relevant_packages = ["pydub", "speech_recognition", "numpy", "wave"]
    logger.info("Checking for relevant packages:")
    for package in relevant_packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            logger.info(f"✓ {package}: {version}")
        except ImportError:
            logger.error(f"✗ {package}: Not installed")

def check_file_access():
    """Test basic file operations in key directories"""
    logger.info("Testing file access in key locations...")
    
    test_locations = [
        MEDIA_ROOT,
        os.path.join(MEDIA_ROOT, 'temp_audio'),
        tempfile.gettempdir(),
    ]
    
    for location in test_locations:
        os.makedirs(location, exist_ok=True)
        test_file = os.path.join(location, f"access_test_{os.urandom(4).hex()}.tmp")
        
        try:
            # Test writing
            with open(test_file, 'wb') as f:
                f.write(b"test data")
            
            # Test reading
            with open(test_file, 'rb') as f:
                data = f.read()
            
            if data == b"test data":
                logger.info(f"✓ Successfully wrote and read file in {location}")
            else:
                logger.error(f"✗ File data mismatch in {location}")
            
            # Test deletion
            os.remove(test_file)
            if not os.path.exists(test_file):
                logger.info(f"✓ Successfully deleted file in {location}")
            else:
                logger.error(f"✗ Failed to delete file in {location}")
                
        except Exception as e:
            logger.error(f"✗ File operation failed in {location}: {str(e)}")

def main():
    """Run all diagnostic checks"""
    logger.info("=" * 50)
    logger.info("Voice Recognition Diagnostics")
    logger.info("=" * 50)
    
    display_system_info()
    logger.info("-" * 50)
    
    check_directories()
    logger.info("-" * 50)
    
    check_file_access()
    logger.info("-" * 50)
    
    test_audio_libraries()
    logger.info("-" * 50)
    
    test_audio_processing()
    logger.info("-" * 50)
    
    logger.info("Diagnostics complete")

if __name__ == "__main__":
    main() 