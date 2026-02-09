from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import logging
import uuid
import shutil
import speech_recognition as sr
import time
import json

from .models import AudioFile, Transcription
from .serializers import (
    AudioFileSerializer, 
    TranscriptionSerializer,
    AudioFileWithTranscriptionSerializer
)
from .whisper_utils import transcribe_audio
from .file_utils import save_uploaded_file, cleanup_temp_file, normalize_path

# Configure logging
logger = logging.getLogger(__name__)

def voice_recognition_ui(request):
    """View to render the voice recognition UI"""
    return render(request, 'voice_recognition/voice_ui.html')

class AudioFileViewSet(viewsets.ModelViewSet):
    """ViewSet for managing audio files and transcriptions"""
    queryset = AudioFile.objects.all().order_by('-uploaded_at')
    serializer_class = AudioFileWithTranscriptionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter queryset to show only the current user's audio files"""
        return AudioFile.objects.filter(user=self.request.user).order_by('-uploaded_at')
    
    def get_serializer_class(self):
        """Return appropriate serializer class based on action"""
        if self.action == 'create':
            return AudioFileSerializer
        return AudioFileWithTranscriptionSerializer
    
    def get_permissions(self):
        """
        Override to allow unauthenticated access to the transcribe_audio_data endpoint
        """
        if self.action == 'transcribe_audio_data':
            return []  # No permissions required
        return super().get_permissions()
    
    def perform_create(self, serializer):
        """Save the user with the audio file when creating"""
        audio_file = serializer.save(user=self.request.user)
        
        # Get the file path
        file_path = audio_file.file.path
        
        try:
            # Transcribe the audio
            transcription_result = transcribe_audio(file_path)
            
            # Create a transcription record
            Transcription.objects.create(
                audio_file=audio_file,
                text=transcription_result['text'],
                language=transcription_result.get('language', ''),
                confidence=0.0  # Placeholder since we're using a mock implementation
            )
        except Exception as e:
            # Log the error but don't fail the upload
            logger.error(f"Error during transcription: {str(e)}")
    
    @action(detail=True, methods=['post'])
    def transcribe(self, request, pk=None):
        """
        Endpoint to re-transcribe an existing audio file.
        Useful if the initial transcription failed or if you want to try a different model.
        """
        audio_file = self.get_object()
        
        try:
            # Get the file path
            file_path = audio_file.file.path
            
            # Transcribe the audio
            transcription_result = transcribe_audio(file_path)
            
            # Update or create transcription
            transcription, created = Transcription.objects.update_or_create(
                audio_file=audio_file,
                defaults={
                    'text': transcription_result['text'],
                    'language': transcription_result.get('language', ''),
                    'confidence': 0.0
                }
            )
            
            serializer = TranscriptionSerializer(transcription)
            return Response(serializer.data)
        except Exception as e:
            logger.exception(f"Transcription failed: {str(e)}")
            return Response(
                {'error': f'Transcription failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['post'])
    def transcribe_audio_data(self, request):
        """
        Endpoint to transcribe audio data directly without saving it permanently.
        The audio data is sent in the request body.
        """
        if 'audio' not in request.FILES:
            return Response(
                {'error': 'No audio file provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        audio_file = request.FILES['audio']
        logger.info(f"Received audio file '{audio_file.name}' of type {audio_file.content_type} and size {audio_file.size} bytes")
        
        # Save the file using our improved file utility
        temp_file_path = save_uploaded_file(audio_file)
        
        if not temp_file_path:
            return Response(
                {'error': 'Failed to save audio file. Please try recording again.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        try:
            # Verify the file can be accessed
            logger.info(f"Checking if file exists at: {temp_file_path}")
            if not os.path.exists(temp_file_path):
                logger.error(f"File does not exist at path: {temp_file_path}")
                return Response(
                    {'error': 'The audio file could not be found after saving. Please try recording again.'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
            # Log file details
            file_size = os.path.getsize(temp_file_path)
            logger.info(f"File exists at {temp_file_path} with size: {file_size} bytes")
            
            # Transcribe the audio - use normalized path for extra reliability
            norm_path = normalize_path(temp_file_path)
            logger.info(f"Using normalized path for transcription: {norm_path}")
            transcription_result = transcribe_audio(norm_path)
            
            logger.info(f"Transcription complete: {transcription_result}")
            
            # Return the result
            return Response({
                'text': transcription_result['text'],
                'language': transcription_result.get('language', '')
            })
        except Exception as e:
            logger.exception(f"Error processing audio file: {str(e)}")
            return Response(
                {'error': f'Error processing audio: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        finally:
            # Clean up the temporary file
            cleanup_temp_file(temp_file_path)


@api_view(['GET'])
def get_available_microphones(request):
    """
    Get a list of available microphones on the server
    """
    try:
        mics = sr.Microphone.list_microphone_names()
        return Response({'microphones': mics})
    except Exception as e:
        logger.exception(f"Error listing microphones: {str(e)}")
        return Response(
            {'error': f'Error listing microphones: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def realtime_voice_recognition(request):
    """
    Start a real-time voice recognition session
    This endpoint simulates real-time voice recognition using the approach from voice.py
    """
    recognizer = sr.Recognizer()
    
    # Configure recognizer
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    
    try:
        # Extract microphone index from request if provided, otherwise use default
        mic_index = None
        if request.data and 'microphone_index' in request.data:
            try:
                mic_index = int(request.data['microphone_index'])
            except (ValueError, TypeError):
                pass
                
        # Use microphone as source
        source = sr.Microphone(device_index=mic_index) if mic_index is not None else sr.Microphone()
        
        with source as audio_source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(audio_source, duration=1)
            
            # Listen for audio
            try:
                audio = recognizer.listen(audio_source, timeout=5, phrase_time_limit=10)
                logger.info("Audio captured, processing...")
                
                # Convert speech to text
                text = recognizer.recognize_google(audio).lower()
                logger.info(f"Recognized text: {text}")
                
                # Process commands (simplified version from voice.py)
                response = {
                    'text': text,
                    'command_recognized': False,
                    'command_response': None
                }
                
                # Handle simple commands
                if "hello" in text:
                    response['command_recognized'] = True
                    response['command_response'] = "Hello there!"
                elif "time" in text:
                    current_time = time.strftime('%H:%M')
                    response['command_recognized'] = True
                    response['command_response'] = f"Current time is {current_time}"
                
                return Response(response)
                
            except sr.WaitTimeoutError:
                return Response({'error': 'No speech detected. Please try again.'}, 
                                status=status.HTTP_400_BAD_REQUEST)
            except sr.UnknownValueError:
                return Response({'error': 'Could not understand audio. Please try again.'},
                                status=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        logger.exception(f"Error in real-time voice recognition: {str(e)}")
        return Response(
            {'error': f'Error in voice recognition: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
