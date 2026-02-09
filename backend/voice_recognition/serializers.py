from rest_framework import serializers
from .models import AudioFile, Transcription

class AudioFileSerializer(serializers.ModelSerializer):
    """Serializer for the AudioFile model"""
    class Meta:
        model = AudioFile
        fields = ['id', 'title', 'file', 'uploaded_at', 'user']
        read_only_fields = ['uploaded_at', 'user']

class TranscriptionSerializer(serializers.ModelSerializer):
    """Serializer for the Transcription model"""
    class Meta:
        model = Transcription
        fields = ['id', 'audio_file', 'text', 'language', 'confidence', 'created_at']
        read_only_fields = ['created_at']

class AudioFileWithTranscriptionSerializer(serializers.ModelSerializer):
    """Serializer that includes the transcription with the audio file"""
    transcription = TranscriptionSerializer(read_only=True)
    
    class Meta:
        model = AudioFile
        fields = ['id', 'title', 'file', 'uploaded_at', 'user', 'transcription']
        read_only_fields = ['uploaded_at', 'user', 'transcription'] 