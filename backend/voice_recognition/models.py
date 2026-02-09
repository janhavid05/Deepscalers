from django.db import models
from django.conf import settings

class AudioFile(models.Model):
    """Model to store audio files for transcription"""
    title = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to='audio_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE,
        related_name="audio_files",
        null=True, 
        blank=True
    )
    
    def __str__(self):
        return self.title or f"Audio {self.id}"

class Transcription(models.Model):
    """Model to store transcriptions of audio files"""
    audio_file = models.OneToOneField(
        AudioFile,
        on_delete=models.CASCADE,
        related_name="transcription"
    )
    text = models.TextField()
    language = models.CharField(max_length=10, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Transcription for {self.audio_file}"
