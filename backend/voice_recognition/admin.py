from django.contrib import admin
from .models import AudioFile, Transcription

@admin.register(AudioFile)
class AudioFileAdmin(admin.ModelAdmin):
    list_display = ('title', 'uploaded_at', 'user')
    search_fields = ('title', 'user__email')
    list_filter = ('uploaded_at',)

@admin.register(Transcription)
class TranscriptionAdmin(admin.ModelAdmin):
    list_display = ('audio_file', 'language', 'created_at')
    search_fields = ('text', 'audio_file__title')
    list_filter = ('language', 'created_at')
