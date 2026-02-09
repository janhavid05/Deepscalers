from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AudioFileViewSet, get_available_microphones, realtime_voice_recognition, voice_recognition_ui

router = DefaultRouter()
router.register(r'audio-files', AudioFileViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('microphones/', get_available_microphones, name='get-microphones'),
    path('recognize/', realtime_voice_recognition, name='realtime-voice-recognition'),
    path('ui/', voice_recognition_ui, name='voice-recognition-ui'),
] 