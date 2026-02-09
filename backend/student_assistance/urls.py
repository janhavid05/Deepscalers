from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'questions', views.QuestionViewSet, basename='question')
router.register(r'answers', views.AnswerViewSet, basename='answer')
router.register(r'knowledge-base', views.KnowledgeBaseViewSet, basename='knowledge-base')

urlpatterns = [
    path('', include(router.urls)),
    path('pdf/extract-text/', views.PDFTextView.as_view(), name='pdf-extract-text'),
] 