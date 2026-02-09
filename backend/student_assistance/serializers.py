from rest_framework import serializers
from .models import Question, Answer, KnowledgeBase

class QuestionSerializer(serializers.ModelSerializer):
    student_username = serializers.CharField(source='student.username', read_only=True)
    
    class Meta:
        model = Question
        fields = ['id', 'student', 'student_username', 'text', 'status', 'created_at', 'updated_at']
        read_only_fields = ['student', 'status', 'created_at', 'updated_at']

class AnswerSerializer(serializers.ModelSerializer):
    faculty_username = serializers.CharField(source='faculty.username', read_only=True)
    
    class Meta:
        model = Answer
        fields = ['id', 'question', 'faculty', 'faculty_username', 'text', 'is_verified', 'created_at', 'updated_at']
        read_only_fields = ['faculty', 'created_at', 'updated_at']

class KnowledgeBaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeBase
        fields = ['id', 'question', 'answer', 'confidence_score', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at'] 