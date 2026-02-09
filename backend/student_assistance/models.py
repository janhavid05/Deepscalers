from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone

User = get_user_model()

class Question(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending Faculty Review'),
        ('answered', 'Answered'),
        ('rejected', 'Rejected')
    ]

    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name='asked_questions')
    text = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    vector_embedding = models.JSONField(null=True, blank=True)  # Store the question's vector embedding

    def __str__(self):
        return f"Question from {self.student.username}: {self.text[:50]}..."

class Answer(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='answers')
    faculty = models.ForeignKey(User, on_delete=models.CASCADE, related_name='provided_answers')
    text = models.TextField()
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    vector_embedding = models.JSONField(null=True, blank=True)  # Store the answer's vector embedding

    def __str__(self):
        return f"Answer to {self.question.text[:50]}..."

class KnowledgeBase(models.Model):
    question = models.TextField()
    answer = models.TextField()
    vector_embedding = models.JSONField()  # Store the combined question-answer vector embedding
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    confidence_score = models.FloatField(default=1.0)  # Track how confident we are in this answer

    def __str__(self):
        return f"KB Entry: {self.question[:50]}..."
