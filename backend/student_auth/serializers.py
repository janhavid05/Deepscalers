from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import PhoneVerification

User = get_user_model()

class PhoneNumberSerializer(serializers.Serializer):
    """Serializer for phone number verification request."""
    phone_number = serializers.CharField(max_length=15)

class VerifyCodeSerializer(serializers.Serializer):
    """Serializer for code verification."""
    phone_number = serializers.CharField(max_length=15)
    code = serializers.CharField(max_length=6)

class UserSerializer(serializers.ModelSerializer):
    """Serializer for the user object."""
    class Meta:
        model = User
        fields = ('id', 'phone_number', 'first_name', 'last_name', 'email', 'is_verified')
        read_only_fields = ('id', 'is_verified')

class UserCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a user."""
    class Meta:
        model = User
        fields = ('phone_number', 'first_name', 'last_name', 'email')
        
    def create(self, validated_data):
        user = User(**validated_data)
        user.is_verified = True  # Since they've verified their phone
        user.save()
        return user 