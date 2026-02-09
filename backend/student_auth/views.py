from django.shortcuts import render
from django.contrib.auth import get_user_model
from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import (
    PhoneNumberSerializer, 
    VerifyCodeSerializer, 
    UserSerializer,
    UserCreateSerializer
)
from .utils import create_verification_entry, verify_verification_code

User = get_user_model()

class SendVerificationAPIView(APIView):
    """API for sending phone verification code."""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = PhoneNumberSerializer(data=request.data)
        
        if serializer.is_valid():
            phone_number = serializer.validated_data['phone_number']
            success, verification, message = create_verification_entry(phone_number)
            
            if success:
                return Response({
                    'success': True,
                    'message': 'Verification code sent successfully'
                }, status=status.HTTP_200_OK)
            
            return Response({
                'success': False,
                'message': f'Failed to send verification code: {message}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class VerifyCodeAPIView(APIView):
    """API for verifying phone verification code."""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = VerifyCodeSerializer(data=request.data)
        
        if serializer.is_valid():
            phone_number = serializer.validated_data['phone_number']
            code = serializer.validated_data['code']
            
            success, verification = verify_verification_code(phone_number, code)
            
            if success:
                # Check if user exists with this phone number
                try:
                    user = User.objects.get(phone_number=phone_number)
                    user.is_verified = True
                    user.save()
                    
                    # Generate JWT tokens
                    refresh = RefreshToken.for_user(user)
                    
                    return Response({
                        'success': True,
                        'message': 'Verification successful',
                        'user_exists': True,
                        'refresh': str(refresh),
                        'access': str(refresh.access_token),
                        'user': UserSerializer(user).data
                    }, status=status.HTTP_200_OK)
                
                except User.DoesNotExist:
                    # User doesn't exist yet, but phone is verified
                    return Response({
                        'success': True,
                        'message': 'Verification successful',
                        'user_exists': False
                    }, status=status.HTTP_200_OK)
            
            return Response({
                'success': False,
                'message': 'Invalid or expired verification code'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class RegisterUserAPIView(APIView):
    """API for registering a new user."""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = UserCreateSerializer(data=request.data)
        
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            
            return Response({
                'success': True,
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'user': UserSerializer(user).data
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UserProfileAPIView(APIView):
    """API for managing user profile."""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)
    
    def put(self, request):
        serializer = UserSerializer(request.user, data=request.data, partial=True)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
