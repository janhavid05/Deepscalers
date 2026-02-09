from django.urls import path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from .views import (
    SendVerificationAPIView,
    VerifyCodeAPIView,
    RegisterUserAPIView,
    UserProfileAPIView,
)

urlpatterns = [
    # Phone verification
    path('send-verification/', SendVerificationAPIView.as_view(), name='send_verification'),
    path('verify-code/', VerifyCodeAPIView.as_view(), name='verify_code'),
    
    # User registration and profile
    path('register/', RegisterUserAPIView.as_view(), name='register'),
    path('profile/', UserProfileAPIView.as_view(), name='profile'),
    
    # JWT token endpoints
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
] 