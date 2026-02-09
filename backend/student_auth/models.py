from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.utils.translation import gettext_lazy as _

class StudentUserManager(BaseUserManager):
    """Define a model manager for User model with no username field."""

    def _create_user(self, phone_number, **extra_fields):
        """Create and save a User with the given phone number."""
        if not phone_number:
            raise ValueError('The phone number must be set')
        user = self.model(phone_number=phone_number, **extra_fields)
        user.save(using=self._db)
        return user

    def create_user(self, phone_number, **extra_fields):
        """Create and save a regular User with the given phone number."""
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        return self._create_user(phone_number, **extra_fields)

    def create_superuser(self, phone_number, password=None, **extra_fields):
        """Create and save a SuperUser with the given phone number and password."""
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self._create_user(phone_number, **extra_fields)

class StudentUser(AbstractUser):
    """Custom user model that uses phone number instead of username."""

    username = None
    phone_number = models.CharField(_('phone number'), max_length=15, unique=True)
    is_verified = models.BooleanField(default=False)
    
    USERNAME_FIELD = 'phone_number'
    REQUIRED_FIELDS = []

    objects = StudentUserManager()

    def __str__(self):
        return self.phone_number

class PhoneVerification(models.Model):
    """Model to store verification codes for phone number verification."""
    
    phone_number = models.CharField(max_length=15)
    verification_code = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_used = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.phone_number} - {self.verification_code}"
