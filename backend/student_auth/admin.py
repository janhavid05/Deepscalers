from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.translation import gettext_lazy as _

from .models import StudentUser, PhoneVerification

class StudentUserAdmin(BaseUserAdmin):
    """Admin configuration for the custom user model."""
    ordering = ['id']
    list_display = ['phone_number', 'first_name', 'last_name', 'email', 'is_staff', 'is_verified']
    fieldsets = (
        (None, {'fields': ('phone_number', 'password')}),
        (_('Personal Info'), {'fields': ('first_name', 'last_name', 'email')}),
        (
            _('Permissions'),
            {'fields': ('is_active', 'is_staff', 'is_superuser', 'is_verified')}
        ),
        (_('Important dates'), {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('phone_number', 'password1', 'password2')
        }),
    )

class PhoneVerificationAdmin(admin.ModelAdmin):
    """Admin configuration for the phone verification model."""
    list_display = ['phone_number', 'verification_code', 'created_at', 'expires_at', 'is_used']
    list_filter = ['is_used']
    search_fields = ['phone_number']

admin.site.register(StudentUser, StudentUserAdmin)
admin.site.register(PhoneVerification, PhoneVerificationAdmin)
