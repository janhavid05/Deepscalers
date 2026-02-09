import random
import string
from datetime import datetime, timedelta
from django.conf import settings
from twilio.rest import Client
from .models import PhoneVerification

def generate_verification_code(length=6):
    """Generate a random numeric verification code."""
    return ''.join(random.choices(string.digits, k=length))

def send_verification_code(phone_number, code):
    """Send verification code using Twilio Verify API."""
    try:
        # Initialize Twilio client
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        
        # Check if we have a service SID
        if hasattr(settings, 'TWILIO_SERVICE_SID') and settings.TWILIO_SERVICE_SID:
            # Use Verify API
            verification = client.verify \
                .services(settings.TWILIO_SERVICE_SID) \
                .verifications \
                .create(to=phone_number, channel='sms')
            
            return True, verification.sid
        else:
            # Fallback to direct SMS
            message = client.messages.create(
                body=f"Your Student Assistant verification code is: {code}",
                from_=settings.TWILIO_PHONE_NUMBER,
                to=phone_number
            )
            
            return True, message.sid
    except Exception as e:
        # Log the error in production
        return False, str(e)

def verify_verification_code(phone_number, code):
    """Verify a code using Twilio Verify API."""
    try:
        # Initialize Twilio client
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        
        # Check if we have a service SID
        if hasattr(settings, 'TWILIO_SERVICE_SID') and settings.TWILIO_SERVICE_SID:
            # Use Verify API
            verification_check = client.verify \
                .services(settings.TWILIO_SERVICE_SID) \
                .verification_checks \
                .create(to=phone_number, code=code)
            
            return verification_check.status == 'approved', verification_check.sid
        else:
            # Fallback to database verification
            return verify_code(phone_number, code)
    except Exception as e:
        # Log the error in production
        return False, str(e)

def create_verification_entry(phone_number):
    """Create a new verification entry for a phone number."""
    # Generate a new 6-digit code
    code = generate_verification_code()
    
    # Set expiration time (10 minutes from now)
    expires_at = datetime.now() + timedelta(minutes=10)
    
    # Save the verification in the database
    verification = PhoneVerification.objects.create(
        phone_number=phone_number,
        verification_code=code,
        expires_at=expires_at
    )
    
    # Send the verification code
    success, message = send_verification_code(phone_number, code)
    
    return success, verification, message

def verify_code(phone_number, code):
    """Verify a code for a given phone number."""
    try:
        # Get the latest verification for this phone number
        verification = PhoneVerification.objects.filter(
            phone_number=phone_number,
            verification_code=code,
            is_used=False,
            expires_at__gt=datetime.now()
        ).latest('created_at')
        
        # Mark the verification as used
        verification.is_used = True
        verification.save()
        
        return True, verification
    except PhoneVerification.DoesNotExist:
        return False, None
    except Exception as e:
        # Log the error in production
        return False, str(e) 