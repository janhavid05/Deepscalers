"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import JsonResponse
from django.urls.resolvers import URLPattern, URLResolver
from django.urls import get_resolver

def root_view(request):
    resolver = get_resolver()
    url_patterns = []
    
    def extract_urls(urlpatterns, base=''):
        for pattern in urlpatterns:
            if isinstance(pattern, URLPattern):
                url_patterns.append(f"{base}{pattern.pattern}")
            elif isinstance(pattern, URLResolver):
                extract_urls(pattern.url_patterns, f"{base}{pattern.pattern}")
    
    extract_urls(resolver.url_patterns)
    
    return JsonResponse({
        'message': 'API is running',
        'available_urls': url_patterns
    })

def health_check(request):
    return JsonResponse({"status": "healthy"})

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/student/', include('student_auth.urls')),
    path('api/voice-recognition/', include('voice_recognition.urls')),
    path('api/student-assistance/', include('student_assistance.urls')),
    path('api/faq/', include('faq_handler.urls')),
    path('api/health/', health_check, name='health_check'),
    path('', root_view, name='root'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
