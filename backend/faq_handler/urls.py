from django.urls import path
from .views import FAQHandlerView

urlpatterns = [
    path('', FAQHandlerView.as_view(), name='faq_root'),  # Handle root FAQ endpoint
    path('ask/', FAQHandlerView.as_view(), name='ask_question'),  # Handle ask endpoint
] 