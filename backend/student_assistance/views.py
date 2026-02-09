from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.shortcuts import get_object_or_404
from .models import Question, Answer, KnowledgeBase
from .serializers import QuestionSerializer, AnswerSerializer, KnowledgeBaseSerializer
from .ai_service import AIService
from django.contrib.auth import get_user_model
import PyPDF2
from rest_framework.views import APIView
import logging
import os
import tempfile
import uuid
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

User = get_user_model()
ai_service = AIService()
logger = logging.getLogger(__name__)

# -------------------- SAFE CLIENT SETUP --------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment variables")

client = Groq(api_key=GROQ_API_KEY)

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- AI PROCESSING --------------------

def process_text_chunk(text, chunk_number, total_chunks):
    try:
        chunk_context = (
            f"This is chunk {chunk_number} of {total_chunks}. "
            "Generate self-contained, detailed Q&A pairs."
        )

        prompt = f"""
You are an expert at generating educational Q&A pairs.

{chunk_context}

{text}

Format:
## 1. Question:
## Answer:
"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=1,
            max_tokens=8000,
            stream=True
        )

        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content

        faq_data = []
        pattern = r"##\s*\d+\.\s*Question:\s*(.*?)\n##\s*Answer:\s*(.*?)(?=\n##|\Z)"
        matches = re.findall(pattern, full_response, re.DOTALL)

        for q, a in matches:
            faq_data.append({
                "question": q.strip(),
                "answer": a.strip()
            })

        return faq_data

    except Exception as e:
        logger.error(f"Groq error: {str(e)}")
        return []

def generate_qa_pairs(text):
    estimated_tokens = len(text) // 4
    logger.info(f"Estimated tokens: {estimated_tokens}")

    if estimated_tokens <= 15000:
        return process_text_chunk(text, 1, 1)

    # Chunking for large content
    num_chunks = 4
    chunk_size = len(text) // num_chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    all_qa = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_text_chunk, chunk, i+1, len(chunks))
            for i, chunk in enumerate(chunks)
        ]
        for future in as_completed(futures):
            all_qa.extend(future.result())

    return all_qa

# -------------------- QDRANT --------------------

def upload_to_qdrant(faq_data):
    try:
        collection_name = "student_faqs"

        if not qdrant_client.collection_exists(collection_name):
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

        points = []
        for item in faq_data:
            vector = embedding_model.encode(item["question"]).tolist()
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=item
                )
            )

        qdrant_client.upsert(collection_name, points)
        return True

    except Exception as e:
        logger.error(f"Qdrant error: {str(e)}")
        return False
