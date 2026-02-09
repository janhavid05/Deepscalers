from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from typing import List, Tuple, Optional
import os
from dotenv import load_dotenv
from django.conf import settings
from groq import Groq

load_dotenv()

# Configure Groq
client = Groq(api_key=os.getenv('GROQ_API_KEY'))


class AIService:
    def __init__(self):
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = "faq_collection"
        
        # Initialize Qdrant client with cloud configuration
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10.0  # Add timeout to prevent hanging
        )
        
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,  # Size of the vectors from the model
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"Error ensuring collection exists: {str(e)}")
            # Continue without the collection - it will be created when needed
    
    def get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a given text."""
        return self.model.encode(text).tolist()
    
    def search_similar_questions(self, question: str, limit: int = 5) -> List[Tuple[dict, float]]:
        """Search for similar questions in the knowledge base."""
        question_embedding = self.get_embedding(question)
        
        search_result = self.qdrant_client.search(
            collection_name='student_assistance',
            query_vector=question_embedding,
            limit=limit
        )
        
        return [(hit.payload, hit.score) for hit in search_result]
    
    def add_to_knowledge_base(self, question: str, answer: str, confidence_score: float = 1.0):
        """Add a new question-answer pair to the knowledge base."""
        # Create a combined text for embedding
        combined_text = f"Question: {question}\nAnswer: {answer}"
        embedding = self.get_embedding(combined_text)
        
        # Add to Qdrant
        self.qdrant_client.upsert(
            collection_name='student_assistance',
            points=[
                models.PointStruct(
                    id=hash(combined_text) % (2**63 - 1),  # Generate a unique ID
                    vector=embedding,
                    payload={
                        'question': question,
                        'answer': answer,
                        'confidence_score': confidence_score
                    }
                )
            ]
        )
    
    def find_best_answer(self, question: str, confidence_threshold: float = 0.8) -> Optional[Tuple[str, float]]:
        """Find the best matching answer for a question."""
        similar_questions = self.search_similar_questions(question, limit=1)
        
        if not similar_questions:
            # Generate a comprehensive answer using Groq's Llama model
            comprehensive_answer = self.generate_comprehensive_answer(question)
            if comprehensive_answer:
                return comprehensive_answer, 0.7
            return None
        
        best_match, score = similar_questions[0]
        
        if score >= confidence_threshold:
            return best_match['answer'], score
        
        # If the match is not confident enough, generate a comprehensive answer
        comprehensive_answer = self.generate_comprehensive_answer(question)
        if comprehensive_answer:
            return comprehensive_answer, 0.7
        
        return None

    def generate_comprehensive_answer(self, question: str) -> str:
        """Generate a comprehensive answer using Groq's Llama model."""
        try:
            prompt = f"""
            Please provide a detailed and comprehensive answer to the following question. 
            Structure your response with:
            1. A clear introduction
            2. Main points with explanations
            3. Examples where relevant
            4. A brief conclusion
            5. Additional resources or references if applicable

            Question: {question}

            Guidelines:
            - Be thorough and detailed
            - Use clear and academic language
            - Include relevant examples
            - Break down complex concepts
            - Provide context where needed
            """

            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=1,               # creativity level (0.0-1.0)
                max_tokens=2000,            # max tokens in response
                top_p=1,                    # nucleus sampling
                stream=True,                # stream response chunk-by-chunk
                stop=None
            )
            
            # Collect the streamed response
            full_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            return full_response
        except Exception as e:
            print(f"Error generating comprehensive answer: {str(e)}")
            return None 