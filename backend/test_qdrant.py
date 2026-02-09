from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_qdrant_connection():
    try:
        # Get credentials
        url = os.getenv('QDRANT_URL')
        api_key = os.getenv('QDRANT_API_KEY')
        
        if not url:
            raise ValueError("QDRANT_URL environment variable is not set")
        if not api_key:
            raise ValueError("QDRANT_API_KEY environment variable is not set")
        
        logger.info(f"Testing connection to Qdrant at: {url}")
        logger.info(f"API Key: {'*' * len(api_key) if api_key else 'Not set'}")
        
        # Initialize client
        client = QdrantClient(url=url, api_key=api_key)
        
        # Get collections
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        logger.info(f"Available collections: {collection_names}")
        
        # Check if student_faqs collection exists
        if "student_faqs" not in collection_names:
            logger.info("Creating student_faqs collection...")
            client.create_collection(
                collection_name="student_faqs",
                vectors_config=models.VectorParams(
                    size=384,  # Size for all-MiniLM-L6-v2 model
                    distance=models.Distance.COSINE
                )
            )
            logger.info("student_faqs collection created successfully!")
        else:
            logger.info("student_faqs collection already exists!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Qdrant connection: {str(e)}")
        return False

if __name__ == "__main__":
    test_qdrant_connection() 