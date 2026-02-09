from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def clear_qdrant_collection():
    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=os.getenv('QDRANT_URL', 'https://7775af46-4796-47d4-ab44-00c855e262f0.europe-west3-0.gcp.cloud.qdrant.io:6333'),
            api_key=os.getenv('QDRANT_API_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.a_inwL3e0AkODn1eTDyN5crtGKHQGZ0ddIh1wHvHCLY')
        )

        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if "student_faqs" not in collection_names:
            logger.warning("student_faqs collection does not exist")
            return False

        # Delete all points from the collection
        client.delete_collection(collection_name="student_faqs")
        logger.info("Successfully deleted student_faqs collection")

        # Recreate the collection with the same configuration
        client.recreate_collection(
            collection_name="student_faqs",
            vectors_config={
                "size": 384,  # Size for all-MiniLM-L6-v2 model
                "distance": "Cosine"
            }
        )
        logger.info("Successfully recreated student_faqs collection")
        
        return True

    except Exception as e:
        logger.error(f"Error clearing Qdrant collection: {str(e)}")
        return False

if __name__ == "__main__":
    if clear_qdrant_collection():
        print("Successfully cleared and recreated student_faqs collection")
    else:
        print("Failed to clear student_faqs collection") 