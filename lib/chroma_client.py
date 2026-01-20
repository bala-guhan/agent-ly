import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

# ChromaDB Cloud Client
# Get credentials from environment variables
client = chromadb.CloudClient(
    api_key=os.getenv('CHROMA_DB_API_KEY'),
    tenant=os.getenv('CHROMA_TENANT'),
    database=os.getenv('CHROMA_DATABASE')
)