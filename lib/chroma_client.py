import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

client = chromadb.CloudClient(
  api_key=os.getenv('CHROMA_DB_API_KEY'),
  tenant='cbc15610-38b7-4771-a196-bb607f72184f',
  database='test-db'
)