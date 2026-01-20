import sys
import time
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from data_ingestion.inject_data import RAGSystem

rag = RAGSystem(collection_name="documents", use_cloud=True)
print("Using Cloud ChromaDB instance")

data_files = list((parent_dir / "data").glob("*.txt"))
print(f"Processing {len(data_files)} files...\n")

all_ids = []
rate_limit_delay = 21  

for i, file_path in enumerate(data_files, 1):
    print(f"[{i}/{len(data_files)}] Processing: {file_path.name}...")
    try:
        ids = rag.add_documents(file_path, auto_chunk=True)
        all_ids.extend(ids)
        print(f"  ✓ Added {len(ids)} chunks")
        
        if i < len(data_files):
            print(f"  ⏳ Waiting {rate_limit_delay} seconds (rate limit)...\n")
            time.sleep(rate_limit_delay)
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        if i < len(data_files):
            print(f"  ⏳ Waiting {rate_limit_delay} seconds before retry...\n")
            time.sleep(rate_limit_delay)

print(f"\n{'='*60}")
print(f"Complete! Added {len(all_ids)} total chunks from {len(data_files)} files")