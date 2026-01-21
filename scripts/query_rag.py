import sys
import time
from pathlib import Path
from rag_query.rag import RAG


parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Rate limiting for Voyage AI (3 RPM = 20 seconds between requests)
rate_limit_delay = 21
last_query_time = 0

rag = RAG(
    collection_name="documents"
)

def rate_limit_check():
    global last_query_time
    current_time = time.time()
    time_since_last = current_time - last_query_time
    
    if time_since_last < rate_limit_delay:
        wait_time = rate_limit_delay - time_since_last
        print(f"⏳ Rate limiting: waiting {wait_time:.1f} seconds...")
        time.sleep(wait_time)
    
    last_query_time = time.time()

# Test queries
test_queries = [
    "What is the engineering team structure?",
    "How does the API authentication work?",
    "What are the security compliance requirements?",
    "Tell me about customer success stories"
]

for i, question in enumerate(test_queries, 1):
    print("="*70)
    print(f"Query {i}: {question}")
    print("="*70)
    
    # Vector search (requires embedding call)
    print("\n[Vector Search Results - Top 3]")
    print("-"*70)
    rate_limit_check()
    results = rag.query(question, k=3)
    
    for j, doc in enumerate(results, 1):
        print(f"\n{j}. Chunk {j}:")
        print(f"   Content: {doc.page_content[:250]}...")
        if doc.metadata:
            print(f"   Metadata: {doc.metadata}")
    
    # Hybrid search (now integrated in query method)
    print("\n[Hybrid Search Results (Vector + BM25) - Top 3]")
    print("-"*70)
    rate_limit_check()
    hybrid_results = rag.query(question, k=3, hybrid_alpha=0.5)
    
    for j, doc in enumerate(hybrid_results, 1):
        print(f"\n{j}. Chunk {j}:")
        print(f"   Content: {doc.page_content[:250]}...")
        if doc.metadata:
            print(f"   Metadata: {doc.metadata}")
    
    print("\n")

# Interactive mode
print("="*70)
print("Interactive Mode - Enter your queries (type 'exit' to quit)")
print("="*70)

while True:
    question = input("\nQuery: ").strip()
    
    if question.lower() in ['exit', 'quit', 'q']:
        print("Exiting...")
        break
    
    if not question:
        continue
    
    k = input("Number of results (default 3): ").strip()
    k = int(k) if k else 3
    
    print(f"\n[Top {k} Results - Vector Search]")
    print("-"*70)
    rate_limit_check()
    results = rag.query(question, k=k)
    
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Chunk {i}:")
        print(f"   Content: {doc.page_content[:300]}...")
        if doc.metadata:
            print(f"   Source: {doc.metadata.get('source', 'N/A')}")
            print(f"   Metadata: {doc.metadata}")

