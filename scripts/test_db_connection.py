"""Test database connection and table creation."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")

if not db_url:
    print("❌ No DATABASE_URL or SUPABASE_DB_URL found in .env")
    sys.exit(1)

print(f"✓ Found database URL: {db_url[:30]}...")

# Check if using direct connection (db.*.supabase.co) - these can fail if project is paused
if "db." in db_url and ".supabase.co" in db_url:
    print("\n⚠️  WARNING: Using direct connection (db.*.supabase.co)")
    print("   This requires your Supabase project to be active.")
    print("   Free tier projects pause after inactivity - check your Supabase dashboard.")
    print("   Consider using the pooler connection string instead:\n")
    print("   postgresql://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres\n")

print("\nAttempting to connect and create tables...\n")

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    
    print("1. Creating PostgresSaver instance...")
    # PostgresSaver.from_conn_string() returns a context manager
    checkpointer = PostgresSaver.from_conn_string(db_url)
    print("   ✓ Connected to database")
    
    print("\n2. Running setup() to create tables...")
    # Use as context manager or call setup() properly
    with checkpointer as cp:
        cp.setup()
    print("   ✓ Tables created successfully!")
    
    print("\n3. Verifying tables exist...")
    # Try to query the checkpoints table
    import psycopg2
    from urllib.parse import urlparse
    
    parsed = urlparse(db_url)
    conn = psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        user=parsed.username,
        password=parsed.password,
        database=parsed.path[1:] if parsed.path else 'postgres'
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name LIKE 'checkpoints%'
        ORDER BY table_name;
    """)
    tables = cur.fetchall()
    cur.close()
    conn.close()
    
    if tables:
        print(f"   ✓ Found {len(tables)} checkpoint table(s):")
        for table in tables:
            print(f"     - {table[0]}")
    else:
        print("   ⚠ No checkpoint tables found (but setup() succeeded)")
    
    print("\n✅ Database setup complete!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install the package:")
    print("pip install langgraph-checkpoint-postgres")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

