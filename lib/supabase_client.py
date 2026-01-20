import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
)

def test_users_table():
    try:
        print("Testing Supabase connection...")
        response = supabase.table("users").select("*").execute()
        
        print(f"✓ Successfully connected to Supabase")
        print(f"✓ Found {len(response.data)} rows in 'users' table\n")
        
        if response.data:
            print("Sample data:")
            for i, user in enumerate(response.data[:3], 1):  # Show first 3 rows
                print(f"  {i}. {user}")
        else:
            print("  (Table is empty)")
        
        return response.data
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_users_table()
