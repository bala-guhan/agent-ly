import os
from langgraph.checkpoint.memory import MemorySaver


class MemoryManager:
    @staticmethod
    def initialize_checkpointer():
        db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
        
        if db_url:
            try:
                from langgraph.checkpoint.postgres import PostgresSaver
                
                # Setup tables (setup() is idempotent, so safe to call every time)
                checkpointer_cm = PostgresSaver.from_conn_string(db_url)
                print("Setting up PostgreSQL checkpoint tables...")
                with checkpointer_cm as cp:
                    cp.setup()
                print("✓ Checkpoint tables ready")
                
                # Enter the context manager to get the actual saver instance
                # For long-lived applications, we keep the context manager open
                checkpointer = checkpointer_cm.__enter__()
                print("✓ Using PostgreSQL checkpointer for persistent memory")
                return checkpointer, checkpointer_cm
                    
            except ImportError:
                print("PostgresSaver not available. Install langgraph-checkpoint-postgres. Falling back to MemorySaver.")
                return MemorySaver(), None
            except Exception as e:
                print(f"Failed to initialize PostgreSQL checkpointer: {e}. Falling back to MemorySaver.")
                return MemorySaver(), None
        else:
            print("Using MemorySaver (in-memory). Set DATABASE_URL or SUPABASE_DB_URL for persistent memory.")
            return MemorySaver(), None

