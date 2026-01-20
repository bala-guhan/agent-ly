import warnings
from agent.agent import EnterpriseAgent

warnings.filterwarnings("ignore")


def main():
    print("Initializing Enterprise Agent...")
    agent = EnterpriseAgent()
    print("✓ Agent ready!\n")
    
    thread_id = "default"
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        response = agent.chat(query, thread_id=thread_id)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
