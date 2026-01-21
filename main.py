import warnings
import os
import uuid
from colorama import init, Fore, Style

# Initialize colorama for Windows support
init(autoreset=True)

# Suppress Hugging Face warnings before any imports
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"

from agent.agent import EnterpriseAgent


def main():
    print("Initializing Enterprise Agent...")
    agent = EnterpriseAgent()
    print("✓ Agent ready!\n")
    
    # Generate a unique thread ID for this session
    thread_id = str(uuid.uuid4())
    print(f"Session ID: {thread_id}\n")
    
    while True:
        query = input(f"{Fore.CYAN}You: {Style.RESET_ALL}").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
            break
        
        if not query:
            continue
        
        response = agent.chat(query, thread_id=thread_id)
        print(f"{Fore.GREEN}Agent: {Style.RESET_ALL}{response}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
