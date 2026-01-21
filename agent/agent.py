import warnings
import os
from colorama import Fore, Style
from typing import TypedDict, Annotated, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Suppress Hugging Face warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"

from llm import LLMProvider
from agent.tools import create_tools, rag_search, web_search, supabase_query
from agent.memory_manager import MemoryManager
from agent.decision_node import decision_node
from agent.tool_executor import execute_tools_parallel
from agent.synthesis_node import synthesis_node

class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_calls: List[str]
    tool_call_details: List[Dict[str, Any]]
    tool_results: Dict[str, str]
    query: str
    conversation_context: str
    direct_answer: bool
    final_answer: Optional[str]
    decision_reasoning: Optional[str]
    decision_confidence: Optional[float]

def route_after_decision(state: AgentState) -> str:
    """Route after decision: tools, synthesis, or end."""
    if state.get("direct_answer", False):
        return END  # Direct answer goes straight to end
    if state.get("tool_calls"):
        return "tools"
    return "synthesis"  # Fallback if no tools and not direct


class EnterpriseAgent:
    def __init__(
        self,
        llm_provider: str = "groq",
        llm_model: str = "llama-3.3-70b-versatile"
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm = LLMProvider(provider=llm_provider, model=llm_model)
        self.tools = create_tools(llm_provider, llm_model)
        
        if self.tools:
            print(f"Initialized {len(self.tools)} tools: {[t.name for t in self.tools]}")
        
        # Initialize checkpointer using MemoryManager
        self.memory, self._checkpointer_cm = MemoryManager.initialize_checkpointer()
        
        self.graph = self._build_graph()
    
    def _build_graph(self):
        # New architecture: Decision → Tools (parallel) → Synthesis
        workflow = StateGraph(AgentState)
        
        # Create tools dictionary for executor
        tools_dict = {
            "rag_search": rag_search,
            "web_search": web_search,
            "supabase_query": supabase_query
        }
        
        # Decision node: decides which tools to call
        workflow.add_node(
            "decision",
            lambda state: decision_node(state, self.llm, tools_dict)
        )
        
        # Tool executor: executes all tools in parallel
        workflow.add_node(
            "tools",
            lambda state: execute_tools_parallel(state, tools_dict)
        )
        
        # Synthesis node: combines all results
        workflow.add_node(
            "synthesis",
            lambda state: synthesis_node(state, self.llm)
        )
        
        # Set decision as entry point
        workflow.set_entry_point("decision")
        
        # Route after decision
        workflow.add_conditional_edges(
            "decision",
            route_after_decision,
            {
                "tools": "tools",
                "synthesis": "synthesis",
                END: END
            }
        )
        
        # After tools, go to synthesis
        workflow.add_edge("tools", "synthesis")
        
        # Synthesis ends
        workflow.add_edge("synthesis", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def chat(self, query: str, thread_id: str = "default") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Only pass the new message - LangGraph checkpointer will merge with existing state
            # The add_messages reducer automatically appends to previous messages
            # This preserves full conversation history across invocations
            state = {
                "messages": [HumanMessage(content=query)]
            }
            
            result = self.graph.invoke(state, config)
            
            # Return final answer
            if result and "final_answer" in result and result["final_answer"]:
                return result["final_answer"]
            else:
                return "No response generated"
        except Exception as e:
            import traceback
            error_msg = f"Error in chat: {e}\n{traceback.format_exc()}"
            print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def get_history(self, thread_id: str = "default"):
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)
        if state and state.values:
            return state.values.get("messages", [])
        return []


if __name__ == "__main__":
    agent = EnterpriseAgent()
    
    print("Enterprise Agent initialized!")
    print("Type your queries (type 'exit' to quit)\n")
    
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