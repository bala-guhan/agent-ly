import warnings
import logging
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import os

from llm import LLMProvider
from agent.tools import create_tools

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def should_continue(state: AgentState) -> str:
    """Decide whether to continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
        
    return END

def call_model(state: AgentState, llm, tools):
    """Call the LLM with tools and conversation history."""
    from groq import BadRequestError
    
    messages = state["messages"]
    
    try:
        # Bind tools to LLM - simple ReAct pattern
        # Let the LLM decide whether to use tools or answer directly
        llm_with_tools = llm.llm.bind_tools(tools)
        response = llm_with_tools.invoke(messages)
        
        # Check for invalid tool calls
        if hasattr(response, "invalid_tool_calls") and response.invalid_tool_calls:
            logger.error(f"Invalid tool calls detected: {response.invalid_tool_calls}")
        
        return {"messages": [response]}
    except BadRequestError as e:
        # Groq API rejected the tool call format - fallback to no tools
        logger.warning(f"Groq API rejected tool call: {e}. Retrying without tools.")
        try:
            response = llm.llm.invoke(messages)
            return {"messages": [response]}
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content="I'm having trouble processing your request. Please try again.")]}
    except Exception as e:
        # Other errors - log and fallback
        logger.error(f"Tool calling failed: {e}")
        try:
            response = llm.llm.invoke(messages)
            return {"messages": [response]}
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content="I encountered an error. Please try again.")]}


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
        
        # Use PostgreSQL checkpointer if DATABASE_URL is available, else fallback to MemorySaver
        db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
        if db_url:
            try:
                # Try to use PostgreSQL checkpointer for persistent memory
                from langgraph.checkpoint.postgres import PostgresSaver
                # Create checkpointer instance (this is reusable and manages connections internally)
                checkpointer = PostgresSaver.from_conn_string(db_url)
                # Initialize tables (only needed once, safe to call multiple times)
                # setup() creates tables if they don't exist
                with checkpointer as cp:
                    cp.setup()
                # Store the checkpointer for use in graph compilation
                self.memory = checkpointer
                logger.info("✓ Using PostgreSQL checkpointer for persistent memory")
            except ImportError:
                logger.warning("PostgresSaver not available. Install langgraph-checkpoint-postgres. Falling back to MemorySaver.")
                self.memory = MemorySaver()
            except Exception as e:
                logger.warning(f"Failed to initialize PostgreSQL checkpointer: {e}. Falling back to MemorySaver.")
                self.memory = MemorySaver()
        else:
            self.memory = MemorySaver()
            logger.info("Using MemorySaver (in-memory). Set DATABASE_URL or SUPABASE_DB_URL for persistent memory.")
        
        self.graph = self._build_graph()
    
    def _build_graph(self):
        # Simple ReAct agent structure
        tool_node = ToolNode(self.tools)
        
        workflow = StateGraph(AgentState)
        
        # Single agent node - LLM decides whether to use tools or answer directly
        workflow.add_node(
            "agent",
            lambda state: call_model(state, self.llm, self.tools)
        )
        workflow.add_node("tools", tool_node)
        
        # Set agent as entry point
        workflow.set_entry_point("agent")
        
        # Conditional edges: if tool calls exist, go to tools, else end
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        
        # After tools execute, always return to agent to process results
        workflow.add_edge("tools", "agent")
        
        return workflow.compile(checkpointer=self.memory)
    
    def chat(self, query: str, thread_id: str = "default") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Just add the new message - checkpointer handles the rest!
            # The add_messages reducer will automatically append to existing messages
            state = {"messages": [HumanMessage(content=query)]}
            result = self.graph.invoke(state, config)
            
            # Get the last message from the result
            if result and "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                return str(last_message)
            else:
                return "No response generated"
        except Exception as e:
            import traceback
            error_msg = f"Error in chat: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            print(f"Error: {e}")
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