import warnings
from colorama import Fore, Style
from langchain_core.messages import SystemMessage, HumanMessage
from agent.tools import ToolDecision

warnings.filterwarnings("ignore")


def _format_messages(messages, limit=None):
    """Format messages for context. Returns formatted string."""
    from langchain_core.messages import HumanMessage
    messages_to_format = messages[-limit:] if limit else messages
    return "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content if hasattr(m, 'content') else str(m)}"
        for m in messages_to_format
    ])


def decision_node(state, llm, tools_dict):
    """Decide which tools to call using structured output."""
    from prompts.decision_prompt import get_decision_prompt
    
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    # Format conversation - full for decision, last 3 for web_search context
    full_conversation = _format_messages(messages)
    conversation_context = _format_messages(messages, limit=3)
    
    # Create structured LLM for decision
    structured_llm = llm.llm.with_structured_output(ToolDecision)
    
    # Get decision with full conversation context
    prompt = get_decision_prompt(query, full_conversation)
    decision = structured_llm.invoke(prompt)
    
    # Log decision with reasoning and confidence
    print(f"{Fore.YELLOW}[decision]{Style.RESET_ALL} confidence={decision.confidence:.2f}")
    print(f"{Fore.YELLOW}[decision]{Style.RESET_ALL} reasoning: {decision.reasoning[:100]}...")
    
    # If direct answer with sufficient confidence
    if decision.direct_answer:
        if decision.confidence > 0.7:  # High confidence threshold
            print(f"{Fore.YELLOW}[general_reply]{Style.RESET_ALL}")
            from agent.tools import _general_llm
            if _general_llm:
                answer = _general_llm.invoke(query)
                return {
                    "tool_calls": [],
                    "tool_call_details": [],
                    "direct_answer": True,
                    "query": query,
                    "decision_reasoning": decision.reasoning,
                    "decision_confidence": decision.confidence,
                    "final_answer": answer
                }
            return {
                "tool_calls": [],
                "tool_call_details": [],
                "direct_answer": True,
                "query": query,
                "decision_reasoning": decision.reasoning,
                "decision_confidence": decision.confidence,
                "final_answer": "I'm here to help! What would you like to know?"
            }
        else:
            # Low confidence in direct answer - might need tools after all
            # Fallback: treat as if tools might be needed
            print(f"{Fore.YELLOW}[decision]{Style.RESET_ALL} Low confidence ({decision.confidence:.2f}) in direct answer, reconsidering...")
    
    # Filter tools by confidence threshold and remove duplicates
    MIN_TOOL_CONFIDENCE = 0.5
    seen_tools = {}
    for tc in decision.tool_calls:
        if tc.confidence > MIN_TOOL_CONFIDENCE and tc.tool_name not in seen_tools:
            seen_tools[tc.tool_name] = tc
    
    unique_tool_details = list(seen_tools.values())
    tool_calls = list(seen_tools.keys())
    
    # Log each tool call with its reasoning and confidence
    for tc in unique_tool_details:
        print(f"{Fore.MAGENTA}[tool_call]{Style.RESET_ALL} {tc.tool_name} (confidence={tc.confidence:.2f}): {tc.reasoning[:60]}...")
    
    return {
        "tool_calls": tool_calls,
        "tool_call_details": [{"tool_name": tc.tool_name, "reasoning": tc.reasoning, "confidence": tc.confidence} for tc in unique_tool_details],
        "direct_answer": False,
        "query": query,
        "conversation_context": conversation_context,
        "decision_reasoning": decision.reasoning,
        "decision_confidence": decision.confidence
    }

