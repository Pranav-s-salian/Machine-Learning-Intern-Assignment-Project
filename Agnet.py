import os
import json
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google import genai
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

load_dotenv()


# ─────────────────────────────────────────────
# 0. PYDANTIC RESPONSE MODELS
# ─────────────────────────────────────────────

class AgentResponse(BaseModel):
    """Structured response from the AI agent."""
    message: str = Field(description="The agent's response message")
    intent: str = Field(description="Detected user intent")
    requires_action: bool = Field(default=False, description="Whether action is needed")
    action_type: Optional[str] = Field(default=None, description="Type of action if needed")


class LeadCollectionResponse(BaseModel):
    """Response during lead collection phase."""
    message: str = Field(description="Message to send to user")
    field_requesting: str = Field(description="Which field is being requested")
    next_step: str = Field(description="What happens next")


class LeadData(BaseModel):
    """Structured lead information."""
    name: str = Field(description="Lead name")
    email: str = Field(description="Lead email address")
    platform: str = Field(description="Creator platform (YouTube, Instagram, TikTok, etc.)")

# ─────────────────────────────────────────────
# 1. LOAD KNOWLEDGE BASE (RAG)
# ─────────────────────────────────────────────

def load_knowledge_base() -> str:
    """Load and format the local knowledge base into a readable string for RAG."""
    with open("knowledge_base.json", "r") as f:
        kb = json.load(f)

    rag_text = f"""
PRODUCT: {kb['product']} — {kb['tagline']}

PRICING PLANS:
- Basic Plan: {kb['plans']['basic']['price']}
  Features: {', '.join(kb['plans']['basic']['features'])}

- Pro Plan: {kb['plans']['pro']['price']}
  Features: {', '.join(kb['plans']['pro']['features'])}

COMPANY POLICIES:
- Refund Policy: {kb['policies']['refund_policy']}
- Support Policy: {kb['policies']['support_policy']}
- Trial Policy: {kb['policies']['trial_policy']}

FAQ:
""" + "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in kb['faq']])

    return rag_text


KNOWLEDGE_BASE = load_knowledge_base()


# ─────────────────────────────────────────────
# 2. MOCK LEAD CAPTURE TOOL
# ─────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API function to capture a qualified lead."""
    print(f"\n{'-'*60}")
    print(f"[LEAD CAPTURED]")
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Platform: {platform}")
    print(f"{'-'*60}\n")
    return f"Lead successfully registered: {name}, {email}, {platform}"


# ─────────────────────────────────────────────
# 3. LANGGRAPH STATE DEFINITION
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: List                  # Full conversation history
    intent: str                     # Current detected intent
    lead_name: Optional[str]        # Collected lead name
    lead_email: Optional[str]       # Collected lead email
    lead_platform: Optional[str]    # Collected creator platform
    lead_captured: bool             # Whether mock_lead_capture has been called
    collecting_lead: bool           # Whether we are in lead collection mode


# ─────────────────────────────────────────────
# 4. LLM SETUP (GROQ)
# ─────────────────────────────────────────────

def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
    return genai.Client(api_key=api_key)


# ─────────────────────────────────────────────
# 5. INTENT DETECTION NODE
# ─────────────────────────────────────────────

def detect_intent(state: AgentState) -> AgentState:
    """Classify the latest user message into one of three intent categories."""

    if state.get("collecting_lead"):
        return state

    llm = get_llm()

    last_user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break

    intent_prompt = f"""You are an intent classifier for AutoStream, a SaaS video editing product.

Classify the following user message into EXACTLY one of these three categories:
1. "greeting"      — casual hello, small talk, general intro
2. "product_inquiry" — asking about features, pricing, plans, policies
3. "high_intent"   — user clearly wants to sign up, buy, start a trial, or try the product

User message: "{last_user_message}"

Respond with ONLY one word: greeting, product_inquiry, or high_intent
"""

    response = llm.models.generate_content(
        model="gemini-1.5-flash",
        contents=intent_prompt,
    )
    detected = response.text.strip().lower()

    if detected not in ["greeting", "product_inquiry", "high_intent"]:
        detected = "product_inquiry"

    state["intent"] = detected
    return state


# ─────────────────────────────────────────────
# 6. MAIN AGENT RESPONSE NODE
# ─────────────────────────────────────────────

def agent_respond(state: AgentState) -> AgentState:
    """Generate the agent's response based on intent and current state."""
    llm = get_llm()

    # ── If we are mid lead-collection, handle field gathering ──
    if state.get("collecting_lead"):
        last_user_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content.strip()
                break

        # Fill fields in order: name → email → platform
        if not state.get("lead_name"):
            state["lead_name"] = last_user_msg
            response_text = f"Thanks, {state['lead_name']}! What's your email address?"

        elif not state.get("lead_email"):
            state["lead_email"] = last_user_msg
            response_text = "Perfect! Which creator platform do you primarily use? (e.g., YouTube, Instagram, TikTok)"

        elif not state.get("lead_platform"):
            state["lead_platform"] = last_user_msg

            # ── All 3 collected → trigger mock_lead_capture ──
            mock_lead_capture(
                name=state["lead_name"],
                email=state["lead_email"],
                platform=state["lead_platform"]
            )
            state["lead_captured"] = True
            state["collecting_lead"] = False

            response_text = (
                f"You're all set, {state['lead_name']}! "
                f"We've registered your interest in the AutoStream Pro plan.\n"
                f"Our team will reach out to {state['lead_email']} shortly.\n"
                f"Welcome to AutoStream — built for creators like you on {state['lead_platform']}!"
            )
        else:
            response_text = "Is there anything else I can help you with?"

        state["messages"].append(AIMessage(content=response_text))
        return state

    # ── Intent-based routing ──
    intent = state.get("intent", "product_inquiry")

    if intent == "high_intent":
        # Start lead collection
        state["collecting_lead"] = True
        response_text = (
            "That's awesome! I'd love to get you set up with AutoStream Pro.\n"
            "Let me grab a few quick details.\n\n"
            "First, what's your name?"
        )
        state["messages"].append(AIMessage(content=response_text))
        return state

    # ── For greeting and product_inquiry: use RAG + LLM ──
    system_prompt = f"""You are a friendly and knowledgeable sales assistant for AutoStream — 
an AI-powered video editing SaaS for content creators.

Use ONLY the following knowledge base to answer product and pricing questions.
Do NOT make up any features or prices not listed below.

IMPORTANT FORMATTING RULES:
- Do NOT use asterisks (*), markdown, or any special characters
- Format lists using plain text with dashes (-) or numbers (1, 2, 3)
- Keep text clean, simple, and easy to read
- No bold, italics, or emoji formatting
- Write in a conversational, friendly tone

--- KNOWLEDGE BASE ---
{KNOWLEDGE_BASE}
--- END KNOWLEDGE BASE ---

Conversation rules:
- For greetings: respond warmly, introduce AutoStream briefly, ask how you can help
- For product/pricing questions: answer accurately using the knowledge base only
- When listing features, use plain numbered or bulleted format (no asterisks)
- Keep responses concise, friendly, and helpful
- Never ask for personal details unless the user expresses intent to sign up
"""

    # Build conversation context
    messages_text = system_prompt + "\n\nConversation:\n"
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            messages_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            messages_text += f"Agent: {msg.content}\n"
    
    response = llm.models.generate_content(
        model="gemini-2.0-flash",
        contents=messages_text,
    )

    state["messages"].append(AIMessage(content=response.text))
    return state


# ─────────────────────────────────────────────
# 7. ROUTING LOGIC
# ─────────────────────────────────────────────

def should_end(state: AgentState) -> str:
    """Decide whether to end or continue the graph."""
    return END


# ─────────────────────────────────────────────
# 8. BUILD LANGGRAPH GRAPH
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent", detect_intent)
    graph.add_node("agent_respond", agent_respond)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "agent_respond")
    graph.add_edge("agent_respond", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 9. MAIN CHAT LOOP
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  Welcome to AutoStream AI Assistant")
    print("  Powered by Inflx | ServiceHive")
    print("="*55)
    print("  Type 'quit' or 'exit' to end the conversation.\n")

    app = build_graph()

    # Initialize state
    state: AgentState = {
        "messages": [],
        "intent": "",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False,
    }

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("\nAgent: Thanks for chatting! Have a great day.\n")
            break

        # Append user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Run through LangGraph
        state = app.invoke(state)

        # Print agent's last response
        last_ai_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai_msg = msg.content
                break

        print(f"\nAgent: {last_ai_msg}\n")


if __name__ == "__main__":
    main() 