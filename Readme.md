# AutoStream AI Agent
### Social-to-Lead Agentic Workflow | Built for ServiceHive Inflx Platform

A conversational AI agent for **AutoStream** — an automated video editing SaaS for content creators. The agent identifies user intent, answers product questions using RAG, and captures qualified leads via tool execution.

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Your API Key
```bash
cp .env.example .env
```
Then open `.env` and replace with your actual Groq API key:
```
GEMINI_API_KEY = "your gemini api key here"
```
Get a free Gemini api key key at: https://aistudio.google.com/api-keys

### 4. Run the Agent
```bash
python agent.py
```

### 5. Sample Conversation to Test
```
You: Hi, tell me about your pricing
You: What's included in the Pro plan?
You: That sounds great, I want to sign up for Pro for my YouTube channel
You: Alex Johnson
You: alex@gmail.com
You: YouTube
```

---

## 🏗️ Architecture Explanation

This agent is built using **LangGraph** as the orchestration framework, **Gemini 1.5 Flash** as the LLM, and a local JSON file for RAG-based knowledge retrieval.

**Why LangGraph?** LangGraph models the conversation as a typed state machine, making it ideal for multi-step agentic workflows where different actions must fire depending on the current state. Unlike simple LangChain chains, LangGraph allows conditional routing between nodes (intent detection → response generation) while maintaining a persistent, typed state dictionary across all conversation turns. This prevents the agent from triggering lead capture prematurely — a hard requirement of this project.

**How State is Managed:** A single `AgentState` TypedDict carries the full conversation history (`messages`), detected intent, collected lead fields (`lead_name`, `lead_email`, `lead_platform`), and two control flags (`collecting_lead`, `lead_captured`). Every turn, the graph receives the updated state, runs `detect_intent` then `agent_respond`, and returns the mutated state. The host loop passes this state back into `app.invoke()` on the next turn, giving the agent persistent memory across 5–6+ conversation turns without any external memory store.

**RAG Pipeline:** The knowledge base (`knowledge_base.json`) is loaded at startup and injected into the system prompt as a grounded context string. The LLM is explicitly instructed to answer only from this context, preventing hallucination of pricing or policy details.

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the following architecture would be used:

**Stack:** WhatsApp Business API (Meta Cloud API) + a FastAPI web server + this LangGraph agent

**Step-by-step integration:**

1. **Register a WhatsApp Business Account** on Meta for Developers and obtain a Phone Number ID and Access Token.

2. **Set up a Webhook endpoint** using FastAPI:
```python
from fastapi import FastAPI, Request
app = FastAPI()

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    body = await request.json()
    user_message = body["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]
    phone_number = body["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
    
    # Retrieve or create session state for this phone number
    state = session_store.get(phone_number, default_state())
    state["messages"].append(HumanMessage(content=user_message))
    
    # Run through LangGraph agent
    state = app_graph.invoke(state)
    session_store[phone_number] = state
    
    # Send reply back via WhatsApp API
    reply = get_last_ai_message(state)
    send_whatsapp_message(phone_number, reply)
    return {"status": "ok"}
```

3. **Session Management:** Each WhatsApp phone number maps to its own `AgentState` stored in a dictionary (or Redis for production). This gives every user their own persistent conversation memory.

4. **Verify Webhook:** Meta requires a GET `/webhook` endpoint for initial verification using a `hub.verify_token`.

5. **Deploy** the FastAPI server to any cloud (Render, Railway, AWS EC2) with a public HTTPS URL, then register that URL in the Meta Developer dashboard.

This approach requires no changes to the core LangGraph agent — the WhatsApp layer is purely a transport adapter around the same agent logic.

---

## 📁 Project Structure

```
autostream-agent/
├── agent.py              # Main agent — LangGraph + intent detection + RAG + tool
├── knowledge_base.json   # Local knowledge base (pricing, features, policies)
├── requirements.txt      # Python dependencies
├── .env.example          # API key template
└── README.md             # This file
```

---

## 🧠 Intent Classification

| Intent | Trigger Examples |
|---|---|
| `greeting` | "Hi", "Hello", "Hey there" |
| `product_inquiry` | "What's the pricing?", "Do you have 4K support?" |
| `high_intent` | "I want to sign up", "Let's get started with Pro" |

## 🎥 Demo Video

[Watch the demo](https://youtu.be/n3qOZ01d_yU)

---

## ✅ Evaluation Checklist

- [x] Intent detection (3 categories)
- [x] RAG from local knowledge base
- [x] State management across 5–6 turns (LangGraph TypedDict)
- [x] Lead collection — name, email, platform in sequence
- [x] `mock_lead_capture()` called only after all 3 fields collected
- [x] No premature tool execution
- [x] Clean modular code structure
