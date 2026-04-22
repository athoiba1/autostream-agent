# AutoStream AI Sales Agent
### Social-to-Lead Agentic Workflow · ServiceHive / Inflx Assignment

A production-grade conversational AI agent that converts social media conversations into qualified leads for **AutoStream**, an automated video editing SaaS for content creators.

---

## Quick Start

### 1. Clone & install dependencies
```bash
git clone <your-repo-url>
cd autostream_agent
pip install -r requirements.txt
```

### 2. Set your API key
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run
```bash
python main.py
```

---

## Project Structure
```
autostream_agent/
├── agent.py                  # LangGraph graph, state, agent node
├── rag.py                    # Local knowledge base retrieval
├── tools.py                  # mock_lead_capture tool
├── main.py                   # CLI chat loop
├── requirements.txt
├── knowledge_base/
│   └── autostream_kb.json    # Pricing, features, policies
└── README.md
```

---

## Architecture (~200 words)

### Why LangGraph?
LangGraph was chosen over AutoGen because it provides **explicit, inspectable state management** via a typed `TypedDict` schema. Every field — partial lead data (`lead_name`, `lead_email`, `lead_platform`), conversation history, and `lead_captured` flag — lives in a single `AgentState` object that persists across all turns. This makes it easy to reason about *exactly* what the agent knows at any point, which is critical for guarding against premature tool calls.

### How it works
The graph has two nodes: **`agent`** (the Claude-powered reasoning node) and **`tools`** (LangGraph's `ToolNode`). A conditional edge routes from `agent → tools` when Claude decides to call `mock_lead_capture`, then back to `agent` so Claude can read the tool result and compose a final reply. This loop continues until no tool call is made, at which point the graph exits.

**RAG** is a lightweight keyword-overlap retriever over a local JSON knowledge base. On every turn, the top-3 most relevant chunks are injected directly into the system prompt — no vector DB needed for a focused domain like this.

**State** is passed explicitly through `graph.invoke()`. Because LangGraph appends messages via an `add_messages` reducer, the full conversation history is always available without a separate memory buffer.

---

## WhatsApp Deployment via Webhooks

### Overview
To deploy this agent on WhatsApp, use the **WhatsApp Cloud API (Meta)** combined with a lightweight webhook server.

### Step-by-step

1. **Register a Meta Business App** and enable the WhatsApp product. Obtain a `PHONE_NUMBER_ID` and a permanent `ACCESS_TOKEN`.

2. **Set up a webhook endpoint** (e.g. FastAPI):
```python
@app.post("/webhook")
async def receive_message(payload: dict):
    message = payload["entry"][0]["changes"][0]["value"]["messages"][0]
    user_id = message["from"]          # WhatsApp phone number
    text    = message["text"]["body"]

    # Load or create per-user state from a Redis / DB store
    state = load_state(user_id) or initial_state()
    reply, state = run_agent(text, state)
    save_state(user_id, state)         # Persist across turns

    send_whatsapp_message(user_id, reply)
```

3. **Persist state per user** — store `AgentState` in Redis or a lightweight DB (keyed by WhatsApp phone number) so each user's conversation history and partial lead data survive between messages.

4. **Send replies** using the WhatsApp Cloud API:
```python
requests.post(
    f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages",
    headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
    json={"messaging_product": "whatsapp", "to": user_id,
          "type": "text", "text": {"body": reply}}
)
```

5. **Verify the webhook** by handling Meta's `GET` challenge request on startup.

6. **Deploy** the FastAPI app on any cloud provider (Railway, Render, AWS Lambda) behind an HTTPS URL and register it in the Meta Developer Console.

This architecture scales to any number of users since state is externalized, and the same LangGraph agent runs statelessly per request.

---

## Evaluation Checklist
| Criterion | Implementation |
|---|---|
| Intent detection | System prompt classifies: `casual_greeting`, `product_inquiry`, `high_intent_lead` |
| RAG | `rag.py` — keyword retrieval from `autostream_kb.json` injected per turn |
| State management | `AgentState` TypedDict, persisted via `graph.invoke()` across 5–6 turns |
| Tool calling | `mock_lead_capture` only fires after all 3 fields collected |
| Code clarity | Modular: `rag.py`, `tools.py`, `agent.py`, `main.py` |
| Deployability | WhatsApp webhook architecture documented above |
