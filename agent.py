"""
agent.py – Social-to-Lead Agentic Workflow for AutoStream
Built with LangGraph + Claude 3 Haiku

State machine:
  START → agent_node ←→ tool_node → END
"""

import os
import json
from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from rag import retrieve
from tools import mock_lead_capture

# ─────────────────────────────────────────────
# 1.  State definition
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # Lead collection fields (populated incrementally)
    lead_name:     str | None
    lead_email:    str | None
    lead_platform: str | None
    # Tracks whether the lead has already been captured
    lead_captured: bool


# ─────────────────────────────────────────────
# 2.  LLM + tools
# ─────────────────────────────────────────────

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)
tools = [mock_lead_capture]
llm_with_tools = llm.bind_tools(tools)

# ─────────────────────────────────────────────
# 3.  System prompt (injected on every call)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are AutoStream's friendly sales assistant.
AutoStream is a SaaS platform that provides automated video editing tools for content creators.

## Your responsibilities
1. **Greet** users warmly and help them understand AutoStream.
2. **Answer** product, pricing, and policy questions using ONLY the knowledge base context provided.
3. **Detect intent** and classify each message:
   - casual_greeting  → just greet back
   - product_inquiry  → retrieve and share relevant info
   - high_intent_lead → the user is ready or excited to sign up (phrases like "I want to try",
     "sign me up", "let's do it", "how do I start", "I'm interested in the Pro plan", etc.)

4. **Collect lead details** when high intent is detected.
   - Ask for Name, Email, and Creator Platform ONE AT A TIME if not already known.
   - Do NOT call mock_lead_capture until you have all three values.
   - Once you have all three, call mock_lead_capture immediately.

## Tone
Be concise, warm, and helpful. Never fabricate features or prices not in the context.
If you don't know something, say so honestly.
"""


# ─────────────────────────────────────────────
# 4.  Agent node
# ─────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """Main reasoning node — retrieves context, calls the LLM."""

    # Pull the latest user message for RAG retrieval
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    # Retrieve relevant knowledge-base context
    kb_context = retrieve(last_human)

    # Build the system message with fresh context on every turn
    system = SystemMessage(content=(
        SYSTEM_PROMPT
        + f"\n\n## Current Knowledge Base Context\n{kb_context}"
        + _lead_state_hint(state)
    ))

    response = llm_with_tools.invoke([system] + state["messages"])

    # Update partial lead fields if the LLM requested the tool
    new_state: dict = {"messages": [response]}
    new_state.update(_extract_lead_updates(state, response))
    return new_state


def _lead_state_hint(state: AgentState) -> str:
    """Remind the LLM what lead data has already been collected."""
    parts = []
    if state.get("lead_name"):
        parts.append(f"- Name already collected: {state['lead_name']}")
    if state.get("lead_email"):
        parts.append(f"- Email already collected: {state['lead_email']}")
    if state.get("lead_platform"):
        parts.append(f"- Platform already collected: {state['lead_platform']}")
    if state.get("lead_captured"):
        parts.append("- Lead has already been captured. No need to ask again.")
    if parts:
        return "\n\n## Already Collected Lead Info\n" + "\n".join(parts)
    return ""


def _extract_lead_updates(state: AgentState, response: AIMessage) -> dict:
    """
    Peek at any tool calls in the response and pre-populate lead fields
    so subsequent turns know what's been collected.
    """
    updates: dict = {
        "lead_name":     state.get("lead_name"),
        "lead_email":    state.get("lead_email"),
        "lead_platform": state.get("lead_platform"),
        "lead_captured": state.get("lead_captured", False),
    }

    for tc in getattr(response, "tool_calls", []):
        if tc.get("name") == "mock_lead_capture":
            args = tc.get("args", {})
            updates["lead_name"]     = args.get("name",     updates["lead_name"])
            updates["lead_email"]    = args.get("email",    updates["lead_email"])
            updates["lead_platform"] = args.get("platform", updates["lead_platform"])
            updates["lead_captured"] = True

    return updates


# ─────────────────────────────────────────────
# 5.  Routing
# ─────────────────────────────────────────────

def should_use_tool(state: AgentState) -> Literal["tools", END]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ─────────────────────────────────────────────
# 6.  Build the graph
# ─────────────────────────────────────────────

tool_node = ToolNode(tools)

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_use_tool, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")   # After tool execution → back to agent

graph = builder.compile()


# ─────────────────────────────────────────────
# 7.  Convenience runner
# ─────────────────────────────────────────────

def run_agent(user_input: str, state: AgentState) -> tuple[str, AgentState]:
    """
    Send a single user message through the graph and return
    (assistant_reply, updated_state).
    """
    state["messages"].append(HumanMessage(content=user_input))
    new_state = graph.invoke(state)

    # Find the last AI text response
    reply = ""
    for msg in reversed(new_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            # content may be a list of blocks (tool use + text)
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        reply = block["text"]
                        break
            else:
                reply = msg.content
            if reply:
                break

    return reply, new_state


def initial_state() -> AgentState:
    return AgentState(
        messages=[],
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        lead_captured=False,
    )
