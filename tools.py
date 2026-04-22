"""
tools.py – Tool definitions for the AutoStream agent.
"""

import json
from langchain_core.tools import tool


@tool
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Captures a qualified lead once name, email, and creator platform
    have all been collected.

    Args:
        name:     Full name of the prospective customer.
        email:    Email address of the prospective customer.
        platform: Creator platform (e.g. YouTube, Instagram, TikTok).

    Returns:
        A confirmation string.
    """
    print(f"\n✅ Lead captured successfully: {name}, {email}, {platform}\n")
    return json.dumps({
        "status": "success",
        "message": f"Lead captured! Welcome aboard, {name}. "
                   f"We'll reach out at {email} with your Pro plan details.",
        "lead": {"name": name, "email": email, "platform": platform},
    })
