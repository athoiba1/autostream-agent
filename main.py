"""
main.py – CLI chat loop for the AutoStream agent.

Usage:
    python main.py

The agent maintains full conversation state across turns.
Type 'exit' or 'quit' to end the session.
"""

from agent import run_agent, initial_state

BANNER = """
╔══════════════════════════════════════════════════╗
║   AutoStream AI Sales Assistant  🎬              ║
║   Powered by Claude + LangGraph                  ║
║   Type 'exit' to quit                            ║
╚══════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)
    state = initial_state()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Agent: Thanks for chatting! Have a great day. 👋")
            break

        reply, state = run_agent(user_input, state)
        print(f"\nAgent: {reply}\n")


if __name__ == "__main__":
    main()
