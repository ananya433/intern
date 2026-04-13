import os
import sys
import time
from typing import Iterator

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("  Install: pip install google-genai")
    sys.exit(1)

# ──────────────────────────────────────────────
# Set your API key as an environment variable:
#   Windows:  set GEMINI_API_KEY=your_key_here
#   Mac/Linux: export GEMINI_API_KEY=your_key_here
#
# OR paste it directly below as a fallback:
# ──────────────────────────────────────────────
API_KEY: str = os.environ.get("GEMINI_API_KEY", "PASTE_YOUR_KEY_HERE")

MODEL: str = "gemini-1.5-flash"  # stable & free tier friendly

SYSTEM_INSTRUCTION: str = (
    "You are a helpful, friendly, and knowledgeable AI assistant. "
    "Provide clear, concise, and accurate responses. "
    "If you're unsure about something, say so honestly."
)

BANNER = """
╔══════════════════════════════════════════════╗
║          🤖 Gemini AI Chatbot               ║
╠══════════════════════════════════════════════╣
║  Model: gemini-1.5-flash (free tier)        ║
║                                              ║
║  Commands:                                   ║
║    /reset   - Clear conversation history     ║
║    /stats   - Show conversation stats        ║
║    /help    - Show this help message         ║
║    /quit    - Exit the chatbot               ║
╚══════════════════════════════════════════════╝
"""


class GeminiChatbot:
    def __init__(
        self,
        client: genai.Client,
        model: str = MODEL,
        system_instruction: str = SYSTEM_INSTRUCTION,
    ) -> None:
        self.client = client
        self.model = model
        self.history: list[types.Content] = []
        self.turn_count: int = 0
        self.config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
        )

    def send_message_stream(self, user_input: str) -> Iterator[str]:
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
        )

        try:
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=self.history,
                config=self.config,
            )

            full_reply = ""
            for chunk in stream:
                if chunk.text:
                    full_reply += chunk.text
                    yield chunk.text

            self.history.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=full_reply)],
                )
            )
            self.turn_count += 1

        except Exception as e:
            self.history.pop()  # roll back user message on failure
            raise e

    def get_stats(self) -> str:
        return (
            f"📊 Model: {self.model} | "
            f"{self.turn_count} exchanges | "
            f"{len(self.history)} messages in history"
        )

    def reset(self) -> None:
        self.history.clear()
        self.turn_count = 0


def test_connection(client: genai.Client) -> None:
    """Sends a minimal request to verify the key and model work."""
    client.models.generate_content(
        model=MODEL,
        contents="Hi",
        config=types.GenerateContentConfig(max_output_tokens=5),
    )


def main() -> None:
    if API_KEY in ("PASTE_YOUR_KEY_HERE", "") or not API_KEY.strip():
        print("\n  API KEY NOT SET!")
        print("  Option 1: export GEMINI_API_KEY=your_key  (then re-run)")
        print("  Option 2: open chatbot.py and paste your key into API_KEY")
        print("  Get a free key: https://aistudio.google.com/app/apikey")
        sys.exit(1)

    print(BANNER)
    print("  Connecting...", end=" ", flush=True)

    client = genai.Client(api_key=API_KEY)

    try:
        test_connection(client)
        print("✅ Connected!\n")
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            print("⚠️  Rate limited. Wait 1 minute and try again.")
        else:
            print(f"❌ Connection failed: {e}")
        sys.exit(1)

    bot = GeminiChatbot(client=client)
    print("  Type your message and press Enter.\n")

    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            print("👋 Goodbye!")
            break
        elif cmd == "/reset":
            bot.reset()
            print("🔄 History cleared!\n")
            continue
        elif cmd == "/stats":
            print(bot.get_stats() + "\n")
            continue
        elif cmd == "/help":
            print(BANNER)
            continue

        print("\n🤖 Gemini: ", end="", flush=True)
        try:
            for chunk in bot.send_message_stream(user_input):
                print(chunk, end="", flush=True)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print("\n⚠️  Rate limited! Waiting 60 seconds...", flush=True)
                time.sleep(60)
                print("✅ Ready — please retype your message.\n")
            else:
                print(f"\n⚠️  Error: {e}\n")
            continue

        print("\n")


if __name__ == "__main__":
    main()
