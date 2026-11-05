import os

from dotenv import load_dotenv

# Reads ".env" from the current folder (and its parents) and puts every
# KEY=value pair into os.environ. Existing environment variables win, so a key
# exported in your terminal overrides the one in .env.
load_dotenv()


# -----------------------------------------------------------------------------
# describe_secret()
# -----------------------------------------------------------------------------
# Safely reports whether a secret is loaded WITHOUT printing it.
# Never print a full API key: terminal output ends up in screenshots, screen
# recordings and logs.
#
#   Input : name (str) — the environment variable to check, e.g. "OPENAI_API_KEY"
#   Output: (str) a human-friendly status line
def describe_secret(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        return f"{name} is missing ❌  → copy .env.example to .env and add your key"
    # Show only the first 8 characters, which identify the key TYPE
    # (e.g. 'sk-proj-' for an OpenAI project key) but are useless to an attacker.
    return f"{name} loaded ✅  (starts with {value[:8]!r}, {len(value)} characters)"


# ↑ Examples:
#   describe_secret("OPENAI_API_KEY") → "OPENAI_API_KEY loaded ✅  (starts with 'sk-proj-', 164 characters)"
#   describe_secret("NOT_A_REAL_VAR") → "NOT_A_REAL_VAR is missing ❌  → copy .env.example ..."


def main() -> None:
    print("Hello from Future with Param — AI Harness!")

    # Verify the key that scripts 02 and 04 need.
    print(describe_secret("GROQ_API_KEY" if os.environ.get("GROQ_API_KEY") else "OPENAI_API_KEY"))

    # LangSmith tracing is optional. When LANGSMITH_TRACING=true and
    # LANGSMITH_API_KEY are set, every LangChain call is recorded automatically
    # (no code changes needed) and you can inspect it at https://smith.langchain.com
    tracing = os.environ.get("LANGSMITH_TRACING", "(not set)")
    hint = (
        "→ tracing is optional, see Script 02"
        if tracing != "true"
        else "→ runs will appear in LangSmith"
    )
    print(f"LANGSMITH_TRACING = {tracing} {hint}")


if __name__ == "__main__":
    main()