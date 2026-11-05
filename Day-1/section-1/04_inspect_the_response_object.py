import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate  # noqa: E402

from sample_data import PERSON_INFORMATION  # noqa: E402

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")

PROFILE_TEMPLATE = """Here is a short biography:
{information}

Using only this text, give me:
1. A one-line headline about this person
2. A three-bullet summary a 12-year-old would understand
3. One question you would ask them in an interview
"""


# -----------------------------------------------------------------------------
# make_llm()
# -----------------------------------------------------------------------------
# Returns either a cloud or a local chat model. Both have the SAME interface,
# which is why the rest of the script does not care which one it got.
#
#   Input : use_local (bool)
#   Output: (llm, model_name)
def make_llm(use_local: bool):
    if use_local:
        from langchain_ollama import ChatOllama

        return ChatOllama(temperature=0, model=OLLAMA_MODEL), OLLAMA_MODEL

    from langchain_openai import ChatOpenAI

    return (
        ChatOpenAI(
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model=OPENAI_MODEL,
        ),
        OPENAI_MODEL,
    )


# ↑ Example:
#   make_llm(use_local=False) → (ChatOpenAI(model="gpt-5.6-sol"), "gpt-5.6-sol")
#   make_llm(use_local=True)  → (ChatOllama(model="gemma3:270m"), "gemma3:270m")


# -----------------------------------------------------------------------------
# inspect_ai_message()
# -----------------------------------------------------------------------------
# Prints the most useful fields of an AIMessage, one per line.
#
#   Input : response (AIMessage)
#   Output: None (prints to the terminal)
def inspect_ai_message(response) -> None:
    preview = response.text[:80].replace("\n", "\\n")
    print(f"type(response)        : {type(response).__name__}")
    print(f"response.type         : {response.type}")
    print(f"response.text[:80]    : {preview}...")
    print(f"response.tool_calls   : {response.tool_calls}")
    # usage_metadata is LangChain's provider-independent token count.
    print(f"usage_metadata        : {response.usage_metadata}")
    # response_metadata is provider-specific (OpenAI and Ollama differ).
    print(f"response_metadata keys: {list(response.response_metadata.keys())}")
    print(
        f"model_name            : {response.response_metadata.get('model_name') or response.response_metadata.get('model')}"
    )
    print(f"response.id           : {response.id}")
    kind = type(response.content).__name__
    note = (
        "  (Responses-API content blocks) — that's why we use .text"
        if kind == "list"
        else ""
    )
    print(f"content is a          : {kind}{note}")


# ↑ Example (abridged):
#   inspect_ai_message(AIMessage(content="Hi", usage_metadata={...}))
#   → type(response)        : AIMessage
#     response.type         : ai
#     response.text[:80]    : Hi...


def main() -> None:
    use_local = "--local" in sys.argv
    llm, model_name = make_llm(use_local)
    print(f"Calling {model_name} ...\n")

    chain = PromptTemplate.from_template(PROFILE_TEMPLATE) | llm
    response = chain.invoke({"information": PERSON_INFORMATION})

    inspect_ai_message(response)


if __name__ == "__main__":
    main()