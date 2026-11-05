import os

from dotenv import load_dotenv

# Load OPENAI_API_KEY (and optional LANGSMITH_* settings) from .env.
load_dotenv()

from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from sample_data import PERSON_INFORMATION  # noqa: E402

# Which OpenAI model to use. Change it in .env (OPENAI_MODEL=...) — no code edit.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")

PROFILE_TEMPLATE = """Here is a short biography:
{information}

Using only this text, give me:
1. A one-line headline about this person
2. A three-bullet summary a 12-year-old would understand
3. One question you would ask them in an interview
"""


# -----------------------------------------------------------------------------
# build_profile_chain()
# -----------------------------------------------------------------------------
# Creates the chain  PromptTemplate | ChatOpenAI.
#
#   Input : model_name (str) e.g. "gpt-5.6-sol"
#   Output: a Runnable. Call it with chain.invoke({"information": "..."})
def build_profile_chain(model_name: str):
    # input_variables lists every {placeholder} the template expects.
    profile_prompt_template = PromptTemplate(
        input_variables=["information"], template=PROFILE_TEMPLATE
    )

    # temperature=0 asks for the most deterministic answer the model can give.
    # NOTE: GPT-5.x reasoning models only support their default temperature;
    # LangChain quietly drops the setting for them, so this line is safe for
    # every model (and it does matter for local models in Script 03).
    llm = ChatOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model=OPENAI_MODEL,
    )

    # The pipe operator builds a "RunnableSequence": prompt first, then model.
    return profile_prompt_template | llm


# ↑ Example:
#   chain = build_profile_chain("gpt-5.6-sol")
#   chain.invoke({"information": "Sachin Tendulkar is a former Indian cricketer ..."})
#   → AIMessage(content=..., response_metadata={"model_name": "gpt-5.6-sol", ...})


def main() -> None:
    print("Hello from Future with Param — AI Harness!")
    print(f"Model: {OPENAI_MODEL}\n")

    chain = build_profile_chain(OPENAI_MODEL)

    # invoke() runs the whole chain once. The dict keys must match the
    # template's placeholders ({information}).
    response = chain.invoke(input={"information": PERSON_INFORMATION})

    # `response` is an AIMessage (see Script 04 for everything inside it).
    # `.text` returns the answer as a plain string. We use `.text` instead of
    # `.content` because newer OpenAI models reply through the Responses API,
    # where `.content` may be a LIST of content blocks rather than a string.
    print(response.text)


if __name__ == "__main__":
    main()